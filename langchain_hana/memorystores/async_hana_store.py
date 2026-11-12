"""A native-async LangGraph memory store backed by SAP HANA Cloud.

``AsyncHanaStore`` is the asyncio counterpart of
:class:`~langchain_hana.memorystores.hana_store.HanaStore`. It drives an ``hdbcli``
async connection (``dbapi.AsyncConnection``), awaiting every
``execute``/``fetch``/``commit``. It shares the schema, SQL builders and pure helpers
with the sync store via :class:`BaseHanaStore`.

``hdbcli`` materializes this store's NCLOB (``VALUE``) and NVARCHAR
(``KEY``/``PREFIX``) columns directly as ``str`` on both sync and async connections,
so rows are consumed as-is with no per-column decoding.

Unlike the sync store, ``setup()`` is a coroutine and is **not** called from
``__init__`` — the caller must ``await store.setup()`` once before use. The
synchronous ``BaseStore`` methods (``get``/``put``/``search``/``list_namespaces``)
are not supported on this class; use the ``a``-prefixed coroutines instead.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Iterable
from contextlib import asynccontextmanager
from typing import Any, cast

from hdbcli import dbapi
from langgraph.store.base import (
    GetOp,
    IndexConfig,
    ListNamespacesOp,
    Op,
    PutOp,
    Result,
    SearchOp,
    TTLConfig,
)

from langchain_hana.memorystores.base_hana_store import (
    _DEFAULT_MIGRATIONS_TABLE,
    _DEFAULT_STORE_TABLE,
    _DEFAULT_VECTORS_TABLE,
    BaseHanaStore,
)


class AsyncHanaStore(BaseHanaStore):
    """Store LangGraph long-term memory in SAP HANA Cloud (native asyncio).

    Args:
        connection: An open ``hdbcli`` async connection. Create it with async mode
            enabled, e.g. ``conn = await dbapi.async_connect(...)``.
        index: Optional semantic-search configuration (``dims``/``embed``/``fields``).
        ttl: Optional TTL configuration (expiry + refresh-on-read + sweep).
        schema_name: Optional schema; defaults to the connection's current schema.
        migrations_table: Name of the migrations (version ledger) table.
        store_table: Name of the key-value store table.
        vectors_table: Name of the per-field embeddings table.

    Example:
        >>> from hdbcli import dbapi
        >>> from langchain_hana.memorystores import AsyncHanaStore
        >>> conn = await dbapi.async_connect(address=..., port=..., user=...)
        >>> store = AsyncHanaStore(conn)
        >>> await store.setup()
        >>> await store.aput(("users", "123"), "prefs", {"theme": "dark"})
    """

    def __init__(
        self,
        connection: dbapi.AsyncConnection,
        *,
        index: IndexConfig | None = None,
        ttl: TTLConfig | None = None,
        schema_name: str = "",
        migrations_table: str = _DEFAULT_MIGRATIONS_TABLE,
        store_table: str = _DEFAULT_STORE_TABLE,
        vectors_table: str = _DEFAULT_VECTORS_TABLE,
    ) -> None:
        super().__init__(
            index=index,
            ttl=ttl,
            schema_name=schema_name,
            migrations_table=migrations_table,
            store_table=store_table,
            vectors_table=vectors_table,
        )
        self.connection = connection
        # setautocommit is synchronous at runtime, like cursor() below.
        self.connection.setautocommit(False)
        self.lock = asyncio.Lock()
        self._sweeper_task: asyncio.Task | None = None

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #
    async def setup(self) -> None:
        """Apply any not-yet-applied schema migrations (idempotent).

        Must be awaited once before the store is used (it is not called from
        ``__init__``).
        """
        migrations = self._migrations()
        cur = self._new_cursor()
        try:
            await cur.execute(self._migrations_exist_sql, self._migrations_exist_params)
            row = await cur.fetchone()
            if not (row and int(row[0]) > 0):
                await cur.execute(migrations[0])
                await cur.execute(self._insert_version_sql, (0,))
            await cur.execute(f'SELECT MAX("V") FROM {self._migrations_ref}')
            row = await cur.fetchone()
            version = row[0] if row and row[0] is not None else -1
            for v in range(version + 1, len(migrations)):
                await cur.execute(migrations[v])
                await cur.execute(self._insert_version_sql, (v,))
            if self.index_config is not None and not await self._table_exists(cur):
                await cur.execute(self._vectors_ddl())
            await self.connection.commit()
        finally:
            await cur.close()

    async def _table_exists(self, cur: dbapi.AsyncCursor) -> bool:
        """Whether the vectors table already exists in the target schema."""
        if self.schema_name:
            sql = (
                "SELECT COUNT(*) FROM SYS.TABLES "
                "WHERE SCHEMA_NAME = ? AND TABLE_NAME = ?"
            )
            params: tuple[Any, ...] = (self.schema_name, self.vectors_table)
        else:
            sql = (
                "SELECT COUNT(*) FROM SYS.TABLES "
                "WHERE SCHEMA_NAME = CURRENT_SCHEMA AND TABLE_NAME = ?"
            )
            params = (self.vectors_table,)
        await cur.execute(sql, params)
        row = await cur.fetchone()
        return bool(row and int(row[0]) > 0)

    def _new_cursor(self) -> dbapi.AsyncCursor:
        """Open an async cursor.

        ``AsyncConnection.cursor()`` is synchronous at runtime despite the shipped
        type stub annotating it as a coroutine; the cast reflects the real behavior.
        """
        return cast(dbapi.AsyncCursor, self.connection.cursor())

    @asynccontextmanager
    async def _cursor(
        self, transaction: bool = True
    ) -> AsyncGenerator[dbapi.AsyncCursor]:
        """Yield a cursor under the store lock, committing/rolling back as a unit."""
        async with self.lock:
            cur = self._new_cursor()
            try:
                yield cur
            except BaseException:
                if transaction:
                    await self.connection.rollback()
                raise
            else:
                if transaction:
                    await self.connection.commit()
            finally:
                await cur.close()

    # ------------------------------------------------------------------ #
    # Batch entrypoints
    # ------------------------------------------------------------------ #
    def batch(self, ops: Iterable[Op]) -> list[Result]:
        raise NotImplementedError(
            "AsyncHanaStore does not support synchronous batch(); use the async "
            "methods (aget/aput/asearch/adelete/alist_namespaces/abatch)."
        )

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        """Execute a batch of operations in a single transaction, preserving order."""
        grouped, num_ops = self._group_ops(ops)
        results: list[Result] = [None] * num_ops
        async with self._cursor() as cur:
            if PutOp in grouped:
                await self._abatch_put(cur, grouped[PutOp])  # type: ignore[arg-type]
            if GetOp in grouped:
                await self._abatch_get(cur, grouped[GetOp], results)  # type: ignore[arg-type]
            if SearchOp in grouped:
                await self._abatch_search(cur, grouped[SearchOp], results)  # type: ignore[arg-type]
            if ListNamespacesOp in grouped:
                await self._abatch_list(cur, grouped[ListNamespacesOp], results)  # type: ignore[arg-type]
        return results

    # ------------------------------------------------------------------ #
    # Per-type handlers
    # ------------------------------------------------------------------ #
    async def _abatch_get(
        self,
        cur: dbapi.AsyncCursor,
        get_ops: list[tuple[int, GetOp]],
        results: list[Result],
    ) -> None:
        by_ns: dict[str, list[tuple[int, GetOp]]] = {}
        for idx, op in get_ops:
            by_ns.setdefault(self._namespace_to_text(op.namespace), []).append(
                (idx, op)
            )
        for ns_text, items in by_ns.items():
            namespace = self._decode_ns(ns_text)
            keys = [op.key for _, op in items]
            await cur.execute(self._get_sql(len(keys)), (ns_text, *keys))
            rows = await cur.fetchall()
            found = {row[0]: self._row_to_item(namespace, row) for row in rows}
            for idx, op in items:
                results[idx] = found.get(op.key)
            refresh = [op.key for _, op in items if op.refresh_ttl and op.key in found]
            if self._omit_expired and refresh:
                await cur.execute(
                    self._refresh_ttl_sql(len(refresh)), (ns_text, *refresh)
                )

    async def _abatch_put(
        self, cur: dbapi.AsyncCursor, put_ops: list[tuple[int, PutOp]]
    ) -> None:
        deduped = self._dedupe_put_ops(put_ops)
        index_rows: list[tuple[str, str, str]] = []
        index_texts: list[str] = []
        for (prefix, key), op in deduped.items():
            if op.value is None:
                await cur.execute(self._delete_store_sql, (prefix, key))
                if self.index_config is not None:
                    await cur.execute(self._delete_vectors_sql, (prefix, key))
                continue
            await cur.execute(
                self._merge_store_sql, self._merge_store_params(prefix, key, op)
            )
            if self.index_config is not None:
                await cur.execute(self._delete_vectors_sql, (prefix, key))
                for field, text in self._extract_index_texts(op):
                    index_rows.append((prefix, key, field))
                    index_texts.append(text)
        if index_texts and self.embeddings is not None:
            vectors = await self.embeddings.aembed_documents(index_texts)
            params = [
                (*index_rows[i], self._serialize_vector(vectors[i]))
                for i in range(len(index_rows))
            ]
            await cur.executemany(self._insert_vector_sql, params)

    async def _abatch_search(
        self,
        cur: dbapi.AsyncCursor,
        search_ops: list[tuple[int, SearchOp]],
        results: list[Result],
    ) -> None:
        for idx, op in search_ops:
            use_vector = bool(op.query) and self.index_config is not None
            if use_vector:
                assert self.embeddings is not None
                query_vec = await self.embeddings.aembed_query(op.query)  # type: ignore[arg-type]
                sql, params = self._vector_search_sql(op.namespace_prefix)
                await cur.execute(sql, (self._vector_to_str(query_vec), *params))
                rows = await cur.fetchall()
                candidates = [self._row_to_search_item(row) for row in rows]
            else:
                simple = op.filter is None
                sql, params = self._search_sql(
                    op.namespace_prefix, simple, op.limit, op.offset
                )
                await cur.execute(sql, tuple(params))
                rows = await cur.fetchall()
                candidates = [self._row_to_search_item(row) for row in rows]
                if simple:
                    results[idx] = candidates
                    await self._arefresh_search_ttl(cur, op, candidates)
                    continue
            matched = [
                item
                for item in candidates
                if self._matches_filter(item.value, op.filter)
            ]
            page = matched[op.offset : op.offset + op.limit]
            results[idx] = page
            await self._arefresh_search_ttl(cur, op, page)

    async def _arefresh_search_ttl(
        self, cur: dbapi.AsyncCursor, op: SearchOp, items: list[Any]
    ) -> None:
        if not (self._omit_expired and op.refresh_ttl and items):
            return
        by_ns: dict[str, list[str]] = {}
        for item in items:
            by_ns.setdefault(self._namespace_to_text(item.namespace), []).append(
                item.key
            )
        for ns_text, keys in by_ns.items():
            await cur.execute(self._refresh_ttl_sql(len(keys)), (ns_text, *keys))

    async def _abatch_list(
        self,
        cur: dbapi.AsyncCursor,
        list_ops: list[tuple[int, ListNamespacesOp]],
        results: list[Result],
    ) -> None:
        await cur.execute(self._list_namespaces_sql)
        rows = await cur.fetchall()
        prefixes = [row[0] for row in rows]
        for idx, op in list_ops:
            results[idx] = self._filter_namespaces(prefixes, op)

    # ------------------------------------------------------------------ #
    # TTL sweeping
    # ------------------------------------------------------------------ #
    async def asweep_ttl(self) -> int:
        """Delete expired items (and orphaned embeddings). Returns rows removed."""
        async with self._cursor() as cur:
            await cur.execute(self._sweep_store_sql)
            deleted = cur.rowcount
            if self.index_config is not None:
                await cur.execute(self._sweep_vectors_sql)
        return deleted

    def start_ttl_sweeper(self) -> None:
        """Start a background task that periodically sweeps expired items.

        No-op unless ``ttl`` was configured with a ``sweep_interval_minutes``.
        """
        if self.ttl_config is None:
            return
        interval = self.ttl_config.get("sweep_interval_minutes")
        if not interval:
            return
        if self._sweeper_task is not None and not self._sweeper_task.done():
            return
        seconds = float(interval) * 60

        async def _run() -> None:
            while True:
                await asyncio.sleep(seconds)
                try:
                    await self.asweep_ttl()
                except Exception:  # noqa: BLE001 - best-effort background sweep
                    pass

        self._sweeper_task = asyncio.create_task(_run())

    async def stop_ttl_sweeper(self) -> None:
        """Cancel the background sweeper task if running."""
        if self._sweeper_task is not None:
            self._sweeper_task.cancel()
            try:
                await self._sweeper_task
            except asyncio.CancelledError:
                pass
            self._sweeper_task = None

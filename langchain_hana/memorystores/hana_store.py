"""A synchronous LangGraph memory store backed by SAP HANA Cloud.

``HanaStore`` persists long-term, cross-thread memory (hierarchical key-value items
with optional TTL expiry and optional semantic/vector search) in SAP HANA Cloud,
following the design of langgraph's **Postgres** store. The schema, SQL builders and
pure helpers are shared with the native-async ``AsyncHanaStore`` via
:class:`BaseHanaStore`.

Vector search uses *external* embeddings (a LangChain ``Embeddings`` instance, a
sync/async embedding function, or a ``"provider:model"`` string) and HANA
``REAL_VECTOR`` columns with ``COSINE_SIMILARITY``. HANA *internal* embeddings
(``VECTOR_EMBEDDING``) are not wired up in this version.
"""

from __future__ import annotations

import threading
from collections.abc import Generator, Iterable
from contextlib import contextmanager
from typing import Any

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
from langchain_hana.utils import _table_exists


class HanaStore(BaseHanaStore):
    """Store LangGraph long-term memory in SAP HANA Cloud (synchronous).

    Args:
        connection: An open ``hdbcli`` database connection.
        index: Optional semantic-search configuration (``dims``/``embed``/``fields``).
        ttl: Optional TTL configuration (expiry + refresh-on-read + sweep).
        schema_name: Optional schema; defaults to the connection's current schema.
        migrations_table: Name of the migrations (version ledger) table.
        store_table: Name of the key-value store table.
        vectors_table: Name of the per-field embeddings table.

    Example:
        >>> from hdbcli import dbapi
        >>> from langchain_hana.memorystores import HanaStore
        >>> conn = dbapi.connect(address=..., port=..., user=..., password=...)
        >>> store = HanaStore(conn)
        >>> store.put(("users", "123"), "prefs", {"theme": "dark"})
        >>> store.get(("users", "123"), "prefs").value
        {'theme': 'dark'}
    """

    def __init__(
        self,
        connection: dbapi.Connection,
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
        # Disable autocommit so _cursor() controls the transaction boundary and
        # multi-statement batches are atomic.
        self.connection.setautocommit(False)
        self.lock = threading.Lock()
        self._sweeper_stop: threading.Event | None = None
        self._sweeper_thread: threading.Thread | None = None
        self.setup()

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #
    def setup(self) -> None:
        """Apply any not-yet-applied schema migrations (idempotent).

        A migrations version ledger records the highest applied migration index; only
        later migrations run. The vectors table is created on demand when an ``index``
        configuration is present.
        """
        migrations = self._migrations()
        cur = self.connection.cursor()
        try:
            if not _table_exists(
                self.connection, self.migrations_table, self.schema_name
            ):
                cur.execute(migrations[0])
                cur.execute(self._insert_version_sql, (0,))
            cur.execute(f'SELECT MAX("V") FROM {self._migrations_ref}')
            row = cur.fetchone() if cur.has_result_set() else None
            version = row[0] if row and row[0] is not None else -1
            for v in range(version + 1, len(migrations)):
                cur.execute(migrations[v])
                cur.execute(self._insert_version_sql, (v,))
            if self.index_config is not None and not _table_exists(
                self.connection, self.vectors_table, self.schema_name
            ):
                cur.execute(self._vectors_ddl())
            self.connection.commit()
        finally:
            cur.close()

    @contextmanager
    def _cursor(self, transaction: bool = True) -> Generator[dbapi.Cursor]:
        """Yield a cursor under the store lock, committing/rolling back as a unit."""
        with self.lock:
            cur = self.connection.cursor()
            try:
                yield cur
            except BaseException:
                if transaction:
                    self.connection.rollback()
                raise
            else:
                if transaction:
                    self.connection.commit()
            finally:
                cur.close()

    # ------------------------------------------------------------------ #
    # Batch entrypoints
    # ------------------------------------------------------------------ #
    def batch(self, ops: Iterable[Op]) -> list[Result]:
        """Execute a batch of operations in a single transaction, preserving order."""
        grouped, num_ops = self._group_ops(ops)
        results: list[Result] = [None] * num_ops
        with self._cursor() as cur:
            # Puts first so a get/search later in the same batch sees them.
            if PutOp in grouped:
                self._batch_put(cur, grouped[PutOp])  # type: ignore[arg-type]
            if GetOp in grouped:
                self._batch_get(cur, grouped[GetOp], results)  # type: ignore[arg-type]
            if SearchOp in grouped:
                self._batch_search(cur, grouped[SearchOp], results)  # type: ignore[arg-type]
            if ListNamespacesOp in grouped:
                self._batch_list(cur, grouped[ListNamespacesOp], results)  # type: ignore[arg-type]
        return results

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        """Run :meth:`batch` in a worker thread so async callers don't block."""
        import asyncio

        return await asyncio.get_running_loop().run_in_executor(
            None, self.batch, list(ops)
        )

    # ------------------------------------------------------------------ #
    # Per-type handlers
    # ------------------------------------------------------------------ #
    def _batch_get(
        self,
        cur: dbapi.Cursor,
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
            cur.execute(self._get_sql(len(keys)), (ns_text, *keys))
            rows = cur.fetchall() if cur.has_result_set() else []
            found = {row[0]: self._row_to_item(namespace, row) for row in rows}
            for idx, op in items:
                results[idx] = found.get(op.key)
            refresh = [op.key for _, op in items if op.refresh_ttl and op.key in found]
            if self._omit_expired and refresh:
                cur.execute(self._refresh_ttl_sql(len(refresh)), (ns_text, *refresh))

    def _batch_put(self, cur: dbapi.Cursor, put_ops: list[tuple[int, PutOp]]) -> None:
        deduped = self._dedupe_put_ops(put_ops)
        index_rows: list[tuple[str, str, str]] = []
        index_texts: list[str] = []
        for (prefix, key), op in deduped.items():
            if op.value is None:
                cur.execute(self._delete_store_sql, (prefix, key))
                if self.index_config is not None:
                    cur.execute(self._delete_vectors_sql, (prefix, key))
                continue
            cur.execute(
                self._merge_store_sql, self._merge_store_params(prefix, key, op)
            )
            if self.index_config is not None:
                # Replace any previous embeddings for this item.
                cur.execute(self._delete_vectors_sql, (prefix, key))
                for field, text in self._extract_index_texts(op):
                    index_rows.append((prefix, key, field))
                    index_texts.append(text)
        if index_texts and self.embeddings is not None:
            vectors = self.embeddings.embed_documents(index_texts)
            params = [
                (*index_rows[i], self._serialize_vector(vectors[i]))
                for i in range(len(index_rows))
            ]
            cur.executemany(self._insert_vector_sql, params)

    def _batch_search(
        self,
        cur: dbapi.Cursor,
        search_ops: list[tuple[int, SearchOp]],
        results: list[Result],
    ) -> None:
        for idx, op in search_ops:
            use_vector = bool(op.query) and self.index_config is not None
            if use_vector:
                assert self.embeddings is not None
                query_vec = self.embeddings.embed_query(op.query)  # type: ignore[arg-type]
                sql, params = self._vector_search_sql(op.namespace_prefix)
                cur.execute(sql, (self._vector_to_str(query_vec), *params))
                rows = cur.fetchall() if cur.has_result_set() else []
                candidates = [self._row_to_search_item(row) for row in rows]
            else:
                simple = op.filter is None
                sql, params = self._search_sql(
                    op.namespace_prefix, simple, op.limit, op.offset
                )
                cur.execute(sql, tuple(params))
                rows = cur.fetchall() if cur.has_result_set() else []
                candidates = [self._row_to_search_item(row) for row in rows]
                if simple:
                    results[idx] = candidates
                    self._refresh_search_ttl(cur, op, candidates)
                    continue
            matched = [
                item
                for item in candidates
                if self._matches_filter(item.value, op.filter)
            ]
            page = matched[op.offset : op.offset + op.limit]
            results[idx] = page
            self._refresh_search_ttl(cur, op, page)

    def _refresh_search_ttl(
        self, cur: dbapi.Cursor, op: SearchOp, items: list[Any]
    ) -> None:
        if not (self._omit_expired and op.refresh_ttl and items):
            return
        by_ns: dict[str, list[str]] = {}
        for item in items:
            by_ns.setdefault(self._namespace_to_text(item.namespace), []).append(
                item.key
            )
        for ns_text, keys in by_ns.items():
            cur.execute(self._refresh_ttl_sql(len(keys)), (ns_text, *keys))

    def _batch_list(
        self,
        cur: dbapi.Cursor,
        list_ops: list[tuple[int, ListNamespacesOp]],
        results: list[Result],
    ) -> None:
        cur.execute(self._list_namespaces_sql)
        rows = cur.fetchall() if cur.has_result_set() else []
        prefixes = [row[0] for row in rows]
        for idx, op in list_ops:
            results[idx] = self._filter_namespaces(prefixes, op)

    # ------------------------------------------------------------------ #
    # TTL sweeping
    # ------------------------------------------------------------------ #
    def sweep_ttl(self) -> int:
        """Delete expired items (and their orphaned embeddings); returns row count."""
        with self._cursor() as cur:
            cur.execute(self._sweep_store_sql)
            deleted = cur.rowcount
            if self.index_config is not None:
                cur.execute(self._sweep_vectors_sql)
        return deleted

    def start_ttl_sweeper(self) -> None:
        """Start a background daemon thread that periodically sweeps expired items.

        No-op unless ``ttl`` was configured with a ``sweep_interval_minutes``.
        """
        if self.ttl_config is None:
            return
        interval = self.ttl_config.get("sweep_interval_minutes")
        if not interval:
            return
        if self._sweeper_thread is not None and self._sweeper_thread.is_alive():
            return
        self._sweeper_stop = threading.Event()
        seconds = float(interval) * 60

        def _run() -> None:
            assert self._sweeper_stop is not None
            while not self._sweeper_stop.wait(seconds):
                try:
                    self.sweep_ttl()
                except Exception:  # noqa: BLE001 - best-effort background sweep
                    pass

        self._sweeper_thread = threading.Thread(target=_run, daemon=True)
        self._sweeper_thread.start()

    def stop_ttl_sweeper(self, timeout: float | None = None) -> None:
        """Signal the background sweeper to stop and join it."""
        if self._sweeper_stop is not None:
            self._sweeper_stop.set()
        if self._sweeper_thread is not None:
            self._sweeper_thread.join(timeout)
            self._sweeper_thread = None

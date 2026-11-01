"""A native-async LangGraph checkpointer backed by SAP HANA Cloud.

``AsyncHanaCheckpointSaver`` is the asyncio counterpart of
:class:`~langchain_hana.checkpointers.hana_checkpoint_saver.HanaCheckpointSaver`.
It drives an ``hdbcli`` async connection (``dbapi.AsyncConnection``), awaiting every
``execute``/``fetch``/``commit``. It shares the schema, SQL builders and serialization
helpers with the sync saver via :class:`BaseHanaCheckpointSaver`.

``hdbcli`` returns LOB columns on an async connection as ``dbapi.AsyncLob`` objects
whose ``read()`` is a coroutine, so the read path awaits each LOB before the shared
(synchronous) deserializers run.

Unlike the sync saver, ``setup()`` is a coroutine and is **not** called from
``__init__`` — the caller must ``await saver.setup()`` once before use.
"""

from __future__ import annotations

import asyncio
import json
import warnings
from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Any, cast

from hdbcli import dbapi
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    SerializerProtocol,
    get_checkpoint_id,
    get_checkpoint_metadata,
)

from langchain_hana.checkpointers.base_hana_checkpoint_saver import (
    _DEFAULT_BLOBS_TABLE,
    _DEFAULT_CHECKPOINTS_TABLE,
    _DEFAULT_MIGRATIONS_TABLE,
    _DEFAULT_WRITES_TABLE,
    BaseHanaCheckpointSaver,
)


class AsyncHanaCheckpointSaver(BaseHanaCheckpointSaver):
    """Store LangGraph checkpoints in SAP HANA Cloud (native asyncio).

    Args:
        connection: An open ``hdbcli`` async connection. Create it with async mode
            enabled, e.g.
            ``dbapi.set_async_mode(True); conn = await dbapi.connect(...)``.
        serde: Optional serializer. Defaults to langgraph's ``JsonPlusSerializer``.
        schema_name: Optional schema; defaults to the connection's current schema.
        migrations_table: Name of the migrations (version ledger) table.
        checkpoints_table: Name of the checkpoints table.
        blobs_table: Name of the per-channel blobs table.
        writes_table: Name of the pending-writes table.

    Example:
        >>> from hdbcli import dbapi
        >>> from langchain_hana.checkpointers import AsyncHanaCheckpointSaver
        >>> dbapi.set_async_mode(True)
        >>> conn = await dbapi.connect(address=..., port=..., user=..., password=...)
        >>> checkpointer = AsyncHanaCheckpointSaver(conn)
        >>> await checkpointer.setup()
        >>> graph = builder.compile(checkpointer=checkpointer)
        >>> await graph.ainvoke(inputs, {"configurable": {"thread_id": "1"}})
    """

    def __init__(
        self,
        connection: dbapi.AsyncConnection,
        *,
        serde: SerializerProtocol | None = None,
        schema_name: str = "",
        migrations_table: str = _DEFAULT_MIGRATIONS_TABLE,
        checkpoints_table: str = _DEFAULT_CHECKPOINTS_TABLE,
        blobs_table: str = _DEFAULT_BLOBS_TABLE,
        writes_table: str = _DEFAULT_WRITES_TABLE,
    ) -> None:
        super().__init__(
            serde=serde,
            schema_name=schema_name,
            migrations_table=migrations_table,
            checkpoints_table=checkpoints_table,
            blobs_table=blobs_table,
            writes_table=writes_table,
        )
        self.connection = connection
        # Disable autocommit so _cursor() controls the transaction boundary and
        # multi-statement writes are atomic. setautocommit/getautocommit are
        # synchronous at runtime, like cursor() above.
        if self.connection.getautocommit():
            warnings.warn(
                "AsyncHanaCheckpointSaver is disabling autocommit on the provided "
                "connection so checkpoint writes commit atomically.",
                stacklevel=2,
            )
        self.connection.setautocommit(False)
        self.lock = asyncio.Lock()

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #
    async def setup(self) -> None:
        """Apply any not-yet-applied schema migrations.

        Must be awaited once before the saver is used (it is not called from
        ``__init__``). Idempotent: a migrations version ledger records the highest
        applied migration index and only later migrations are run.
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
            await self.connection.commit()
        finally:
            await cur.close()

    def _new_cursor(self) -> dbapi.AsyncCursor:
        """Open an async cursor.

        ``AsyncConnection.cursor()`` is synchronous at runtime (it returns the cursor
        directly, matching SAP's documented usage), but the shipped type stub wrongly
        annotates it as a coroutine; the cast reflects the real behavior.
        """
        return cast(dbapi.AsyncCursor, self.connection.cursor())

    @asynccontextmanager
    async def _cursor(
        self, transaction: bool = True
    ) -> AsyncGenerator[dbapi.AsyncCursor]:
        """Yield a cursor under the saver lock.

        When ``transaction`` is set, commit once the body completes and roll back
        if it raises, so a failed multi-statement write leaves no partial state.
        """
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

    @staticmethod
    async def _materialize(row: Any) -> tuple[Any, ...]:
        """Await any ``AsyncLob`` columns so a row carries only ``bytes``/``str``.

        The shared deserializers (``_load_blobs``/``_load_pending_writes``/
        ``loads_typed``) are synchronous and expect materialized values, so LOB
        columns must be read here first.
        """
        out: list[Any] = []
        for value in row:
            if isinstance(value, dbapi.AsyncLob):
                out.append(await value.read())
            else:
                out.append(value)
        return tuple(out)

    @classmethod
    async def _materialize_all(cls, rows: Any) -> list[tuple[Any, ...]]:
        return [await cls._materialize(row) for row in rows]

    # ------------------------------------------------------------------ #
    # Reads
    # ------------------------------------------------------------------ #
    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Fetch a checkpoint by id, or the latest for the thread, with its writes."""
        thread_id = str(config["configurable"]["thread_id"])
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        async with self._cursor(transaction=False) as cur:
            if checkpoint_id := get_checkpoint_id(config):
                select_by_id_sql = (
                    self._select_checkpoint_sql
                    + 'WHERE "THREAD_ID" = ? AND "CHECKPOINT_NS" = ? '
                    'AND "CHECKPOINT_ID" = ?'
                )
                await cur.execute(
                    select_by_id_sql, (thread_id, checkpoint_ns, checkpoint_id)
                )
            else:
                select_latest_sql = (
                    self._select_checkpoint_sql
                    + 'WHERE "THREAD_ID" = ? AND "CHECKPOINT_NS" = ? '
                    'ORDER BY "CHECKPOINT_ID" DESC LIMIT 1'
                )
                await cur.execute(select_latest_sql, (thread_id, checkpoint_ns))
            raw = await cur.fetchone()
            if not raw:
                return None
            (
                thread_id,
                checkpoint_id,
                parent_checkpoint_id,
                serde_type,
                checkpoint,
                metadata,
            ) = await self._materialize(raw)
            final_config: RunnableConfig = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                }
            }
            checkpoint_value = await self._aload_checkpoint(
                cur, thread_id, checkpoint_ns, serde_type, checkpoint
            )
            await cur.execute(
                self._pending_writes_sql(),
                (thread_id, checkpoint_ns, str(checkpoint_id)),
            )
            write_rows = await self._materialize_all(await cur.fetchall())
            metadata_str = self._to_str(metadata)
            return CheckpointTuple(
                final_config,
                checkpoint_value,
                cast(
                    CheckpointMetadata,
                    json.loads(metadata_str) if metadata_str else {},
                ),
                (
                    {
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": parent_checkpoint_id,
                        }
                    }
                    if parent_checkpoint_id
                    else None
                ),
                self._load_pending_writes(write_rows, self.serde),
            )

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """Yield checkpoints scoped by ``config`` (thread/ns/id), newest-first.

        ``filter`` and ``before`` narrow further; a ``None`` config lists all threads.
        """
        where_clause, params = self._build_where_clause(config, filter, before)
        list_checkpoints_sql = self._list_checkpoints_sql(where_clause, limit)
        async with self._cursor(transaction=False) as cur:
            await cur.execute(list_checkpoints_sql, tuple(params))
            checkpoint_rows = await self._materialize_all(await cur.fetchall())
            results: list[CheckpointTuple] = []
            for (
                thread_id,
                checkpoint_ns,
                checkpoint_id,
                parent_checkpoint_id,
                serde_type,
                checkpoint,
                metadata,
            ) in checkpoint_rows:
                checkpoint_value = await self._aload_checkpoint(
                    cur, thread_id, checkpoint_ns, serde_type, checkpoint
                )
                await cur.execute(
                    self._pending_writes_sql(),
                    (thread_id, checkpoint_ns, str(checkpoint_id)),
                )
                write_rows = await self._materialize_all(await cur.fetchall())
                metadata_str = self._to_str(metadata)
                results.append(
                    CheckpointTuple(
                        {
                            "configurable": {
                                "thread_id": thread_id,
                                "checkpoint_ns": checkpoint_ns,
                                "checkpoint_id": checkpoint_id,
                            }
                        },
                        checkpoint_value,
                        cast(
                            CheckpointMetadata,
                            json.loads(metadata_str) if metadata_str else {},
                        ),
                        (
                            {
                                "configurable": {
                                    "thread_id": thread_id,
                                    "checkpoint_ns": checkpoint_ns,
                                    "checkpoint_id": parent_checkpoint_id,
                                }
                            }
                            if parent_checkpoint_id
                            else None
                        ),
                        self._load_pending_writes(write_rows, self.serde),
                    )
                )
        for tuple_ in results:
            yield tuple_

    async def _aload_checkpoint(
        self,
        cur: dbapi.AsyncCursor,
        thread_id: str,
        checkpoint_ns: str,
        serde_type: Any,
        checkpoint: Any,
    ) -> Checkpoint:
        """Deserialize a checkpoint skeleton and merge its per-channel blob values."""
        checkpoint_value = cast(
            Checkpoint,
            self.serde.loads_typed(
                cast(
                    "tuple[str, bytes]",
                    (self._to_str(serde_type), self._to_bytes(checkpoint)),
                )
            ),
        )
        channel_versions = checkpoint_value.get("channel_versions") or {}
        if channel_versions:
            await cur.execute(
                *self._blob_query(thread_id, checkpoint_ns, channel_versions)
            )
            blob_rows = await self._materialize_all(await cur.fetchall())
            checkpoint_value["channel_values"] = {
                **checkpoint_value.get("channel_values", {}),
                **self._load_blobs(blob_rows, self.serde),
            }
        return checkpoint_value

    # ------------------------------------------------------------------ #
    # Writes
    # ------------------------------------------------------------------ #
    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Persist a checkpoint and return its fully-qualified config.

        Non-primitive channel values are offloaded to the blobs table; primitive
        values (``None``/``str``/``int``/``float``/``bool``) stay inline in the
        checkpoint skeleton.
        """
        thread_id = str(config["configurable"]["thread_id"])
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

        copy, blob_values = self._split_checkpoint(checkpoint)
        blob_rows = self._dump_blobs(
            self.serde, thread_id, checkpoint_ns, blob_values, new_versions
        )
        serde_type, serialized_checkpoint = self.serde.dumps_typed(copy)
        serialized_metadata = json.dumps(
            get_checkpoint_metadata(config, metadata), ensure_ascii=False
        )
        async with self._cursor() as cur:
            if blob_rows:
                await cur.executemany(self._upsert_blobs_sql(), blob_rows)
            await cur.execute(
                self._upsert_checkpoint_sql,
                (
                    thread_id,
                    checkpoint_ns,
                    checkpoint["id"],
                    config["configurable"].get("checkpoint_id"),
                    serde_type,
                    serialized_checkpoint,
                    serialized_metadata,
                ),
            )
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store intermediate (pending) writes for a checkpoint.

        Writes to special channels (``WRITES_IDX_MAP``) replace existing rows;
        regular channel writes are inserted only if absent (first-write-wins), so
        a retried task does not clobber already-recorded output.
        """
        thread_id = str(config["configurable"]["thread_id"])
        checkpoint_ns = str(config["configurable"].get("checkpoint_ns", ""))
        checkpoint_id = str(config["configurable"]["checkpoint_id"])
        rows = [
            (
                thread_id,
                checkpoint_ns,
                checkpoint_id,
                task_id,
                task_path,
                WRITES_IDX_MAP.get(channel, idx),
                channel,
                *self.serde.dumps_typed(value),
            )
            for idx, (channel, value) in enumerate(writes)
        ]
        upsert = all(w[0] in WRITES_IDX_MAP for w in writes)
        async with self._cursor() as cur:
            await cur.executemany(self._put_writes_sql(upsert), rows)

    async def adelete_thread(self, thread_id: str) -> None:
        """Delete every checkpoint, blob and write for a thread."""
        async with self._cursor() as cur:
            for sql in self._delete_thread_sql():
                await cur.execute(sql, (str(thread_id),))

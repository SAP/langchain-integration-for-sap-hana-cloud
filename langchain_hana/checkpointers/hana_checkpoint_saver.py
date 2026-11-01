"""A synchronous LangGraph checkpointer backed by SAP HANA Cloud.

``HanaCheckpointSaver`` persists LangGraph checkpoints and pending writes in HANA,
following the design of langgraph's **Postgres** saver (a single synchronous
connection using ``?`` placeholders — the same model as ``hdbcli``). The schema, SQL
builders and serialization helpers are shared with the native-async
``AsyncHanaCheckpointSaver`` via :class:`BaseHanaCheckpointSaver`.
"""

from __future__ import annotations

import json
import threading
import warnings
from collections.abc import Generator, Iterator, Sequence
from contextlib import contextmanager
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
from langchain_hana.utils import _table_exists


class HanaCheckpointSaver(BaseHanaCheckpointSaver):
    """Store LangGraph checkpoints in SAP HANA Cloud (synchronous).

    Args:
        connection: An open ``hdbcli`` database connection.
        serde: Optional serializer. Defaults to langgraph's ``JsonPlusSerializer``.
        schema_name: Optional schema; defaults to the connection's current schema.
        migrations_table: Name of the migrations (version ledger) table.
        checkpoints_table: Name of the checkpoints table.
        blobs_table: Name of the per-channel blobs table.
        writes_table: Name of the pending-writes table.

    Example:
        >>> from hdbcli import dbapi
        >>> from langchain_hana.checkpointers import HanaCheckpointSaver
        >>> conn = dbapi.connect(address=..., port=..., user=..., password=...)
        >>> checkpointer = HanaCheckpointSaver(conn)
        >>> graph = builder.compile(checkpointer=checkpointer)
        >>> graph.invoke(inputs, {"configurable": {"thread_id": "1"}})
    """

    def __init__(
        self,
        connection: dbapi.Connection,
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
        # multi-statement writes are atomic.
        if self.connection.getautocommit():
            warnings.warn(
                "HanaCheckpointSaver is disabling autocommit on the provided "
                "connection so checkpoint writes commit atomically.",
                stacklevel=2,
            )
        self.connection.setautocommit(False)
        self.lock = threading.Lock()
        self.setup()

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #
    def setup(self) -> None:
        """Apply any not-yet-applied schema migrations.

        A migrations (version ledger) table records the highest migration index that
        has been applied; only migrations after it are run, so ``setup`` is idempotent
        and the schema can evolve by appending to the migrations list in future
        releases.
        """
        migrations = self._migrations()
        # DDL auto-commits in HANA; commit once at the end to flush the ledger inserts.
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
            self.connection.commit()
        finally:
            cur.close()

    @contextmanager
    def _cursor(self, transaction: bool = True) -> Generator[dbapi.Cursor]:
        """Yield a cursor under the saver lock.

        When ``transaction`` is set, commit once the body completes and roll back
        if it raises, so a failed multi-statement write leaves no partial state.
        """
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
    # Reads
    # ------------------------------------------------------------------ #
    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Fetch a checkpoint by id, or the latest for the thread, with its writes."""
        thread_id = str(config["configurable"]["thread_id"])
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        with self._cursor(transaction=False) as cur:
            if checkpoint_id := get_checkpoint_id(config):
                select_by_id_sql = (
                    self._select_checkpoint_sql
                    + 'WHERE "THREAD_ID" = ? AND "CHECKPOINT_NS" = ? '
                    'AND "CHECKPOINT_ID" = ?'
                )
                cur.execute(select_by_id_sql, (thread_id, checkpoint_ns, checkpoint_id))
            else:
                select_latest_sql = (
                    self._select_checkpoint_sql
                    + 'WHERE "THREAD_ID" = ? AND "CHECKPOINT_NS" = ? '
                    'ORDER BY "CHECKPOINT_ID" DESC LIMIT 1'
                )
                cur.execute(select_latest_sql, (thread_id, checkpoint_ns))
            row = cur.fetchone() if cur.has_result_set() else None
            if not row:
                return None
            (
                thread_id,
                checkpoint_id,
                parent_checkpoint_id,
                serde_type,
                checkpoint,
                metadata,
            ) = row
            final_config: RunnableConfig = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                }
            }
            checkpoint_value = self._load_checkpoint(
                cur, thread_id, checkpoint_ns, serde_type, checkpoint
            )
            cur.execute(
                self._pending_writes_sql(),
                (thread_id, checkpoint_ns, str(checkpoint_id)),
            )
            write_rows = cur.fetchall() if cur.has_result_set() else []
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

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """Yield checkpoints scoped by ``config`` (thread/ns/id), newest-first.

        ``filter`` and ``before`` narrow further; a ``None`` config lists all threads.
        """
        where_clause, params = self._build_where_clause(config, filter, before)
        list_checkpoints_sql = self._list_checkpoints_sql(where_clause, limit)
        with self._cursor(transaction=False) as cur:
            cur.execute(list_checkpoints_sql, tuple(params))
            checkpoint_rows = cur.fetchall() if cur.has_result_set() else []
            for (
                thread_id,
                checkpoint_ns,
                checkpoint_id,
                parent_checkpoint_id,
                serde_type,
                checkpoint,
                metadata,
            ) in checkpoint_rows:
                checkpoint_value = self._load_checkpoint(
                    cur, thread_id, checkpoint_ns, serde_type, checkpoint
                )
                cur.execute(
                    self._pending_writes_sql(),
                    (thread_id, checkpoint_ns, str(checkpoint_id)),
                )
                write_rows = cur.fetchall() if cur.has_result_set() else []
                metadata_str = self._to_str(metadata)
                yield CheckpointTuple(
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

    def _load_checkpoint(
        self,
        cur: dbapi.Cursor,
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
            cur.execute(*self._blob_query(thread_id, checkpoint_ns, channel_versions))
            blob_rows = cur.fetchall() if cur.has_result_set() else []
            checkpoint_value["channel_values"] = {
                **checkpoint_value.get("channel_values", {}),
                **self._load_blobs(blob_rows, self.serde),
            }
        return checkpoint_value

    # ------------------------------------------------------------------ #
    # Writes
    # ------------------------------------------------------------------ #
    def put(
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
        with self._cursor() as cur:
            if blob_rows:
                cur.executemany(self._upsert_blobs_sql(), blob_rows)
            cur.execute(
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

    def put_writes(
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
        with self._cursor() as cur:
            cur.executemany(self._put_writes_sql(upsert), rows)

    def delete_thread(self, thread_id: str) -> None:
        """Delete every checkpoint, blob and write for a thread."""
        with self._cursor() as cur:
            for sql in self._delete_thread_sql():
                cur.execute(sql, (str(thread_id),))

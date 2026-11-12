"""A synchronous LangGraph checkpointer backed by SAP HANA Cloud.

``HanaCheckpointSaver`` persists LangGraph checkpoints and pending writes in HANA,
following the design of langgraph's **Postgres** saver (a single synchronous
connection using ``?`` placeholders — the same model as ``hdbcli``). The schema, SQL
builders and serialization helpers are shared with the native-async
``AsyncHanaCheckpointSaver`` via :class:`BaseHanaCheckpointSaver`.
"""

from __future__ import annotations

import threading
import warnings
from collections.abc import Generator, Iterator, Sequence
from contextlib import contextmanager
from typing import Any

from hdbcli import dbapi
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    SerializerProtocol,
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
        thread_id, checkpoint_ns = self._thread_ns(config)
        select_sql, select_params = self._get_checkpoint_query(
            config, thread_id, checkpoint_ns
        )
        with self._cursor(transaction=False) as cur:
            cur.execute(select_sql, select_params)
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
            checkpoint_value = self._load_checkpoint(
                cur, thread_id, checkpoint_ns, serde_type, checkpoint
            )
            cur.execute(
                self._pending_writes_sql(),
                (thread_id, checkpoint_ns, str(checkpoint_id)),
            )
            write_rows = cur.fetchall() if cur.has_result_set() else []
            return self._build_checkpoint_tuple(
                thread_id=thread_id,
                checkpoint_ns=checkpoint_ns,
                checkpoint_id=checkpoint_id,
                parent_checkpoint_id=parent_checkpoint_id,
                checkpoint_value=checkpoint_value,
                metadata=metadata,
                write_rows=write_rows,
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
                yield self._build_checkpoint_tuple(
                    thread_id=thread_id,
                    checkpoint_ns=checkpoint_ns,
                    checkpoint_id=checkpoint_id,
                    parent_checkpoint_id=parent_checkpoint_id,
                    checkpoint_value=checkpoint_value,
                    metadata=metadata,
                    write_rows=write_rows,
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
        checkpoint_value, channel_versions = self._decode_checkpoint(
            serde_type, checkpoint
        )
        if channel_versions:
            cur.execute(*self._blob_query(thread_id, checkpoint_ns, channel_versions))
            blob_rows = cur.fetchall() if cur.has_result_set() else []
            self._merge_channel_values(checkpoint_value, blob_rows)
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
        blob_rows, checkpoint_params, next_config = self._prepare_put(
            config, checkpoint, metadata, new_versions
        )
        with self._cursor() as cur:
            if blob_rows:
                cur.executemany(self._upsert_blobs_sql(), blob_rows)
            cur.execute(self._upsert_checkpoint_sql, checkpoint_params)
        return next_config

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
        rows, upsert = self._prepare_put_writes(config, writes, task_id, task_path)
        with self._cursor() as cur:
            cur.executemany(self._put_writes_sql(upsert), rows)

    def delete_thread(self, thread_id: str) -> None:
        """Delete every checkpoint, blob and write for a thread."""
        with self._cursor() as cur:
            for sql in self._delete_thread_sql():
                cur.execute(sql, (str(thread_id),))

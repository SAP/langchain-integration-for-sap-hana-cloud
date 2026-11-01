"""Shared base for the SAP HANA Cloud LangGraph checkpointers.

``BaseHanaCheckpointSaver`` holds everything that does not depend on *how* the
database connection is driven: the schema/DDL, the HANA SQL-dialect query builders,
and the pure serialization helpers. The synchronous ``HanaCheckpointSaver`` and the
native-async ``AsyncHanaCheckpointSaver`` subclass it and add their own connection,
lock, ``setup`` and read/write methods.

Four tables are used:

* a **migrations** table (version ledger) recording which schema migrations have been
  applied, so the schema can evolve safely;
* a **checkpoints** table holding the checkpoint skeleton (metadata, the
  ``channel_versions`` map and any inline primitive channel values);
* a **blobs** table holding each non-primitive channel value keyed by
  ``(thread_id, checkpoint_ns, channel, version)`` — because a channel's version only
  changes when its value changes, identical values are written once and reused across
  checkpoints (de-duplication);
* a **writes** table holding pending (intermediate) writes.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Mapping
from typing import Any, List, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    PendingWrite,
    SerializerProtocol,
    get_checkpoint_id,
)

from langchain_hana.utils import (
    _json_value,
    _sanitize_name,
    _table_ref,
    _validate_identifier,
)

_DEFAULT_MIGRATIONS_TABLE = "LANGGRAPH_CHECKPOINT_MIGRATIONS"
_DEFAULT_CHECKPOINTS_TABLE = "LANGGRAPH_CHECKPOINTS"
_DEFAULT_BLOBS_TABLE = "LANGGRAPH_CHECKPOINT_BLOBS"
_DEFAULT_WRITES_TABLE = "LANGGRAPH_CHECKPOINT_WRITES"

# Bounded identifier lengths keep the composite primary keys within HANA's
# index key-length limit. ``channel`` and ``version`` are part of the blobs table
# primary key, so they are kept shorter than the (non-key) writes ``channel`` column.
_ID_LEN = 256
_NS_LEN = 256
_CHANNEL_LEN = 1024
_BLOB_CHANNEL_LEN = 512
_VERSION_LEN = 256
_TYPE_LEN = 128
_TASK_PATH_LEN = 1024


class BaseHanaCheckpointSaver(BaseCheckpointSaver[str]):
    """Connection-agnostic base: schema, SQL builders and serialization helpers.

    Args:
        serde: Optional serializer. Defaults to langgraph's ``JsonPlusSerializer``.
        schema_name: Optional schema; defaults to the connection's current schema.
        migrations_table: Name of the migrations (version ledger) table.
        checkpoints_table: Name of the checkpoints table.
        blobs_table: Name of the per-channel blobs table.
        writes_table: Name of the pending-writes table.
    """

    def __init__(
        self,
        *,
        serde: SerializerProtocol | None = None,
        schema_name: str = "",
        migrations_table: str = _DEFAULT_MIGRATIONS_TABLE,
        checkpoints_table: str = _DEFAULT_CHECKPOINTS_TABLE,
        blobs_table: str = _DEFAULT_BLOBS_TABLE,
        writes_table: str = _DEFAULT_WRITES_TABLE,
    ) -> None:
        super().__init__(serde=serde)
        self.schema_name = _sanitize_name(schema_name) if schema_name else ""
        self.migrations_table = _sanitize_name(migrations_table)
        self.checkpoints_table = _sanitize_name(checkpoints_table)
        self.blobs_table = _sanitize_name(blobs_table)
        self.writes_table = _sanitize_name(writes_table)
        if not all(
            (
                self.migrations_table,
                self.checkpoints_table,
                self.blobs_table,
                self.writes_table,
            )
        ):
            raise ValueError("Table names must not be empty after sanitization.")

    # ------------------------------------------------------------------ #
    # SQL identifiers / schema
    # ------------------------------------------------------------------ #
    @property
    def _migrations_ref(self) -> str:
        return _table_ref(self.migrations_table, self.schema_name)

    @property
    def _checkpoints_ref(self) -> str:
        return _table_ref(self.checkpoints_table, self.schema_name)

    @property
    def _blobs_ref(self) -> str:
        return _table_ref(self.blobs_table, self.schema_name)

    @property
    def _writes_ref(self) -> str:
        return _table_ref(self.writes_table, self.schema_name)

    def _migrations(self) -> list[str]:
        """Ordered DDL statements. Index 0 creates the migrations ledger itself."""
        migrations_ddl = (
            f'CREATE TABLE {self._migrations_ref} ("V" INTEGER PRIMARY KEY)'
        )
        checkpoints_ddl = (
            f"CREATE TABLE {self._checkpoints_ref} ("
            f'"THREAD_ID" NVARCHAR({_ID_LEN}) NOT NULL, '
            f"\"CHECKPOINT_NS\" NVARCHAR({_NS_LEN}) DEFAULT '' NOT NULL, "
            f'"CHECKPOINT_ID" NVARCHAR({_ID_LEN}) NOT NULL, '
            f'"PARENT_CHECKPOINT_ID" NVARCHAR({_ID_LEN}), '
            f'"TYPE" NVARCHAR({_TYPE_LEN}), '
            '"CHECKPOINT" BLOB, '
            '"METADATA" NCLOB, '
            'PRIMARY KEY ("THREAD_ID", "CHECKPOINT_NS", "CHECKPOINT_ID"))'
        )
        blobs_ddl = (
            f"CREATE TABLE {self._blobs_ref} ("
            f'"THREAD_ID" NVARCHAR({_ID_LEN}) NOT NULL, '
            f"\"CHECKPOINT_NS\" NVARCHAR({_NS_LEN}) DEFAULT '' NOT NULL, "
            f'"CHANNEL" NVARCHAR({_BLOB_CHANNEL_LEN}) NOT NULL, '
            f'"VERSION" NVARCHAR({_VERSION_LEN}) NOT NULL, '
            f'"TYPE" NVARCHAR({_TYPE_LEN}) NOT NULL, '
            '"BLOB" BLOB, '
            'PRIMARY KEY ("THREAD_ID", "CHECKPOINT_NS", "CHANNEL", "VERSION"))'
        )
        writes_ddl = (
            f"CREATE TABLE {self._writes_ref} ("
            f'"THREAD_ID" NVARCHAR({_ID_LEN}) NOT NULL, '
            f"\"CHECKPOINT_NS\" NVARCHAR({_NS_LEN}) DEFAULT '' NOT NULL, "
            f'"CHECKPOINT_ID" NVARCHAR({_ID_LEN}) NOT NULL, '
            f'"TASK_ID" NVARCHAR({_ID_LEN}) NOT NULL, '
            f"\"TASK_PATH\" NVARCHAR({_TASK_PATH_LEN}) DEFAULT '' NOT NULL, "
            '"IDX" INTEGER NOT NULL, '
            f'"CHANNEL" NVARCHAR({_CHANNEL_LEN}) NOT NULL, '
            f'"TYPE" NVARCHAR({_TYPE_LEN}), '
            '"VALUE" BLOB, '
            'PRIMARY KEY ("THREAD_ID", "CHECKPOINT_NS", "CHECKPOINT_ID", '
            '"TASK_ID", "IDX"))'
        )
        return [migrations_ddl, checkpoints_ddl, blobs_ddl, writes_ddl]

    @property
    def _insert_version_sql(self) -> str:
        return f'INSERT INTO {self._migrations_ref} ("V") VALUES (?)'

    @property
    def _migrations_exist_sql(self) -> str:
        """Existence check for the migrations table via ``SYS.TABLES``."""
        if self.schema_name:
            return (
                "SELECT COUNT(*) FROM SYS.TABLES "
                "WHERE SCHEMA_NAME = ? AND TABLE_NAME = ?"
            )
        return (
            "SELECT COUNT(*) FROM SYS.TABLES "
            "WHERE SCHEMA_NAME = CURRENT_SCHEMA AND TABLE_NAME = ?"
        )

    @property
    def _migrations_exist_params(self) -> tuple[Any, ...]:
        if self.schema_name:
            return (self.schema_name, self.migrations_table)
        return (self.migrations_table,)

    # ------------------------------------------------------------------ #
    # Query builders (pure)
    # ------------------------------------------------------------------ #
    @property
    def _select_checkpoint_sql(self) -> str:
        return (
            'SELECT "THREAD_ID", "CHECKPOINT_ID", "PARENT_CHECKPOINT_ID", "TYPE", '
            f'"CHECKPOINT", "METADATA" FROM {self._checkpoints_ref} '
        )

    def _list_checkpoints_sql(self, where_clause: str, limit: int | None) -> str:
        sql = (
            'SELECT "THREAD_ID", "CHECKPOINT_NS", "CHECKPOINT_ID", '
            '"PARENT_CHECKPOINT_ID", "TYPE", "CHECKPOINT", "METADATA" FROM '
            f'{self._checkpoints_ref} {where_clause} ORDER BY "CHECKPOINT_ID" DESC'
        )
        if limit is not None:
            if not isinstance(limit, int) or limit <= 0:
                raise ValueError("Parameter 'limit' must be a positive integer.")
            sql += f" LIMIT {int(limit)}"
        return sql

    def _blob_query(
        self,
        thread_id: str,
        checkpoint_ns: str,
        channel_versions: Mapping[str, Any],
    ) -> tuple[str, tuple[Any, ...]]:
        """SQL + params to fetch the blob rows matching a ``channel_versions`` map."""
        or_clauses = " OR ".join(
            ['("CHANNEL" = ? AND "VERSION" = ?)'] * len(channel_versions)
        )
        sql = (
            f'SELECT "CHANNEL", "TYPE", "BLOB" FROM {self._blobs_ref} '
            f'WHERE "THREAD_ID" = ? AND "CHECKPOINT_NS" = ? AND ({or_clauses})'
        )
        params: list[Any] = [thread_id, checkpoint_ns]
        for channel, version in channel_versions.items():
            params.append(channel)
            params.append(str(version))
        return sql, tuple(params)

    def _pending_writes_sql(self) -> str:
        """SQL to fetch the writes attached to a single checkpoint."""
        return (
            'SELECT "TASK_ID", "CHANNEL", "TYPE", "VALUE", "TASK_PATH", "IDX" FROM '
            f"{self._writes_ref} "
            'WHERE "THREAD_ID" = ? AND "CHECKPOINT_NS" = ? AND "CHECKPOINT_ID" = ?'
        )

    def _upsert_blobs_sql(self) -> str:
        """MERGE that inserts a blob row only if its primary key is absent.

        Identical ``(channel, version)`` values are therefore written once and
        reused across checkpoints (first-write-wins de-duplication).
        """
        cols = '"THREAD_ID", "CHECKPOINT_NS", "CHANNEL", "VERSION", "TYPE", "BLOB"'
        return (
            f"MERGE INTO {self._blobs_ref} t USING ("
            'SELECT ? "THREAD_ID", ? "CHECKPOINT_NS", ? "CHANNEL", ? "VERSION", '
            '? "TYPE", ? "BLOB" FROM DUMMY) s '
            'ON (t."THREAD_ID" = s."THREAD_ID" '
            'AND t."CHECKPOINT_NS" = s."CHECKPOINT_NS" '
            'AND t."CHANNEL" = s."CHANNEL" AND t."VERSION" = s."VERSION") '
            f"WHEN NOT MATCHED THEN INSERT ({cols}) VALUES ("
            's."THREAD_ID", s."CHECKPOINT_NS", s."CHANNEL", s."VERSION", '
            's."TYPE", s."BLOB")'
        )

    @property
    def _upsert_checkpoint_sql(self) -> str:
        return (
            f"UPSERT {self._checkpoints_ref} "
            '("THREAD_ID", "CHECKPOINT_NS", "CHECKPOINT_ID", '
            '"PARENT_CHECKPOINT_ID", "TYPE", "CHECKPOINT", "METADATA") '
            "VALUES (?, ?, ?, ?, ?, ?, ?) WITH PRIMARY KEY"
        )

    def _put_writes_sql(self, upsert: bool) -> str:
        """SQL for ``put_writes``: UPSERT (replace) or MERGE (insert-if-absent)."""
        cols = (
            '"THREAD_ID", "CHECKPOINT_NS", "CHECKPOINT_ID", "TASK_ID", "TASK_PATH", '
            '"IDX", "CHANNEL", "TYPE", "VALUE"'
        )
        if upsert:
            # Replace: special control-channel writes may legitimately overwrite.
            return (
                f"UPSERT {self._writes_ref} ({cols}) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) WITH PRIMARY KEY"
            )
        # Ignore (insert-if-absent): preserve the first write for each pk.
        return (
            f"MERGE INTO {self._writes_ref} t USING ("
            'SELECT ? "THREAD_ID", ? "CHECKPOINT_NS", ? "CHECKPOINT_ID", '
            '? "TASK_ID", ? "TASK_PATH", ? "IDX", ? "CHANNEL", ? "TYPE", '
            '? "VALUE" FROM DUMMY) s '
            'ON (t."THREAD_ID" = s."THREAD_ID" '
            'AND t."CHECKPOINT_NS" = s."CHECKPOINT_NS" '
            'AND t."CHECKPOINT_ID" = s."CHECKPOINT_ID" '
            'AND t."TASK_ID" = s."TASK_ID" AND t."IDX" = s."IDX") '
            f"WHEN NOT MATCHED THEN INSERT ({cols}) VALUES ("
            's."THREAD_ID", s."CHECKPOINT_NS", s."CHECKPOINT_ID", s."TASK_ID", '
            's."TASK_PATH", s."IDX", s."CHANNEL", s."TYPE", s."VALUE")'
        )

    def _delete_thread_sql(self) -> list[str]:
        """DELETE statements removing a thread's rows from every table.

        Each statement takes a single ``thread_id`` parameter. Returned together so
        both the sync and async savers run one loop instead of repeating the SQL.
        """
        return [
            f'DELETE FROM {self._checkpoints_ref} WHERE "THREAD_ID" = ?',
            f'DELETE FROM {self._blobs_ref} WHERE "THREAD_ID" = ?',
            f'DELETE FROM {self._writes_ref} WHERE "THREAD_ID" = ?',
        ]

    @staticmethod
    def _metadata_predicate(
        metadata_filter: dict[str, Any],
    ) -> tuple[List[str], List[Any]]:
        """Build ``JSON_VALUE`` WHERE predicates for a metadata filter.

        Only scalar filter values (and ``None``) are supported, since HANA's
        ``JSON_VALUE`` returns a scalar string.
        """
        predicates: list[str] = []
        param_values: list[Any] = []
        for query_key, query_value in metadata_filter.items():
            _validate_identifier(query_key)
            json_value = _json_value('"METADATA"', query_key)
            if query_value is None:
                predicates.append(f"{json_value} IS NULL")
                continue
            if isinstance(query_value, bool):
                param: Any = "true" if query_value else "false"
            elif isinstance(query_value, (str, int, float)):
                param = str(query_value)
            else:
                raise ValueError(
                    "HanaCheckpointSaver metadata filters only support scalar "
                    f"values (str, int, float, bool, None); got "
                    f"{type(query_value)!r} for key '{query_key}'."
                )
            predicates.append(f"{json_value} = ?")
            param_values.append(param)
        return predicates, param_values

    @staticmethod
    def _build_where_clause(
        config: RunnableConfig | None,
        filter: dict[str, Any] | None,
        before: RunnableConfig | None = None,
    ) -> tuple[str, List[Any]]:
        """Return the WHERE clause (incl. the ``WHERE`` keyword) and its params."""
        where_clauses: list[str] = []
        param_values: list[Any] = []
        if config is not None:
            where_clauses.append('"THREAD_ID" = ?')
            param_values.append(str(config["configurable"]["thread_id"]))
            checkpoint_ns = config["configurable"].get("checkpoint_ns")
            if checkpoint_ns is not None:
                where_clauses.append('"CHECKPOINT_NS" = ?')
                param_values.append(checkpoint_ns)
            if checkpoint_id := get_checkpoint_id(config):
                where_clauses.append('"CHECKPOINT_ID" = ?')
                param_values.append(checkpoint_id)
        if filter:
            predicates, values = BaseHanaCheckpointSaver._metadata_predicate(filter)
            where_clauses.extend(predicates)
            param_values.extend(values)
        if before is not None:
            where_clauses.append('"CHECKPOINT_ID" < ?')
            param_values.append(get_checkpoint_id(before))
        return (
            "WHERE " + " AND ".join(where_clauses) if where_clauses else "",
            param_values,
        )

    @staticmethod
    def _writes_sort_key(task_path: str, task_id: str, idx: int) -> tuple:
        """Deterministic ordering for pending writes.

        Replicates ``langgraph.checkpoint.base.writes_sort_key`` (not exported by
        the installed ``langgraph-checkpoint`` version).
        """
        return (task_path or "", task_id, idx)

    # ------------------------------------------------------------------ #
    # Serialization helpers (pure; kept static for DB-free testing)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_bytes(value: Any) -> bytes | None:
        """Normalise a HANA BLOB/VARBINARY column value to ``bytes``.

        ``hdbcli`` may return a LOB column as raw ``bytes``/``bytearray``, a
        ``memoryview``, or a LOB object exposing ``read()``. In the async driver,
        ``AsyncLob.read()`` is awaited by the caller first, so by this point the
        value is already ``bytes``/``str``.
        """
        if value is None:
            return None
        if isinstance(value, (bytes, bytearray, memoryview)):
            return bytes(value)
        if hasattr(value, "read"):
            return bytes(value.read())
        if isinstance(value, str):
            return value.encode("utf-8")
        raise TypeError(f"Cannot convert value of type {type(value)!r} to bytes")

    @staticmethod
    def _to_str(value: Any) -> str | None:
        """Normalise a HANA NCLOB/NVARCHAR column value to ``str``."""
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if hasattr(value, "read"):
            value = value.read()
        if isinstance(value, (bytes, bytearray, memoryview)):
            return bytes(value).decode("utf-8")
        return str(value)

    @staticmethod
    def _split_checkpoint(checkpoint: Checkpoint) -> tuple[Checkpoint, dict[str, Any]]:
        """Split a checkpoint into an inline skeleton and its non-primitive values.

        Returns ``(skeleton, blob_values)`` where ``skeleton`` is a shallow copy with
        every non-primitive channel value removed, and ``blob_values`` holds those
        removed values keyed by channel. Primitive values
        (``None``/``str``/``int``/``float``/``bool``) stay inline in the skeleton.
        """
        copy = checkpoint.copy()
        copy["channel_values"] = copy["channel_values"].copy()
        blob_values: dict[str, Any] = {}
        for channel, value in checkpoint["channel_values"].items():
            if value is None or isinstance(value, (str, int, float, bool)):
                continue
            blob_values[channel] = copy["channel_values"].pop(channel)
        return copy, blob_values

    @staticmethod
    def _dump_blobs(
        serde: SerializerProtocol,
        thread_id: str,
        checkpoint_ns: str,
        values: Mapping[str, Any],
        versions: ChannelVersions,
    ) -> List[tuple[Any, ...]]:
        """Build blob rows for every channel in ``versions``.

        Channels present in ``values`` are serialized; channels only present in
        ``versions`` (e.g. inline primitives) get an ``"empty"`` placeholder row.
        """
        if not versions:
            return []
        rows: list[tuple[Any, ...]] = []
        for channel, version in versions.items():
            if channel in values:
                serde_type, blob = serde.dumps_typed(values[channel])
            else:
                serde_type, blob = "empty", None
            rows.append(
                (thread_id, checkpoint_ns, channel, str(version), serde_type, blob)
            )
        return rows

    @staticmethod
    def _load_blobs(rows: Iterable[Any], serde: SerializerProtocol) -> dict[str, Any]:
        """Deserialize blob rows ``(channel, type, blob)``, skipping placeholders."""
        result: dict[str, Any] = {}
        for channel, serde_type, blob in rows:
            type_str = BaseHanaCheckpointSaver._to_str(serde_type)
            if type_str is None or type_str == "empty":
                continue
            result[cast(str, BaseHanaCheckpointSaver._to_str(channel))] = (
                serde.loads_typed(
                    cast(
                        "tuple[str, bytes]",
                        (type_str, BaseHanaCheckpointSaver._to_bytes(blob)),
                    )
                )
            )
        return result

    @staticmethod
    def _load_pending_writes(
        rows: Iterable[Any], serde: SerializerProtocol
    ) -> List[PendingWrite]:
        """Deserialize ``_pending_writes_sql`` rows ordered by ``_writes_sort_key``.

        Each row is ``(task_id, channel, type, value, task_path, idx)``.
        """
        return [
            (
                task_id,
                channel,
                serde.loads_typed(
                    cast(
                        "tuple[str, bytes]",
                        (serde_type, BaseHanaCheckpointSaver._to_bytes(value)),
                    )
                ),
            )
            for task_id, channel, serde_type, value, _, _ in sorted(
                rows,
                key=lambda r: BaseHanaCheckpointSaver._writes_sort_key(
                    r[4], r[0], r[5]
                ),
            )
        ]

    def get_next_version(self, current: str | None, channel: None = None) -> str:
        """Return the next monotonically-increasing channel version string."""
        if current is None:
            current_v = 0
        elif isinstance(current, int):
            current_v = current
        else:
            current_v = int(current.split(".")[0])
        next_v = current_v + 1
        next_h = random.random()
        return f"{next_v:032}.{next_h:016}"

"""Shared base for the SAP HANA Cloud LangGraph memory stores.

``BaseHanaStore`` holds everything that does not depend on *how* the database
connection is driven: the schema/DDL, the HANA SQL-dialect query builders, and the
pure helpers (namespace encoding, metadata-filter matching, row -> :class:`Item`
conversion, operation grouping and vector (de)serialization). The synchronous
``HanaStore`` and the native-async ``AsyncHanaStore`` subclass it and add their own
connection, lock, ``setup`` and batch read/write methods.

Three tables are used:

* a **migrations** table (version ledger) recording which schema migrations have
  been applied, so the schema can evolve safely;
* a **store** table holding one row per ``(namespace, key)`` item: its JSON value,
  create/update timestamps and (optional) TTL bookkeeping columns;
* a **vectors** table (created only when semantic search is configured) holding one
  ``REAL_VECTOR`` embedding per indexed text field of an item, keyed by
  ``(prefix, key, field_name)``.
"""

from __future__ import annotations

import json
import struct
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any

from langchain_core.embeddings import Embeddings
from langgraph.store.base import (
    BaseStore,
    IndexConfig,
    Item,
    ListNamespacesOp,
    MatchCondition,
    Op,
    PutOp,
    SearchItem,
    TTLConfig,
    ensure_embeddings,
    get_text_at_path,
)

from langchain_hana.utils import _sanitize_name, _table_ref

_DEFAULT_MIGRATIONS_TABLE = "LANGGRAPH_STORE_MIGRATIONS"
_DEFAULT_STORE_TABLE = "LANGGRAPH_STORE"
_DEFAULT_VECTORS_TABLE = "LANGGRAPH_STORE_VECTORS"

# Bounded identifier lengths keep the composite primary keys within HANA's index
# key-length limit. ``PREFIX`` (the dotted namespace) and ``KEY`` form the store
# primary key; ``FIELD_NAME`` additionally keys the vectors table.
_PREFIX_LEN = 1536
_KEY_LEN = 512
_FIELD_LEN = 512

# Comparison operators accepted inside a filter value, mapped to their meaning in
# ``_apply_operator``. Mirrors langgraph's ``InMemoryStore`` semantics.
_FILTER_OPERATORS = ("$eq", "$ne", "$gt", "$gte", "$lt", "$lte")

# Columns selected for an item row, in the order ``_row_to_item`` expects.
_ITEM_COLS = '"VALUE", "CREATED_AT", "UPDATED_AT"'


class BaseHanaStore(BaseStore):
    """Connection-agnostic base: schema, SQL builders and pure helpers.

    Args:
        index: Optional semantic-search configuration (``dims``/``embed``/``fields``).
            When omitted the store has no vectors table and ``query`` searches fall
            back to metadata-only search. ``embed`` may be an ``Embeddings`` instance,
            a sync/async embedding function, or a ``"provider:model"`` string.
        ttl: Optional TTL (time-to-live) configuration. When provided, expired items
            are hidden from reads and can be swept; TTLs refresh on read/write.
        schema_name: Optional schema; defaults to the connection's current schema.
        migrations_table: Name of the migrations (version ledger) table.
        store_table: Name of the key-value store table.
        vectors_table: Name of the per-field embeddings table.
    """

    supports_ttl: bool = True

    def __init__(
        self,
        *,
        index: IndexConfig | None = None,
        ttl: TTLConfig | None = None,
        schema_name: str = "",
        migrations_table: str = _DEFAULT_MIGRATIONS_TABLE,
        store_table: str = _DEFAULT_STORE_TABLE,
        vectors_table: str = _DEFAULT_VECTORS_TABLE,
    ) -> None:
        super().__init__()
        self.ttl_config = ttl
        self.index_config = index
        self.embeddings: Embeddings | None = None
        if index is not None:
            self.embeddings = ensure_embeddings(index["embed"])
            self.embedding_dims = int(index["dims"])
            self.index_fields = list(index.get("fields") or ["$"])
        else:
            self.embedding_dims = 0
            self.index_fields = []

        self.schema_name = _sanitize_name(schema_name) if schema_name else ""
        self.migrations_table = _sanitize_name(migrations_table)
        self.store_table = _sanitize_name(store_table)
        self.vectors_table = _sanitize_name(vectors_table)
        if not all((self.migrations_table, self.store_table, self.vectors_table)):
            raise ValueError("Table names must not be empty after sanitization.")

    # ------------------------------------------------------------------ #
    # SQL identifiers / schema
    # ------------------------------------------------------------------ #
    @property
    def _migrations_ref(self) -> str:
        return _table_ref(self.migrations_table, self.schema_name)

    @property
    def _store_ref(self) -> str:
        return _table_ref(self.store_table, self.schema_name)

    @property
    def _vectors_ref(self) -> str:
        return _table_ref(self.vectors_table, self.schema_name)

    def _migrations(self) -> list[str]:
        """Ordered DDL statements. Index 0 creates the migrations ledger itself."""
        migrations_ddl = (
            f'CREATE TABLE {self._migrations_ref} ("V" INTEGER PRIMARY KEY)'
        )
        store_ddl = (
            f"CREATE TABLE {self._store_ref} ("
            f'"PREFIX" NVARCHAR({_PREFIX_LEN}) NOT NULL, '
            f'"KEY" NVARCHAR({_KEY_LEN}) NOT NULL, '
            '"VALUE" NCLOB NOT NULL, '
            '"CREATED_AT" TIMESTAMP DEFAULT CURRENT_UTCTIMESTAMP NOT NULL, '
            '"UPDATED_AT" TIMESTAMP DEFAULT CURRENT_UTCTIMESTAMP NOT NULL, '
            '"EXPIRES_AT" TIMESTAMP, '
            '"TTL_MINUTES" DOUBLE, '
            'PRIMARY KEY ("PREFIX", "KEY"))'
        )
        return [migrations_ddl, store_ddl]

    def _vectors_ddl(self) -> str:
        """DDL for the embeddings table (dimensioned by ``index["dims"]``)."""
        return (
            f"CREATE TABLE {self._vectors_ref} ("
            f'"PREFIX" NVARCHAR({_PREFIX_LEN}) NOT NULL, '
            f'"KEY" NVARCHAR({_KEY_LEN}) NOT NULL, '
            f'"FIELD_NAME" NVARCHAR({_FIELD_LEN}) NOT NULL, '
            f'"EMBEDDING" REAL_VECTOR({self.embedding_dims}), '
            '"CREATED_AT" TIMESTAMP DEFAULT CURRENT_UTCTIMESTAMP NOT NULL, '
            '"UPDATED_AT" TIMESTAMP DEFAULT CURRENT_UTCTIMESTAMP NOT NULL, '
            'PRIMARY KEY ("PREFIX", "KEY", "FIELD_NAME"))'
        )

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
    def _omit_expired(self) -> bool:
        """Whether reads should hide expired rows (TTL configured)."""
        return self.ttl_config is not None

    @property
    def _not_expired_clause(self) -> str:
        """SQL fragment (no params) excluding expired rows, or empty string."""
        if not self._omit_expired:
            return ""
        return ' AND ("EXPIRES_AT" IS NULL OR "EXPIRES_AT" > CURRENT_UTCTIMESTAMP)'

    def _get_sql(self, num_keys: int) -> str:
        placeholders = ",".join(["?"] * num_keys)
        return (
            f'SELECT "KEY", {_ITEM_COLS} FROM {self._store_ref} '
            f'WHERE "PREFIX" = ? AND "KEY" IN ({placeholders})'
            f"{self._not_expired_clause}"
        )

    def _refresh_ttl_sql(self, num_keys: int) -> str:
        placeholders = ",".join(["?"] * num_keys)
        return (
            f"UPDATE {self._store_ref} "
            'SET "EXPIRES_AT" = ADD_SECONDS(CURRENT_UTCTIMESTAMP, "TTL_MINUTES" * 60) '
            f'WHERE "PREFIX" = ? AND "KEY" IN ({placeholders}) '
            'AND "TTL_MINUTES" IS NOT NULL'
        )

    @property
    def _merge_store_sql(self) -> str:
        """MERGE that inserts a new item or updates an existing one in place.

        ``CREATED_AT`` is preserved on update (a plain ``UPSERT ... WITH PRIMARY
        KEY`` would reset it); ``EXPIRES_AT`` is derived from the TTL using the
        database clock so it stays consistent with read-time refreshes.
        """
        return (
            f"MERGE INTO {self._store_ref} t USING ("
            'SELECT ? "PREFIX", ? "KEY", ? "VALUE", '
            "CASE WHEN ? IS NULL THEN NULL "
            'ELSE ADD_SECONDS(CURRENT_UTCTIMESTAMP, ? * 60) END "EXPIRES_AT", '
            '? "TTL_MINUTES" FROM DUMMY) s '
            'ON (t."PREFIX" = s."PREFIX" AND t."KEY" = s."KEY") '
            'WHEN MATCHED THEN UPDATE SET t."VALUE" = s."VALUE", '
            't."UPDATED_AT" = CURRENT_UTCTIMESTAMP, '
            't."EXPIRES_AT" = s."EXPIRES_AT", t."TTL_MINUTES" = s."TTL_MINUTES" '
            'WHEN NOT MATCHED THEN INSERT ("PREFIX", "KEY", "VALUE", '
            '"CREATED_AT", "UPDATED_AT", "EXPIRES_AT", "TTL_MINUTES") VALUES ('
            's."PREFIX", s."KEY", s."VALUE", CURRENT_UTCTIMESTAMP, '
            'CURRENT_UTCTIMESTAMP, s."EXPIRES_AT", s."TTL_MINUTES")'
        )

    @property
    def _delete_store_sql(self) -> str:
        return f'DELETE FROM {self._store_ref} WHERE "PREFIX" = ? AND "KEY" = ?'

    @property
    def _delete_vectors_sql(self) -> str:
        return f'DELETE FROM {self._vectors_ref} WHERE "PREFIX" = ? AND "KEY" = ?'

    @property
    def _insert_vector_sql(self) -> str:
        return (
            f"INSERT INTO {self._vectors_ref} "
            '("PREFIX", "KEY", "FIELD_NAME", "EMBEDDING") VALUES (?, ?, ?, ?)'
        )

    def _merge_store_params(self, prefix: str, key: str, op: PutOp) -> tuple[Any, ...]:
        value_json = json.dumps(op.value, ensure_ascii=False)
        ttl = op.ttl
        return (prefix, key, value_json, ttl, ttl, ttl)

    def _search_namespace_clause(
        self, prefix: tuple[str, ...]
    ) -> tuple[str, list[Any]]:
        """``("PREFIX" = ? OR "PREFIX" LIKE ? ESCAPE '\\')`` + params for a prefix.

        Matches the namespace itself and any descendant namespace, with LIKE
        metacharacters in the prefix escaped so they are treated literally.
        """
        prefix_text = self._namespace_to_text(prefix)
        like = self._escape_like(prefix_text) + ".%"
        return ('("PREFIX" = ? OR "PREFIX" LIKE ? ESCAPE \'\\\')', [prefix_text, like])

    def _search_sql(
        self, prefix: tuple[str, ...], ordered_limited: bool, limit: int, offset: int
    ) -> tuple[str, list[Any]]:
        """Metadata (non-vector) search SQL. Fetches an item's namespace too."""
        clause, params = self._search_namespace_clause(prefix)
        sql = (
            f'SELECT "PREFIX", "KEY", {_ITEM_COLS} FROM {self._store_ref} '
            f"WHERE {clause}{self._not_expired_clause} "
            'ORDER BY "UPDATED_AT" DESC'
        )
        if ordered_limited:
            sql += f" LIMIT {int(limit)} OFFSET {int(offset)}"
        return sql, params

    def _vector_search_sql(self, prefix: tuple[str, ...]) -> tuple[str, list[Any]]:
        """Vector search SQL: best cosine similarity per item, newest-first ties.

        The query embedding is bound as the leading parameter via ``TO_REAL_VECTOR``.
        ``VALUE`` is an NCLOB and cannot be grouped, so similarity is aggregated per
        ``(PREFIX, KEY)`` in a sub-select and then joined back to the store.
        """
        clause, params = self._search_namespace_clause(prefix)
        sql = (
            "WITH scored AS ("
            'SELECT "PREFIX", "KEY", '
            'MAX(COSINE_SIMILARITY("EMBEDDING", TO_REAL_VECTOR(?))) AS "SCORE" '
            f'FROM {self._vectors_ref} GROUP BY "PREFIX", "KEY") '
            'SELECT s."PREFIX", s."KEY", s."VALUE", s."CREATED_AT", '
            's."UPDATED_AT", sc."SCORE" '
            f"FROM {self._store_ref} s "
            'JOIN scored sc ON s."PREFIX" = sc."PREFIX" AND s."KEY" = sc."KEY" '
            f"WHERE {clause}{self._not_expired_clause} "
            'ORDER BY sc."SCORE" DESC, s."UPDATED_AT" DESC'
        )
        return sql, params

    @property
    def _list_namespaces_sql(self) -> str:
        sql = f'SELECT DISTINCT "PREFIX" FROM {self._store_ref}'
        if self._omit_expired:
            # Strip the leading " AND " from the shared clause to start a WHERE.
            sql += " WHERE " + self._not_expired_clause[len(" AND ") :]
        return sql

    @property
    def _sweep_store_sql(self) -> str:
        return (
            f"DELETE FROM {self._store_ref} "
            'WHERE "EXPIRES_AT" IS NOT NULL AND "EXPIRES_AT" < CURRENT_UTCTIMESTAMP'
        )

    @property
    def _sweep_vectors_sql(self) -> str:
        """Delete embeddings whose item no longer exists (post-sweep cleanup)."""
        return (
            f"DELETE FROM {self._vectors_ref} v WHERE NOT EXISTS ("
            f"SELECT 1 FROM {self._store_ref} s "
            'WHERE s."PREFIX" = v."PREFIX" AND s."KEY" = v."KEY")'
        )

    # ------------------------------------------------------------------ #
    # Pure helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _namespace_to_text(namespace: tuple[str, ...]) -> str:
        """Encode a namespace tuple as a dot-joined string (the ``PREFIX`` column)."""
        return ".".join(namespace)

    @staticmethod
    def _decode_ns(text: str) -> tuple[str, ...]:
        """Decode a ``PREFIX`` string back into a namespace tuple."""
        return tuple(text.split("."))

    @staticmethod
    def _escape_like(text: str) -> str:
        r"""Escape LIKE metacharacters (``\``, ``%``, ``_``) for ``ESCAPE '\'``."""
        return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

    @classmethod
    def _row_to_item(cls, namespace: tuple[str, ...], row: Any) -> Item:
        """Build an :class:`Item` from a ``(key, value, created_at, updated_at)``."""
        key, value, created_at, updated_at = row
        return Item(
            value=json.loads(value),
            key=key,
            namespace=namespace,
            created_at=created_at,
            updated_at=updated_at,
        )

    @classmethod
    def _row_to_search_item(cls, row: Any) -> SearchItem:
        """Build a :class:`SearchItem` from a search result row.

        Row is ``(prefix, key, value, created_at, updated_at[, score])``.
        """
        prefix, key, value, created_at, updated_at = row[:5]
        score = row[5] if len(row) > 5 else None
        return SearchItem(
            namespace=cls._decode_ns(prefix),
            key=key,
            value=json.loads(value),
            created_at=created_at,
            updated_at=updated_at,
            score=float(score) if score is not None else None,
        )

    @staticmethod
    def _serialize_vector(values: Sequence[float]) -> bytes:
        """Serialize a float vector to HANA ``REAL_VECTOR`` (fvecs) binary format."""
        return struct.pack(f"<I{len(values)}f", len(values), *values)

    @staticmethod
    def _vector_to_str(values: Sequence[float]) -> str:
        """Render a vector as the ``'[v1, v2, ...]'`` literal for ``TO_REAL_VECTOR``."""
        return "[" + ",".join(repr(float(v)) for v in values) + "]"

    @staticmethod
    def _group_ops(ops: Iterable[Op]) -> tuple[dict[type, list[tuple[int, Op]]], int]:
        """Group operations by type, keeping each op's original index for results."""
        grouped: dict[type, list[tuple[int, Op]]] = defaultdict(list)
        total = 0
        for idx, op in enumerate(ops):
            grouped[type(op)].append((idx, op))
            total += 1
        return grouped, total

    @classmethod
    def _dedupe_put_ops(
        cls, put_ops: list[tuple[int, PutOp]]
    ) -> "dict[tuple[str, str], PutOp]":
        """Collapse puts on the same ``(namespace, key)`` keeping the last one."""
        out: dict[tuple[str, str], PutOp] = {}
        for _, op in put_ops:
            out[(cls._namespace_to_text(op.namespace), op.key)] = op
        return out

    def _extract_index_texts(self, op: PutOp) -> list[tuple[str, str]]:
        """Return ``(field_name, text)`` pairs to embed for an indexed put.

        Honours a per-op ``index`` override, falls back to the store's configured
        ``fields``, and yields nothing when indexing is disabled for the item.
        """
        if op.value is None or self.index_config is None or op.index is False:
            return []
        paths = list(op.index) if op.index else self.index_fields
        pairs: list[tuple[str, str]] = []
        for path in paths:
            texts = get_text_at_path(op.value, path)
            if len(texts) == 1:
                pairs.append((path, texts[0]))
            else:
                for i, text in enumerate(texts):
                    pairs.append((f"{path}[{i}]", text))
        return pairs

    # -- metadata filtering (python-side, mirrors InMemoryStore) -------- #
    @classmethod
    def _matches_filter(
        cls, value: dict[str, Any], filter: dict[str, Any] | None
    ) -> bool:
        """Whether an item value satisfies every key in a search ``filter``."""
        if not filter:
            return True
        return all(
            cls._compare_values(value.get(key), fval) for key, fval in filter.items()
        )

    @classmethod
    def _compare_values(cls, item_value: Any, filter_value: Any) -> bool:
        """Compare a stored value against a filter value (JSONB-like semantics)."""
        if isinstance(filter_value, dict):
            if any(k.startswith("$") for k in filter_value):
                return all(
                    cls._apply_operator(item_value, op_key, op_value)
                    for op_key, op_value in filter_value.items()
                )
            if not isinstance(item_value, dict):
                return False
            return all(
                cls._compare_values(item_value.get(k), v)
                for k, v in filter_value.items()
            )
        if isinstance(filter_value, (list, tuple)):
            return (
                isinstance(item_value, (list, tuple))
                and len(item_value) == len(filter_value)
                and all(
                    cls._compare_values(iv, fv)
                    for iv, fv in zip(item_value, filter_value)
                )
            )
        return item_value == filter_value

    @staticmethod
    def _apply_operator(value: Any, operator: str, op_value: Any) -> bool:
        """Apply a single ``$``-operator, matching PostgreSQL JSONB behaviour."""
        if operator == "$eq":
            return value == op_value
        if operator == "$ne":
            return value != op_value
        if operator == "$gt":
            return float(value) > float(op_value)
        if operator == "$gte":
            return float(value) >= float(op_value)
        if operator == "$lt":
            return float(value) < float(op_value)
        if operator == "$lte":
            return float(value) <= float(op_value)
        raise ValueError(f"Unsupported operator: {operator}")

    # -- namespace listing (python-side) -------------------------------- #
    @staticmethod
    def _does_match(match_condition: MatchCondition, key: tuple[str, ...]) -> bool:
        """Whether a namespace ``key`` matches a prefix/suffix condition with wildcards.

        Re-implements langgraph's ``_does_match`` (not exported by the installed
        ``langgraph`` version).
        """
        path = match_condition.path
        if len(key) < len(path):
            return False
        if match_condition.match_type == "prefix":
            pairs = zip(key, path)
        elif match_condition.match_type == "suffix":
            pairs = zip(reversed(key), reversed(path))
        else:
            raise ValueError(f"Unsupported match type: {match_condition.match_type}")
        return all(p == "*" or k == p for k, p in pairs)

    @classmethod
    def _filter_namespaces(
        cls, prefixes: Iterable[str], op: ListNamespacesOp
    ) -> list[tuple[str, ...]]:
        """Apply match conditions, depth truncation, dedupe and pagination.

        Reproduces ``InMemoryStore.list_namespaces`` semantics over the distinct
        ``PREFIX`` strings read from the store.
        """
        conditions = op.match_conditions or ()
        seen: set[tuple[str, ...]] = set()
        namespaces: list[tuple[str, ...]] = []
        for prefix in prefixes:
            ns = cls._decode_ns(prefix)
            if not all(cls._does_match(c, ns) for c in conditions):
                continue
            if op.max_depth is not None:
                ns = ns[: op.max_depth]
            if ns not in seen:
                seen.add(ns)
                namespaces.append(ns)
        namespaces.sort()
        return namespaces[op.offset : op.offset + op.limit]

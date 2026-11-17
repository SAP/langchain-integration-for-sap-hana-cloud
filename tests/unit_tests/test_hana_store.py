"""Unit tests for ``HanaStore`` SQL builders and batch-handler helpers (DB-free).

These exercise the per-store SQL builders and the pure batch-handler helpers through a
``HanaStore`` instance constructed with ``object.__new__`` (so no live database is
needed). Class-level ``BaseHanaStore`` helpers that can be called without an instance
live in ``test_base_hana_store.py``.
"""

import struct

import pytest
from langgraph.store.base import (
    GetOp,
    PutOp,
    SearchItem,
    SearchOp,
)

from langchain_hana.memorystores import HanaStore


def _bare_store(
    *,
    schema_name: str = "",
    ttl_config: object | None = None,
    index_config: object | None = None,
    embedding_dims: int = 0,
    vector_column_type: str = "REAL_VECTOR",
    use_internal_embeddings: bool = False,
    internal_embedding_model_id: str = "",
    internal_embedding_remote_source_schema: str = "",
    internal_embedding_remote_source: str = "",
) -> HanaStore:
    """Build a store without running __init__ (which needs a live DB)."""
    store = object.__new__(HanaStore)
    store.schema_name = schema_name
    store.migrations_table = "LANGGRAPH_STORE_MIGRATIONS"
    store.store_table = "LANGGRAPH_STORE"
    store.vectors_table = "LANGGRAPH_STORE_VECTORS"
    store.ttl_config = ttl_config  # type: ignore[assignment]
    store.index_config = index_config  # type: ignore[assignment]
    store.embedding_dims = embedding_dims
    store.index_fields = ["$"]
    store.embeddings = None
    store.vector_column_type = vector_column_type
    store.use_internal_embeddings = use_internal_embeddings
    store.internal_embedding_model_id = internal_embedding_model_id
    store.internal_embedding_remote_source_schema = (
        internal_embedding_remote_source_schema
    )
    store.internal_embedding_remote_source = internal_embedding_remote_source
    return store


def _search_item(namespace: tuple[str, ...], key: str, value: dict) -> SearchItem:
    return SearchItem(
        namespace=namespace,
        key=key,
        value=value,
        created_at="2024-01-01",  # type: ignore[arg-type]
        updated_at="2024-01-02",  # type: ignore[arg-type]
    )


# --------------------------------------------------------------------------- #
# Vector (de)serialization
# --------------------------------------------------------------------------- #
def test_serialize_vector_fvecs_layout() -> None:
    values = [0.5, -1.25, 3.0]
    blob = _bare_store()._serialize_vector(values)
    count = struct.unpack_from("<I", blob, 0)[0]
    assert count == 3
    unpacked = list(struct.unpack_from("<3f", blob, 4))
    assert unpacked == pytest.approx(values)


def test_serialize_vector_half_precision_layout() -> None:
    values = [0.5, -1.25, 3.0]
    store = _bare_store(vector_column_type="HALF_VECTOR")
    blob = store._serialize_vector(values)
    # half precision is 2 bytes/element: 4-byte count header + 3 * 2 bytes
    assert len(blob) == 4 + 3 * 2
    count = struct.unpack_from("<I", blob, 0)[0]
    assert count == 3
    unpacked = list(struct.unpack_from("<3e", blob, 4))
    assert unpacked == pytest.approx(values)


# --------------------------------------------------------------------------- #
# SQL builders
# --------------------------------------------------------------------------- #
def test_get_sql_no_ttl_has_no_expiry_clause() -> None:
    store = _bare_store()
    sql = store._get_sql(2)
    assert '"KEY" IN (?,?)' in sql
    assert "EXPIRES_AT" not in sql


def test_get_sql_with_ttl_adds_expiry_clause() -> None:
    store = _bare_store(ttl_config={})
    sql = store._get_sql(1)
    assert "EXPIRES_AT" in sql


def test_merge_store_sql_preserves_created_at() -> None:
    store = _bare_store()
    sql = store._merge_store_sql
    assert sql.startswith("MERGE INTO")
    assert '"CREATED_AT"' in sql
    # update branch must not reset CREATED_AT
    update_branch = sql[sql.index("WHEN MATCHED") : sql.index("WHEN NOT MATCHED")]
    assert "CREATED_AT" not in update_branch


def test_merge_store_sql_casts_ttl_to_double() -> None:
    # Without the CAST, HANA infers the bound TTL param as INTEGER and truncates
    # fractional minutes (e.g. 0.02) to 0, so EXPIRES_AT lands in the past.
    store = _bare_store()
    sql = store._merge_store_sql
    assert "ADD_SECONDS(CURRENT_UTCTIMESTAMP, CAST(? AS DOUBLE) * 60)" in sql


def test_merge_store_params_shape() -> None:
    store = _bare_store()
    op = PutOp(namespace=("a",), key="k", value={"v": 1}, ttl=5.0)
    params = store._merge_store_params("a", "k", op)
    assert params == ("a", "k", '{"v": 1}', 5.0, 5.0, 5.0)


def test_vector_search_sql_shape_and_schema_ref() -> None:
    store = _bare_store(schema_name="S", index_config={}, embedding_dims=4)
    sql, params = store._vector_search_sql(("users",))
    assert "COSINE_SIMILARITY" in sql
    assert "TO_REAL_VECTOR(?)" in sql
    assert '"S"."LANGGRAPH_STORE_VECTORS"' in sql
    assert '"S"."LANGGRAPH_STORE"' in sql
    # PREFIX in the WHERE clause must be alias-qualified: the outer query joins the
    # store and the scored sub-select, both of which expose a PREFIX column.
    assert '(s."PREFIX" = ? OR s."PREFIX" LIKE ?' in sql
    # namespace prefix params: exact + descendant LIKE
    assert params == ["users", r"users.%"]


def test_vector_search_sql_half_vector_conversion() -> None:
    store = _bare_store(
        index_config={}, embedding_dims=4, vector_column_type="HALF_VECTOR"
    )
    sql, _ = store._vector_search_sql(("users",))
    assert "TO_HALF_VECTOR(?)" in sql
    assert "TO_REAL_VECTOR(?)" not in sql


def test_search_sql_limits_in_sql_when_ordered() -> None:
    store = _bare_store()
    sql, params = store._search_sql(("a",), True, 5, 2)
    assert "LIMIT 5 OFFSET 2" in sql
    assert params == ["a", r"a.%"]


def test_search_sql_no_limit_when_python_paginated() -> None:
    store = _bare_store()
    sql, _ = store._search_sql(("a",), False, 5, 2)
    assert "LIMIT" not in sql


def test_search_namespace_clause_empty_prefix_matches_all() -> None:
    # Empty prefix must match every namespace, not the no-match "" / ".%" form.
    store = _bare_store()
    clause, params = store._search_namespace_clause(())
    assert clause == "1=1"
    assert params == []
    assert "PREFIX" not in clause


def test_search_sql_empty_prefix_has_no_prefix_filter() -> None:
    store = _bare_store()
    sql, params = store._search_sql((), True, 5, 0)
    assert params == []
    assert '"PREFIX" LIKE' not in sql


def test_vector_search_sql_empty_prefix_has_no_prefix_filter() -> None:
    store = _bare_store(schema_name="S", index_config={}, embedding_dims=4)
    _, params = store._vector_search_sql(())
    assert params == []


def test_list_namespaces_sql_with_ttl_has_where() -> None:
    store = _bare_store(ttl_config={})
    sql = store._list_namespaces_sql
    assert sql.startswith('SELECT DISTINCT "PREFIX"')
    assert " WHERE (" in sql


def test_vectors_ddl_uses_dims() -> None:
    store = _bare_store(index_config={}, embedding_dims=16)
    ddl = store._vectors_ddl()
    assert "REAL_VECTOR(16)" in ddl
    assert 'PRIMARY KEY ("PREFIX", "KEY", "FIELD_NAME")' in ddl


def test_vectors_ddl_half_vector_column_type() -> None:
    store = _bare_store(
        index_config={}, embedding_dims=16, vector_column_type="HALF_VECTOR"
    )
    ddl = store._vectors_ddl()
    assert "HALF_VECTOR(16)" in ddl
    assert "REAL_VECTOR" not in ddl


# --------------------------------------------------------------------------- #
# Batch handler helpers (pure; shared by sync and async)
# --------------------------------------------------------------------------- #
def test_group_get_ops_groups_by_namespace_preserving_order() -> None:
    store = _bare_store()
    get_ops = [
        (0, GetOp(namespace=("a",), key="k1")),
        (1, GetOp(namespace=("b",), key="k2")),
        (2, GetOp(namespace=("a",), key="k3")),
    ]
    grouped = store._group_get_ops(get_ops)
    assert list(grouped) == ["a", "b"]
    assert [op.key for _, op in grouped["a"]] == ["k1", "k3"]
    assert [op.key for _, op in grouped["b"]] == ["k2"]


def test_assign_get_results_fills_slots_and_returns_refresh_keys() -> None:
    store = _bare_store()
    namespace = ("a",)
    # rows are (key, value_json, created_at, updated_at)
    rows = [
        ("k1", '{"v": 1}', "2024-01-01", "2024-01-02"),
        ("k3", '{"v": 3}', "2024-01-01", "2024-01-02"),
    ]
    items = [
        (0, GetOp(namespace=namespace, key="k1", refresh_ttl=True)),
        (1, GetOp(namespace=namespace, key="k2", refresh_ttl=True)),
        (2, GetOp(namespace=namespace, key="k3", refresh_ttl=False)),
    ]
    results: list[object] = [None, None, None]
    refresh = store._assign_get_results(namespace, rows, items, results)
    assert results[0].value == {"v": 1}  # type: ignore[attr-defined]
    assert results[1] is None  # not found
    assert results[2].value == {"v": 3}  # type: ignore[attr-defined]
    # only found keys that requested a refresh: k1 (found+refresh), not k2
    # (missing), not k3 (refresh_ttl False)
    assert refresh == ["k1"]


def test_finalize_search_filters_then_pages() -> None:
    store = _bare_store()
    candidates = [
        _search_item(("a",), "k1", {"color": "red"}),
        _search_item(("a",), "k2", {"color": "blue"}),
        _search_item(("a",), "k3", {"color": "red"}),
        _search_item(("a",), "k4", {"color": "red"}),
    ]
    op = SearchOp(namespace_prefix=("a",), filter={"color": "red"}, limit=1, offset=1)
    page = store._finalize_search(op, candidates)
    # matches are k1, k3, k4; offset 1 limit 1 -> [k3]
    assert [item.key for item in page] == ["k3"]


def test_finalize_search_no_filter_returns_window() -> None:
    store = _bare_store()
    candidates = [_search_item(("a",), f"k{i}", {}) for i in range(5)]
    op = SearchOp(namespace_prefix=("a",), filter=None, limit=2, offset=1)
    page = store._finalize_search(op, candidates)
    assert [item.key for item in page] == ["k1", "k2"]


def test_group_refresh_keys_groups_by_namespace() -> None:
    store = _bare_store()
    items = [
        _search_item(("a",), "k1", {}),
        _search_item(("b",), "k2", {}),
        _search_item(("a",), "k3", {}),
    ]
    grouped = store._group_refresh_keys(items)
    assert grouped == [("a", ["k1", "k3"]), ("b", ["k2"])]


def test_vector_insert_params_pairs_rows_with_serialized_vectors() -> None:
    store = _bare_store(index_config={}, embedding_dims=2)
    index_rows = [("a", "k1", "$"), ("a", "k2", "title")]
    vectors = [[1.0, 2.0], [3.0, 4.0]]
    params = store._vector_insert_params(index_rows, vectors)
    assert len(params) == 2
    assert params[0][:3] == ("a", "k1", "$")
    assert params[1][:3] == ("a", "k2", "title")
    # 4th element is the fvecs blob for the matching vector
    blob = params[1][3]
    count = struct.unpack_from("<I", blob, 0)[0]
    assert count == 2
    assert list(struct.unpack_from("<2f", blob, 4)) == pytest.approx([3.0, 4.0])


# --------------------------------------------------------------------------- #
# Internal (in-database) embeddings
# --------------------------------------------------------------------------- #
def _internal_store(
    *, remote_source_schema: str = "", remote_source: str = "", **kw: object
) -> HanaStore:
    return _bare_store(
        index_config={},
        embedding_dims=4,
        use_internal_embeddings=True,
        internal_embedding_model_id="SAP_NEB.20240715",
        internal_embedding_remote_source_schema=remote_source_schema,
        internal_embedding_remote_source=remote_source,
        **kw,  # type: ignore[arg-type]
    )


def test_vector_embedding_sql_and_params_three_arg_form() -> None:
    store = _internal_store()
    sql, params = store._vector_embedding_sql_and_params("hello", "QUERY")
    assert sql == "VECTOR_EMBEDDING(?, 'QUERY', ?)"
    assert params == ("hello", "SAP_NEB.20240715")


def test_vector_embedding_sql_and_params_remote_source_form() -> None:
    store = _internal_store(remote_source_schema="REMOTE", remote_source="SRC")
    sql, params = store._vector_embedding_sql_and_params("doc", "DOCUMENT")
    assert sql == "VECTOR_EMBEDDING(?, 'DOCUMENT', ?, ?, ?)"
    assert params == ("doc", "SAP_NEB.20240715", "REMOTE", "SRC")


def test_vector_embedding_sql_and_params_rejects_bad_kind() -> None:
    store = _internal_store()
    with pytest.raises(ValueError):
        store._vector_embedding_sql_and_params("x", "SENTENCE")


def test_insert_vector_internal_sql_embeds_in_db() -> None:
    store = _internal_store()
    sql = store._insert_vector_internal_sql
    assert sql.startswith("INSERT INTO")
    assert '("PREFIX", "KEY", "FIELD_NAME", "EMBEDDING")' in sql
    # the embedding value is computed in-database, not a bound blob param
    assert "VALUES (?, ?, ?, VECTOR_EMBEDDING(?, 'DOCUMENT', ?))" in sql
    assert "TO_HALF_VECTOR" not in sql


def test_insert_vector_internal_sql_half_vector_wraps_embedding() -> None:
    store = _internal_store(vector_column_type="HALF_VECTOR")
    sql = store._insert_vector_internal_sql
    assert "TO_HALF_VECTOR(VECTOR_EMBEDDING(?, 'DOCUMENT', ?))" in sql


def test_vector_insert_internal_params_trails_embedding_params() -> None:
    store = _internal_store()
    index_rows = [("a", "k1", "$"), ("a", "k2", "title")]
    index_texts = ["first", "second"]
    params = store._vector_insert_internal_params(index_rows, index_texts)
    assert params == [
        ("a", "k1", "$", "first", "SAP_NEB.20240715"),
        ("a", "k2", "title", "second", "SAP_NEB.20240715"),
    ]


def test_vector_insert_internal_params_with_remote_source() -> None:
    store = _internal_store(remote_source_schema="REMOTE", remote_source="SRC")
    params = store._vector_insert_internal_params([("a", "k", "$")], ["t"])
    assert params == [("a", "k", "$", "t", "SAP_NEB.20240715", "REMOTE", "SRC")]


def test_vector_search_sql_internal_embeds_query_in_db() -> None:
    store = _internal_store()
    sql, params = store._vector_search_sql(("users",))
    assert "VECTOR_EMBEDDING(?, 'QUERY', ?)" in sql
    assert "TO_REAL_VECTOR(?)" not in sql
    # only namespace-prefix params are returned; the query-embedding params lead,
    # supplied by the handler.
    assert params == ["users", r"users.%"]


def test_vector_search_sql_internal_half_vector_wraps_embedding() -> None:
    store = _internal_store(vector_column_type="HALF_VECTOR")
    sql, _ = store._vector_search_sql(("users",))
    assert "TO_HALF_VECTOR(VECTOR_EMBEDDING(?, 'QUERY', ?))" in sql


def test_validate_embedding_sql_probe_shape() -> None:
    store = _internal_store()
    embed_sql, _ = store._vector_embedding_sql_and_params("test", "QUERY")
    probe = store._validate_embedding_sql(embed_sql)
    assert probe == (
        "SELECT COUNT(TO_NVARCHAR(VECTOR_EMBEDDING(?, 'QUERY', ?))) "
        'AS "CNT" FROM sys.DUMMY'
    )

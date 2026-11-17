"""Unit tests for the HANA memory store helpers and SQL builders (DB-free)."""

import pytest
from langgraph.store.base import (
    GetOp,
    ListNamespacesOp,
    MatchCondition,
    Op,
    PutOp,
    SearchItem,
    SearchOp,
)

from langchain_hana.memorystores import HanaStore
from langchain_hana.memorystores.base_hana_store import BaseHanaStore


def _bare_store(
    *,
    schema_name: str = "",
    ttl_config: object | None = None,
    index_config: object | None = None,
    embedding_dims: int = 0,
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
    return store


# --------------------------------------------------------------------------- #
# Namespace encode/decode
# --------------------------------------------------------------------------- #
def test_namespace_roundtrip() -> None:
    ns = ("users", "123", "prefs")
    text = BaseHanaStore._namespace_to_text(ns)
    assert text == "users.123.prefs"
    assert BaseHanaStore._decode_ns(text) == ns


def test_escape_like_escapes_metacharacters() -> None:
    assert BaseHanaStore._escape_like(r"a%b_c\d") == r"a\%b\_c\\d"


# --------------------------------------------------------------------------- #
# Vector (de)serialization
# --------------------------------------------------------------------------- #
def test_serialize_vector_fvecs_layout() -> None:
    import struct

    values = [0.5, -1.25, 3.0]
    blob = BaseHanaStore._serialize_vector(values)
    count = struct.unpack_from("<I", blob, 0)[0]
    assert count == 3
    unpacked = list(struct.unpack_from("<3f", blob, 4))
    assert unpacked == pytest.approx(values)


def test_vector_to_str_literal() -> None:
    assert BaseHanaStore._vector_to_str([1.0, 2.5]) == "[1.0,2.5]"


# --------------------------------------------------------------------------- #
# Op grouping / dedupe
# --------------------------------------------------------------------------- #
def test_group_ops_preserves_index_and_counts() -> None:
    ops: list[Op] = [
        GetOp(namespace=("a",), key="k1"),
        PutOp(namespace=("a",), key="k2", value={"x": 1}),
        GetOp(namespace=("a",), key="k3"),
    ]
    grouped, total = BaseHanaStore._group_ops(ops)
    assert total == 3
    assert [idx for idx, _ in grouped[GetOp]] == [0, 2]
    assert [idx for idx, _ in grouped[PutOp]] == [1]


def test_dedupe_put_ops_last_wins() -> None:
    ops = [
        (0, PutOp(namespace=("a",), key="k", value={"v": 1})),
        (1, PutOp(namespace=("a",), key="k", value={"v": 2})),
    ]
    deduped = BaseHanaStore._dedupe_put_ops(ops)
    assert list(deduped) == [("a", "k")]
    assert deduped[("a", "k")].value == {"v": 2}


# --------------------------------------------------------------------------- #
# Metadata filtering (parity with InMemoryStore semantics)
# --------------------------------------------------------------------------- #
def test_matches_filter_scalar_and_operators() -> None:
    value = {"color": "red", "score": 5}
    assert BaseHanaStore._matches_filter(value, {"color": "red"})
    assert not BaseHanaStore._matches_filter(value, {"color": "blue"})
    assert BaseHanaStore._matches_filter(value, {"score": {"$gte": 5}})
    assert BaseHanaStore._matches_filter(value, {"score": {"$lt": 10, "$gt": 1}})
    assert not BaseHanaStore._matches_filter(value, {"score": {"$lt": 5}})
    assert BaseHanaStore._matches_filter(value, None)


def test_matches_filter_nested_and_list() -> None:
    value = {"meta": {"a": 1, "b": 2}, "tags": [1, 2, 3]}
    assert BaseHanaStore._matches_filter(value, {"meta": {"a": 1}})
    assert not BaseHanaStore._matches_filter(value, {"meta": {"a": 9}})
    assert BaseHanaStore._matches_filter(value, {"tags": [1, 2, 3]})
    assert not BaseHanaStore._matches_filter(value, {"tags": [1, 2]})


def test_apply_operator_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        BaseHanaStore._apply_operator(1, "$in", [1, 2])


# --------------------------------------------------------------------------- #
# Namespace listing
# --------------------------------------------------------------------------- #
def test_does_match_prefix_suffix_wildcard() -> None:
    key = ("users", "123", "prefs")
    assert BaseHanaStore._does_match(
        MatchCondition(match_type="prefix", path=("users",)), key
    )
    assert BaseHanaStore._does_match(
        MatchCondition(match_type="prefix", path=("users", "*")), key
    )
    assert BaseHanaStore._does_match(
        MatchCondition(match_type="suffix", path=("prefs",)), key
    )
    assert not BaseHanaStore._does_match(
        MatchCondition(match_type="prefix", path=("orgs",)), key
    )


def test_filter_namespaces_dedupe_depth_and_paginate() -> None:
    prefixes = ["a.b.c", "a.b.d", "a.x", "z"]
    op = ListNamespacesOp(match_conditions=(), max_depth=2, limit=10, offset=0)
    result = BaseHanaStore._filter_namespaces(prefixes, op)
    # depth-truncated to 2 and deduped: ("a","b") appears once
    assert result == [("a", "b"), ("a", "x"), ("z",)]


def test_filter_namespaces_prefix_condition() -> None:
    prefixes = ["users.1", "users.2", "orgs.1"]
    op = ListNamespacesOp(
        match_conditions=(MatchCondition(match_type="prefix", path=("users",)),),
        max_depth=None,
        limit=10,
        offset=0,
    )
    result = BaseHanaStore._filter_namespaces(prefixes, op)
    assert result == [("users", "1"), ("users", "2")]


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


def test_search_sql_limits_in_sql_when_ordered() -> None:
    store = _bare_store()
    sql, params = store._search_sql(("a",), True, 5, 2)
    assert "LIMIT 5 OFFSET 2" in sql
    assert params == ["a", r"a.%"]


def test_search_sql_no_limit_when_python_paginated() -> None:
    store = _bare_store()
    sql, _ = store._search_sql(("a",), False, 5, 2)
    assert "LIMIT" not in sql


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


def _search_item(namespace: tuple[str, ...], key: str, value: dict) -> SearchItem:
    return SearchItem(
        namespace=namespace,
        key=key,
        value=value,
        created_at="2024-01-01",  # type: ignore[arg-type]
        updated_at="2024-01-02",  # type: ignore[arg-type]
    )


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
    import struct

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

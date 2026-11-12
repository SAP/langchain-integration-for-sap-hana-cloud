"""Unit tests for the HANA memory store helpers and SQL builders (DB-free)."""

import pytest
from langgraph.store.base import (
    GetOp,
    ListNamespacesOp,
    MatchCondition,
    Op,
    PutOp,
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

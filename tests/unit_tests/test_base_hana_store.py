"""Unit tests for the shared ``BaseHanaStore`` pure helpers (DB-free).

These cover the I/O-free logic that both the sync ``HanaStore`` and the native-async
``AsyncHanaStore`` inherit from ``BaseHanaStore`` and that can be exercised directly on
the class (namespace encode/decode, op grouping/dedupe, metadata filtering, namespace
listing). SQL builders and batch-handler helpers that need a constructed store instance
live in ``test_hana_store.py``.
"""

import pytest
from langgraph.store.base import (
    GetOp,
    ListNamespacesOp,
    MatchCondition,
    Op,
    PutOp,
)

from langchain_hana.memorystores.base_hana_store import BaseHanaStore


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

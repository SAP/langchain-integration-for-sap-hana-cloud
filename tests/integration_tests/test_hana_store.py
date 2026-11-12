"""Integration tests for the HANA memory store (live HANA Cloud).

Requires ``HANA_DB_ADDRESS``/``HANA_DB_PORT``/``HANA_DB_USER``/``HANA_DB_PASSWORD``.
Each test module runs against a dedicated throwaway schema.
"""

from __future__ import annotations

import os
import time
from collections.abc import AsyncGenerator, Generator
from typing import Any, TypedDict

import pytest
from hdbcli import dbapi
from langgraph.store.base import IndexConfig, TTLConfig

from langchain_hana.memorystores import AsyncHanaStore, HanaStore
from tests.integration_tests.fake_embeddings import ConsistentFakeEmbeddings
from tests.integration_tests.hana_test_utils import HanaTestUtils

_DIMS = 10


class _Tables(TypedDict):
    migrations_table: str
    store_table: str
    vectors_table: str


class Config:
    conn: dbapi.Connection
    schema_name: str


config = Config()


def _conn_params() -> dict[str, Any]:
    return {
        "address": os.environ["HANA_DB_ADDRESS"],
        "port": int(os.environ["HANA_DB_PORT"]),
        "user": os.environ["HANA_DB_USER"],
        "password": os.environ["HANA_DB_PASSWORD"],
    }


def _index_config() -> IndexConfig:
    return {
        "dims": _DIMS,
        "embed": ConsistentFakeEmbeddings(dimensionality=_DIMS),
        "fields": ["text"],
    }


@pytest.fixture(scope="module", autouse=True)
def setup_schema() -> Generator[None, None, None]:
    config.conn = dbapi.connect(**_conn_params())
    schema_prefix = "LANGCHAIN_STORE"
    HanaTestUtils.drop_old_test_schemas(config.conn, schema_prefix)
    config.schema_name = HanaTestUtils.generate_schema_name(config.conn, schema_prefix)
    HanaTestUtils.create_and_set_schema(config.conn, config.schema_name)
    yield
    HanaTestUtils.drop_schema_if_exists(config.conn, config.schema_name)
    config.conn.close()


@pytest.fixture
def store() -> Generator[HanaStore, None, None]:
    """A plain (no-index, no-TTL) sync store on a unique table set per test."""
    conn = dbapi.connect(**_conn_params())
    suffix = HanaTestUtils.generate_schema_name(conn, "T").split("_")[-1][:16]
    try:
        yield HanaStore(
            conn,
            schema_name=config.schema_name,
            migrations_table=f"M_{suffix}",
            store_table=f"S_{suffix}",
            vectors_table=f"V_{suffix}",
        )
    finally:
        conn.close()


def _unique_tables(conn: dbapi.Connection) -> _Tables:
    suffix = HanaTestUtils.generate_schema_name(conn, "T").split("_")[-1][:16]
    return {
        "migrations_table": f"M_{suffix}",
        "store_table": f"S_{suffix}",
        "vectors_table": f"V_{suffix}",
    }


# --------------------------------------------------------------------------- #
# Core KV
# --------------------------------------------------------------------------- #
def test_put_get_delete(store: HanaStore) -> None:
    ns = ("users", "1")
    store.put(ns, "prefs", {"theme": "dark"})
    item = store.get(ns, "prefs")
    assert item is not None
    assert item.value == {"theme": "dark"}
    assert item.namespace == ns
    assert item.key == "prefs"

    store.delete(ns, "prefs")
    assert store.get(ns, "prefs") is None


def test_overwrite_preserves_created_at(store: HanaStore) -> None:
    ns = ("docs",)
    store.put(ns, "a", {"v": 1})
    first = store.get(ns, "a")
    assert first is not None
    time.sleep(1.1)
    store.put(ns, "a", {"v": 2})
    second = store.get(ns, "a")
    assert second is not None
    assert second.value == {"v": 2}
    assert second.created_at == first.created_at
    assert second.updated_at > first.updated_at


def test_get_missing_returns_none(store: HanaStore) -> None:
    assert store.get(("nope",), "x") is None


# --------------------------------------------------------------------------- #
# Search: metadata filtering + pagination
# --------------------------------------------------------------------------- #
def test_search_filters_and_pagination(store: HanaStore) -> None:
    ns = ("items",)
    store.put(ns, "a", {"color": "red", "score": 1})
    store.put(ns, "b", {"color": "red", "score": 5})
    store.put(ns, "c", {"color": "blue", "score": 9})

    reds = store.search(ns, filter={"color": "red"})
    assert {i.key for i in reds} == {"a", "b"}

    high = store.search(ns, filter={"score": {"$gte": 5}})
    assert {i.key for i in high} == {"b", "c"}

    page = store.search(ns, limit=1, offset=1)
    assert len(page) == 1

    nested = store.search(ns, filter={"color": {"$ne": "blue"}})
    assert {i.key for i in nested} == {"a", "b"}


# --------------------------------------------------------------------------- #
# Namespace listing
# --------------------------------------------------------------------------- #
def test_list_namespaces(store: HanaStore) -> None:
    store.put(("a", "b", "c"), "k", {"v": 1})
    store.put(("a", "b", "d"), "k", {"v": 1})
    store.put(("a", "x"), "k", {"v": 1})

    all_ns = store.list_namespaces()
    assert ("a", "b", "c") in all_ns

    prefixed = store.list_namespaces(prefix=("a", "b"))
    assert set(prefixed) == {("a", "b", "c"), ("a", "b", "d")}

    truncated = store.list_namespaces(prefix=("a",), max_depth=2)
    assert ("a", "b") in truncated and ("a", "x") in truncated

    suffixed = store.list_namespaces(suffix=("c",))
    assert suffixed == [("a", "b", "c")]


# --------------------------------------------------------------------------- #
# TTL
# --------------------------------------------------------------------------- #
def test_ttl_expiry_refresh_and_sweep() -> None:
    conn = dbapi.connect(**_conn_params())
    try:
        ttl: TTLConfig = {"default_ttl": 0.02, "refresh_on_read": True}  # ~1.2s
        s = HanaStore(
            conn,
            schema_name=config.schema_name,
            ttl=ttl,
            **_unique_tables(conn),
        )
        ns = ("ephemeral",)
        s.put(ns, "k", {"v": 1})
        assert s.get(ns, "k", refresh_ttl=False) is not None
        time.sleep(2.0)
        # Expired rows are hidden from reads.
        assert s.get(ns, "k", refresh_ttl=False) is None
        removed = s.sweep_ttl()
        assert removed >= 1
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# Vector / semantic search
# --------------------------------------------------------------------------- #
def test_vector_search_ranks_by_similarity() -> None:
    conn = dbapi.connect(**_conn_params())
    try:
        s = HanaStore(
            conn,
            schema_name=config.schema_name,
            index=_index_config(),
            **_unique_tables(conn),
        )
        ns = ("memories",)
        s.put(ns, "cats", {"text": "cats"})
        s.put(ns, "dogs", {"text": "dogs"})
        s.put(ns, "birds", {"text": "birds"})

        results = s.search(ns, query="cats")
        assert results
        assert results[0].key == "cats"
        assert results[0].score is not None
    finally:
        conn.close()


def test_put_with_index_false_skips_embedding() -> None:
    conn = dbapi.connect(**_conn_params())
    try:
        s = HanaStore(
            conn,
            schema_name=config.schema_name,
            index=_index_config(),
            **_unique_tables(conn),
        )
        ns = ("memories",)
        s.put(ns, "x", {"text": "hello"}, index=False)
        # No embedding stored -> not returned by a vector query.
        assert s.search(ns, query="hello") == []
        # But still retrievable by key and by metadata search.
        assert s.get(ns, "x") is not None
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# Async store
# --------------------------------------------------------------------------- #
@pytest.fixture
async def async_store() -> AsyncGenerator[AsyncHanaStore, None]:
    conn = await dbapi.async_connect(**_conn_params())  # type: ignore[misc]
    try:
        s = AsyncHanaStore(
            conn,
            schema_name=config.schema_name,
            **_unique_tables(config.conn),
        )
        await s.setup()
        yield s
    finally:
        await conn.close()


async def test_async_put_get_search(async_store: AsyncHanaStore) -> None:
    ns = ("async", "1")
    await async_store.aput(ns, "a", {"color": "red"})
    await async_store.aput(ns, "b", {"color": "blue"})

    item = await async_store.aget(ns, "a")
    assert item is not None and item.value == {"color": "red"}

    reds = await async_store.asearch(ns, filter={"color": "red"})
    assert {i.key for i in reds} == {"a"}

    await async_store.adelete(ns, "a")
    assert await async_store.aget(ns, "a") is None


async def test_async_batch_rejects_sync() -> None:
    conn = await dbapi.async_connect(**_conn_params())  # type: ignore[misc]
    try:
        s = AsyncHanaStore(
            conn, schema_name=config.schema_name, **_unique_tables(config.conn)
        )
        with pytest.raises(NotImplementedError):
            s.batch([])
    finally:
        await conn.close()

"""Integration tests for the async HANA memory store (live HANA Cloud).

Mirrors :mod:`test_hana_store.py` against
:class:`~langchain_hana.memorystores.AsyncHanaStore`, driven by an ``hdbcli`` async
connection (``dbapi.async_connect``). Requires
``HANA_DB_ADDRESS``/``HANA_DB_PORT``/``HANA_DB_USER``/``HANA_DB_PASSWORD`` and runs
against a dedicated throwaway schema.

Schema administration uses a blocking connection (``HanaTestUtils`` is synchronous);
the stores under test each drive their own async connection.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncGenerator, Generator
from typing import Any, TypedDict

import pytest
from hdbcli import dbapi
from langgraph.store.base import TTLConfig

from langchain_hana.memorystores import AsyncHanaStore
from tests.integration_tests.hana_test_utils import HanaTestUtils
from tests.integration_tests.store_embedding_configs import STORE_CONFIG_IDS, resolve


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


@pytest.fixture(scope="module", autouse=True)
def setup_schema() -> Generator[None, None, None]:
    config.conn = dbapi.connect(**_conn_params())
    schema_prefix = "LANGCHAIN_ASTORE"
    HanaTestUtils.drop_old_test_schemas(config.conn, schema_prefix)
    config.schema_name = HanaTestUtils.generate_schema_name(config.conn, schema_prefix)
    HanaTestUtils.create_and_set_schema(config.conn, config.schema_name)
    yield
    HanaTestUtils.drop_schema_if_exists(config.conn, config.schema_name)
    config.conn.close()


def _unique_tables(conn: dbapi.Connection) -> _Tables:
    suffix = HanaTestUtils.generate_schema_name(conn, "T").split("_")[-1][:16]
    return {
        "migrations_table": f"M_{suffix}",
        "store_table": f"S_{suffix}",
        "vectors_table": f"V_{suffix}",
    }


@pytest.fixture
async def store() -> AsyncGenerator[AsyncHanaStore, None]:
    """A plain (no-index, no-TTL) async store on a unique table set per test."""
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


@pytest.fixture(params=STORE_CONFIG_IDS)
async def indexed_store(
    request: pytest.FixtureRequest,
) -> AsyncGenerator[AsyncHanaStore, None]:
    """An indexed async store, parametrized over each embedding/column-type config."""
    index_config, vector_column_type = resolve(request.param, config.conn)
    conn = await dbapi.async_connect(**_conn_params())  # type: ignore[misc]
    try:
        s = AsyncHanaStore(
            conn,
            schema_name=config.schema_name,
            index=index_config,
            vector_column_type=vector_column_type,
            **_unique_tables(config.conn),
        )
        await s.setup()
        yield s
    finally:
        await conn.close()


# --------------------------------------------------------------------------- #
# Core KV
# --------------------------------------------------------------------------- #
async def test_put_get_delete(store: AsyncHanaStore) -> None:
    ns = ("users", "1")
    await store.aput(ns, "prefs", {"theme": "dark"})
    item = await store.aget(ns, "prefs")
    assert item is not None
    assert item.value == {"theme": "dark"}
    assert item.namespace == ns
    assert item.key == "prefs"

    await store.adelete(ns, "prefs")
    assert await store.aget(ns, "prefs") is None


async def test_overwrite_preserves_created_at(store: AsyncHanaStore) -> None:
    ns = ("docs",)
    await store.aput(ns, "a", {"v": 1})
    first = await store.aget(ns, "a")
    assert first is not None
    await asyncio.sleep(1.1)
    await store.aput(ns, "a", {"v": 2})
    second = await store.aget(ns, "a")
    assert second is not None
    assert second.value == {"v": 2}
    assert second.created_at == first.created_at
    assert second.updated_at > first.updated_at


async def test_get_missing_returns_none(store: AsyncHanaStore) -> None:
    assert await store.aget(("nope",), "x") is None


# --------------------------------------------------------------------------- #
# Search: metadata filtering + pagination
# --------------------------------------------------------------------------- #
async def test_search_filters_and_pagination(store: AsyncHanaStore) -> None:
    ns = ("items",)
    await store.aput(ns, "a", {"color": "red", "score": 1})
    await store.aput(ns, "b", {"color": "red", "score": 5})
    await store.aput(ns, "c", {"color": "blue", "score": 9})

    reds = await store.asearch(ns, filter={"color": "red"})
    assert {i.key for i in reds} == {"a", "b"}

    high = await store.asearch(ns, filter={"score": {"$gte": 5}})
    assert {i.key for i in high} == {"b", "c"}

    page = await store.asearch(ns, limit=1, offset=1)
    assert len(page) == 1

    nested = await store.asearch(ns, filter={"color": {"$ne": "blue"}})
    assert {i.key for i in nested} == {"a", "b"}


async def test_search_empty_prefix_returns_all_namespaces(
    store: AsyncHanaStore,
) -> None:
    # Empty prefix matches every namespace (regression: it used to match nothing).
    await store.aput(("users", "1"), "a", {"v": 1})
    await store.aput(("docs",), "b", {"v": 2})
    await store.aput(("x", "y", "z"), "c", {"v": 3})

    found = await store.asearch(())
    assert {i.key for i in found} == {"a", "b", "c"}


# --------------------------------------------------------------------------- #
# Namespace listing
# --------------------------------------------------------------------------- #
async def test_list_namespaces(store: AsyncHanaStore) -> None:
    await store.aput(("a", "b", "c"), "k", {"v": 1})
    await store.aput(("a", "b", "d"), "k", {"v": 1})
    await store.aput(("a", "x"), "k", {"v": 1})

    all_ns = await store.alist_namespaces()
    assert ("a", "b", "c") in all_ns

    prefixed = await store.alist_namespaces(prefix=("a", "b"))
    assert set(prefixed) == {("a", "b", "c"), ("a", "b", "d")}

    truncated = await store.alist_namespaces(prefix=("a",), max_depth=2)
    assert ("a", "b") in truncated and ("a", "x") in truncated

    suffixed = await store.alist_namespaces(suffix=("c",))
    assert suffixed == [("a", "b", "c")]


# --------------------------------------------------------------------------- #
# TTL
# --------------------------------------------------------------------------- #
async def test_ttl_expiry_refresh_and_sweep() -> None:
    conn = await dbapi.async_connect(**_conn_params())  # type: ignore[misc]
    try:
        ttl: TTLConfig = {"default_ttl": 0.02, "refresh_on_read": True}  # ~1.2s
        s = AsyncHanaStore(
            conn,
            schema_name=config.schema_name,
            ttl=ttl,
            **_unique_tables(config.conn),
        )
        await s.setup()
        ns = ("ephemeral",)
        await s.aput(ns, "k", {"v": 1})
        assert await s.aget(ns, "k", refresh_ttl=False) is not None
        await asyncio.sleep(2.0)
        # Expired rows are hidden from reads.
        assert await s.aget(ns, "k", refresh_ttl=False) is None
        removed = await s.asweep_ttl()
        assert removed >= 1
    finally:
        await conn.close()


# --------------------------------------------------------------------------- #
# Vector / semantic search
#
# These run over every embedding/column-type config in STORE_CONFIG_IDS via the
# parametrized ``indexed_store`` fixture.
# --------------------------------------------------------------------------- #
async def test_vector_search_ranks_by_similarity(
    indexed_store: AsyncHanaStore,
) -> None:
    s = indexed_store
    ns = ("memories",)
    await s.aput(ns, "cats", {"text": "cats"})
    await s.aput(ns, "dogs", {"text": "dogs"})
    await s.aput(ns, "birds", {"text": "birds"})

    results = await s.asearch(ns, query="cats")
    assert results
    assert results[0].key == "cats"
    assert results[0].score is not None


async def test_put_with_index_false_skips_embedding(
    indexed_store: AsyncHanaStore,
) -> None:
    s = indexed_store
    ns = ("memories",)
    await s.aput(ns, "x", {"text": "hello"}, index=False)
    # No embedding stored -> not returned by a vector query.
    assert await s.asearch(ns, query="hello") == []
    # But still retrievable by key and by metadata search.
    assert await s.aget(ns, "x") is not None


# --------------------------------------------------------------------------- #
# Async-specific behavior
# --------------------------------------------------------------------------- #
async def test_batch_rejects_sync() -> None:
    conn = await dbapi.async_connect(**_conn_params())  # type: ignore[misc]
    try:
        s = AsyncHanaStore(
            conn, schema_name=config.schema_name, **_unique_tables(config.conn)
        )
        with pytest.raises(NotImplementedError):
            s.batch([])
    finally:
        await conn.close()

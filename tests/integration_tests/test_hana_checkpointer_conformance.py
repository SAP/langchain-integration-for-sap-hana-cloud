"""LangGraph conformance tests for ``AsyncHanaCheckpointSaver``.

Runs LangGraph's official ``langgraph-checkpoint-conformance`` suite (async-only)
against the native-async HANA saver.
"""

import os
from collections.abc import AsyncGenerator, Generator
from typing import Any

import pytest
from hdbcli import dbapi
from langgraph.checkpoint.conformance import checkpointer_test, validate
from langgraph.checkpoint.conformance.capabilities import BASE_CAPABILITIES

from langchain_hana.checkpointers import AsyncHanaCheckpointSaver
from tests.integration_tests.hana_test_utils import HanaTestUtils


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
    # A sync admin connection creates/drops the dedicated conformance schema. The
    # saver under test uses its own async connection (``async_connect``), so the two
    # never share driver state.
    config.conn = dbapi.connect(**_conn_params())
    schema_prefix = "LANGCHAIN_CKPT_CONF"
    HanaTestUtils.drop_old_test_schemas(config.conn, schema_prefix)
    config.schema_name = HanaTestUtils.generate_schema_name(config.conn, schema_prefix)
    HanaTestUtils.create_and_set_schema(config.conn, config.schema_name)
    yield
    HanaTestUtils.drop_schema_if_exists(config.conn, config.schema_name)
    config.conn.close()


@checkpointer_test(name="hana-async")
async def hana_saver() -> AsyncGenerator[AsyncHanaCheckpointSaver, None]:
    """Fresh async saver per capability suite, all sharing the test schema.

    ``async_connect`` returns an ``AsyncConnection`` without flipping the global
    ``set_async_mode`` flag, so it does not disturb any synchronous connection. Each
    spec test scopes its assertions by a unique ``thread_id``, so a shared schema is
    safe across capability suites.
    """
    conn = await dbapi.async_connect(**_conn_params())  # type: ignore[misc]
    try:
        saver = AsyncHanaCheckpointSaver(conn, schema_name=config.schema_name)
        await saver.setup()
        yield saver
    finally:
        await conn.close()


async def test_base_conformance() -> None:
    report = await validate(
        hana_saver, capabilities={c.value for c in BASE_CAPABILITIES}
    )
    report.print_report()
    assert report.passed_all_base()

"""LangGraph conformance tests for ``AsyncHanaCheckpointSaver``.

Runs LangGraph's official ``langgraph-checkpoint-conformance`` suite (async-only)
against the native-async HANA saver.
"""

import os
import uuid
from collections.abc import AsyncGenerator, Generator
from typing import Any

import pytest
from hdbcli import dbapi
from langgraph.checkpoint.base import Checkpoint
from langgraph.checkpoint.conformance import checkpointer_test, validate
from langgraph.checkpoint.conformance.capabilities import BASE_CAPABILITIES
from langgraph.checkpoint.serde.types import TASKS
from langgraph.types import Send

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


async def test_pre_v4_pending_sends_migration() -> None:
    """A pre-v4 checkpoint surfaces its parent's ``TASKS`` writes as an inline channel.

    The official conformance suite never constructs the legacy shape (its generated
    checkpoints always have empty ``pending_sends``), so this verifies parity with the
    langgraph Postgres saver's ``_migrate_pending_sends``: ``Send`` objects stored as
    pending writes on the parent under ``__pregel_tasks`` must appear in a v<4 child's
    ``channel_values[TASKS]`` (in ``writes_sort_key`` order) on read.
    """
    conn = await dbapi.async_connect(**_conn_params())  # type: ignore[misc]
    try:
        saver = AsyncHanaCheckpointSaver(conn, schema_name=config.schema_name)
        await saver.setup()

        thread_id = f"pending-sends-{uuid.uuid4()}"
        ns = ""
        parent_id = str(uuid.uuid4())
        child_id = str(uuid.uuid4())

        def cfg(checkpoint_id: str | None) -> Any:
            configurable: dict[str, Any] = {"thread_id": thread_id, "checkpoint_ns": ns}
            if checkpoint_id is not None:
                configurable["checkpoint_id"] = checkpoint_id
            return {"configurable": configurable}

        def make_checkpoint(checkpoint_id: str, v: int) -> Checkpoint:
            return Checkpoint(
                v=v,
                id=checkpoint_id,
                ts="2024-01-01T00:00:00+00:00",
                channel_values={},
                channel_versions={},
                versions_seen={},
                updated_channels=None,
            )

        # Parent checkpoint (no parent of its own).
        await saver.aput(cfg(None), make_checkpoint(parent_id, 1), {}, {})
        # Pre-v4 pending sends: stored as TASKS writes on the parent checkpoint.
        sends = [Send("node_a", {"i": 1}), Send("node_b", {"i": 2})]
        await saver.aput_writes(
            cfg(parent_id),
            [(TASKS, s) for s in sends],
            task_id="task-1",
            task_path="path",
        )
        # Child checkpoint: v<4, with parent_checkpoint_id = parent_id.
        await saver.aput(cfg(parent_id), make_checkpoint(child_id, 1), {}, {})

        tup = await saver.aget_tuple(cfg(child_id))
        assert tup is not None
        assert tup.checkpoint["channel_values"].get(TASKS) == sends
        assert TASKS in tup.checkpoint["channel_versions"]
    finally:
        await conn.close()

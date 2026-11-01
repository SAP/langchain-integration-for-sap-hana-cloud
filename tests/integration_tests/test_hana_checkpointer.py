"""Integration tests for HanaCheckpointSaver (require a live SAP HANA Cloud DB).

Environment variables required:
    HANA_DB_ADDRESS, HANA_DB_PORT, HANA_DB_USER, HANA_DB_PASSWORD
"""

import os
from collections.abc import Generator
from typing import Any

import pytest
from hdbcli import dbapi
from langgraph.checkpoint.base import create_checkpoint, empty_checkpoint

from langchain_hana.checkpointers import HanaCheckpointSaver
from langchain_hana.utils import _table_exists
from tests.integration_tests.hana_test_utils import HanaTestUtils


class Config:
    conn: dbapi.Connection
    schema_name: str


config = Config()


@pytest.fixture(scope="module", autouse=True)
def setup_connection() -> Generator[None, None, None]:
    conn_params: dict[str, Any] = {
        "address": os.environ["HANA_DB_ADDRESS"],
        "port": int(os.environ["HANA_DB_PORT"]),
        "user": os.environ["HANA_DB_USER"],
        "password": os.environ["HANA_DB_PASSWORD"],
    }
    config.conn = dbapi.connect(**conn_params)
    schema_prefix = "LANGCHAIN_CKPT_TEST"
    HanaTestUtils.drop_old_test_schemas(config.conn, schema_prefix)
    config.schema_name = HanaTestUtils.generate_schema_name(config.conn, schema_prefix)
    HanaTestUtils.create_and_set_schema(config.conn, config.schema_name)
    yield
    HanaTestUtils.drop_schema_if_exists(config.conn, config.schema_name)
    config.conn.close()


@pytest.fixture
def saver() -> HanaCheckpointSaver:
    return HanaCheckpointSaver(config.conn, schema_name=config.schema_name)


def _config(thread_id: str, checkpoint_id: str | None = None) -> dict:
    configurable: dict[str, Any] = {"thread_id": thread_id, "checkpoint_ns": ""}
    if checkpoint_id is not None:
        configurable["checkpoint_id"] = checkpoint_id
    return {"configurable": configurable}


def _blob_count(saver: HanaCheckpointSaver, thread_id: str) -> int:
    cur = saver.connection.cursor()
    try:
        cur.execute(
            f'SELECT COUNT(*) FROM {saver._blobs_ref} WHERE "THREAD_ID" = ?',
            (thread_id,),
        )
        return int(cur.fetchone()[0])
    finally:
        cur.close()


def _max_migration(saver: HanaCheckpointSaver) -> int:
    cur = saver.connection.cursor()
    try:
        cur.execute(f'SELECT MAX("V") FROM {saver._migrations_ref}')
        return int(cur.fetchone()[0])
    finally:
        cur.close()


def test_setup_creates_tables_and_is_idempotent(saver: HanaCheckpointSaver) -> None:
    for table in (
        saver.migrations_table,
        saver.checkpoints_table,
        saver.blobs_table,
        saver.writes_table,
    ):
        assert _table_exists(saver.connection, table, saver.schema_name)
    # A second setup applies no new migrations (ledger version unchanged).
    version_before = _max_migration(saver)
    saver.setup()
    assert _max_migration(saver) == version_before


def test_unchanged_channel_reuses_blob_row(saver: HanaCheckpointSaver) -> None:
    base = _config("t-dedup")
    c1 = empty_checkpoint()
    c1["channel_values"] = {"msgs": ["a", "b"]}
    c1["channel_versions"] = {"msgs": "v1"}
    cfg1 = saver.put(base, c1, {"source": "input", "step": 1}, {"msgs": "v1"})

    # Second checkpoint leaves "msgs" unchanged (same value and version): no new
    # version is passed, so no additional blob row should be written.
    c2 = empty_checkpoint()
    c2["channel_values"] = {"msgs": ["a", "b"]}
    c2["channel_versions"] = {"msgs": "v1"}
    saver.put(cfg1, c2, {"source": "loop", "step": 2}, {})

    assert _blob_count(saver, "t-dedup") == 1
    latest = saver.get_tuple(_config("t-dedup"))
    assert latest is not None
    assert latest.checkpoint["channel_values"]["msgs"] == ["a", "b"]


def test_put_and_get_tuple_roundtrip(saver: HanaCheckpointSaver) -> None:
    chkpnt = empty_checkpoint()
    metadata = {"source": "input", "step": 1}
    saved = saver.put(_config("t-roundtrip"), chkpnt, metadata, {})

    got = saver.get_tuple(saved)
    assert got is not None
    assert got.checkpoint["id"] == chkpnt["id"]
    assert got.metadata["source"] == "input"
    assert got.metadata["step"] == 1


def test_get_latest_and_parent_chain(saver: HanaCheckpointSaver) -> None:
    base = _config("t-chain")
    c1 = empty_checkpoint()
    cfg1 = saver.put(base, c1, {"source": "input", "step": 1}, {})
    c2 = create_checkpoint(c1, {}, step=2)
    saver.put(cfg1, c2, {"source": "loop", "step": 2}, {})

    latest = saver.get_tuple(_config("t-chain"))
    assert latest is not None
    assert latest.checkpoint["id"] == c2["id"]
    assert latest.parent_config is not None
    assert latest.parent_config["configurable"]["checkpoint_id"] == c1["id"]


def test_list_ordering_limit_and_filter(saver: HanaCheckpointSaver) -> None:
    base = _config("t-list")
    c1 = empty_checkpoint()
    cfg1 = saver.put(base, c1, {"source": "input", "step": 1}, {})
    c2 = create_checkpoint(c1, {}, step=2)
    saver.put(cfg1, c2, {"source": "loop", "step": 2}, {})

    all_tuples = list(saver.list(_config("t-list")))
    assert [t.checkpoint["id"] for t in all_tuples] == [c2["id"], c1["id"]]

    limited = list(saver.list(_config("t-list"), limit=1))
    assert len(limited) == 1 and limited[0].checkpoint["id"] == c2["id"]

    filtered = list(saver.list(_config("t-list"), filter={"source": "loop"}))
    assert len(filtered) == 1 and filtered[0].checkpoint["id"] == c2["id"]


def test_put_writes_and_pending_writes(saver: HanaCheckpointSaver) -> None:
    chkpnt = empty_checkpoint()
    cfg = saver.put(_config("t-writes"), chkpnt, {"source": "input", "step": 1}, {})
    saver.put_writes(cfg, [("messages", "hello"), ("count", 2)], task_id="task-a")

    got = saver.get_tuple(cfg)
    assert got is not None
    assert got.pending_writes is not None
    channels = {channel: value for _, channel, value in got.pending_writes}
    assert channels["messages"] == "hello"
    assert channels["count"] == 2


def test_delete_thread(saver: HanaCheckpointSaver) -> None:
    chkpnt = empty_checkpoint()
    cfg = saver.put(_config("t-del"), chkpnt, {"source": "input", "step": 1}, {})
    saver.put_writes(cfg, [("messages", "bye")], task_id="task-a")

    saver.delete_thread("t-del")
    assert saver.get_tuple(_config("t-del")) is None
    assert list(saver.list(_config("t-del"))) == []


@pytest.mark.compile
def test_stategraph_smoke(saver: HanaCheckpointSaver) -> None:
    """Compile a StateGraph with the saver and run it end-to-end."""
    from typing import TypedDict

    from langgraph.graph import END, START, StateGraph

    class State(TypedDict):
        value: int

    builder = StateGraph(State)
    builder.add_node("inc", lambda s: {"value": s["value"] + 1})
    builder.add_edge(START, "inc")
    builder.add_edge("inc", END)
    graph = builder.compile(checkpointer=saver)

    thread = {"configurable": {"thread_id": "t-graph"}}
    result = graph.invoke({"value": 3}, thread)
    assert result["value"] == 4

    state = graph.get_state(thread)
    assert state.values["value"] == 4

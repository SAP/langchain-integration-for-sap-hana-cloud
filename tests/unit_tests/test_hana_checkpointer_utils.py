"""Unit tests for the HANA checkpointer SQL helpers."""

import pytest
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from langchain_hana.checkpointers import HanaCheckpointSaver


def test_build_where_clause_empty() -> None:
    where, params = HanaCheckpointSaver._build_where_clause(None, None, None)
    assert where == ""
    assert params == []


def test_build_where_clause_config_only() -> None:
    config = {"configurable": {"thread_id": "1", "checkpoint_ns": ""}}
    where, params = HanaCheckpointSaver._build_where_clause(config, None, None)
    assert where == 'WHERE "THREAD_ID" = ? AND "CHECKPOINT_NS" = ?'
    assert params == ["1", ""]


def test_build_where_clause_with_checkpoint_id_and_before() -> None:
    config = {
        "configurable": {
            "thread_id": "1",
            "checkpoint_ns": "",
            "checkpoint_id": "ckpt-2",
        }
    }
    before = {"configurable": {"checkpoint_id": "ckpt-5"}}
    where, params = HanaCheckpointSaver._build_where_clause(config, None, before)
    assert where == (
        'WHERE "THREAD_ID" = ? AND "CHECKPOINT_NS" = ? AND "CHECKPOINT_ID" = ? '
        'AND "CHECKPOINT_ID" < ?'
    )
    assert params == ["1", "", "ckpt-2", "ckpt-5"]


def test_build_where_clause_scalar_metadata_filter() -> None:
    where, params = HanaCheckpointSaver._build_where_clause(
        None, {"source": "loop", "step": 3}, None
    )
    assert "JSON_VALUE(\"METADATA\", '$.source') = ?" in where
    assert "JSON_VALUE(\"METADATA\", '$.step') = ?" in where
    assert params == ["loop", "3"]


def test_build_where_clause_bool_and_none_metadata_filter() -> None:
    where, params = HanaCheckpointSaver._build_where_clause(
        None, {"done": True, "err": None}, None
    )
    assert "JSON_VALUE(\"METADATA\", '$.done') = ?" in where
    assert "JSON_VALUE(\"METADATA\", '$.err') IS NULL" in where
    assert params == ["true"]  # None contributes no param


def test_build_where_clause_rejects_unsafe_filter_key() -> None:
    with pytest.raises(ValueError):
        HanaCheckpointSaver._build_where_clause(None, {"bad key; DROP": "x"}, None)


def test_build_where_clause_rejects_non_scalar_filter_value() -> None:
    with pytest.raises(ValueError):
        HanaCheckpointSaver._build_where_clause(None, {"nested": {"a": 1}}, None)


def test_pending_writes_sql_uses_table_ref() -> None:
    # Build an instance without running __init__ (which would need a live DB),
    # setting only the attributes _writes_ref depends on.
    saver = object.__new__(HanaCheckpointSaver)
    saver.schema_name = "S"
    saver.writes_table = "LANGGRAPH_WRITES"
    sql = saver._pending_writes_sql()
    assert '"S"."LANGGRAPH_WRITES"' in sql
    assert sql.startswith(
        'SELECT "TASK_ID", "CHANNEL", "TYPE", "VALUE", "TASK_PATH", "IDX"'
    )


def test_dump_blobs_serializes_and_placeholders() -> None:
    serde = JsonPlusSerializer()
    # "msgs" is non-primitive (in values); "step" is a primitive left inline
    # (only in versions, so it becomes an "empty" placeholder row).
    versions = {"msgs": "v1", "step": "v2"}
    values = {"msgs": {"a": 1}}
    rows = HanaCheckpointSaver._dump_blobs(serde, "t1", "", values, versions)

    by_channel = {
        channel: (thread, ns, ver, type_, blob)
        for thread, ns, channel, ver, type_, blob in rows
    }  # noqa: E501
    assert set(by_channel) == {"msgs", "step"}
    # placeholder row for the inline primitive
    _, _, step_ver, step_type, step_blob = by_channel["step"]
    assert (step_ver, step_type, step_blob) == ("v2", "empty", None)
    # serialized row for the non-primitive value
    _, _, msgs_ver, msgs_type, msgs_blob = by_channel["msgs"]
    assert msgs_ver == "v1" and msgs_type != "empty" and msgs_blob is not None


def test_dump_blobs_empty_versions_returns_empty() -> None:
    serde = JsonPlusSerializer()
    assert HanaCheckpointSaver._dump_blobs(serde, "t1", "", {"msgs": 1}, {}) == []


def test_load_blobs_skips_placeholders_and_deserializes() -> None:
    serde = JsonPlusSerializer()
    msgs_type, msgs_blob = serde.dumps_typed({"a": 1})
    # rows: (channel, type, blob) — "step" is an "empty" placeholder to skip
    rows = [
        ("msgs", msgs_type, msgs_blob),
        ("step", "empty", None),
    ]
    assert HanaCheckpointSaver._load_blobs(rows, serde) == {"msgs": {"a": 1}}


def test_load_pending_writes_orders_and_deserializes() -> None:
    serde = JsonPlusSerializer()
    t_a, b_a = serde.dumps_typed("alpha")
    t_b, b_b = serde.dumps_typed("beta")
    # rows: (task_id, channel, type, value, task_path, idx) — intentionally unordered
    rows = [
        ("task2", "messages", t_b, b_b, "~path", 0),
        ("task1", "messages", t_a, b_a, "", 1),
    ]
    result = HanaCheckpointSaver._load_pending_writes(rows, serde)
    # sorted by (task_path, task_id, idx): "" < "~path"
    assert result == [
        ("task1", "messages", "alpha"),
        ("task2", "messages", "beta"),
    ]


class _FakeLob:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data


def test_to_bytes_handles_variants() -> None:
    assert HanaCheckpointSaver._to_bytes(None) is None
    assert HanaCheckpointSaver._to_bytes(b"abc") == b"abc"
    assert HanaCheckpointSaver._to_bytes(bytearray(b"abc")) == b"abc"
    assert HanaCheckpointSaver._to_bytes(memoryview(b"abc")) == b"abc"
    assert HanaCheckpointSaver._to_bytes(_FakeLob(b"abc")) == b"abc"
    assert HanaCheckpointSaver._to_bytes("abc") == b"abc"


def test_to_str_handles_variants() -> None:
    assert HanaCheckpointSaver._to_str(None) is None
    assert HanaCheckpointSaver._to_str("abc") == "abc"
    assert HanaCheckpointSaver._to_str(b"abc") == "abc"
    assert HanaCheckpointSaver._to_str(_FakeLob(b"abc")) == "abc"

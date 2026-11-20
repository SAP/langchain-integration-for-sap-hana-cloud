"""Unit tests for the shared vector-column-type helpers in ``langchain_hana.utils``.

These back both the HanaDB vector store and the LangGraph memory stores, so they
are tested directly here (DB-free; the datatype-support path uses a fake cursor).
"""

import struct

import pytest

from langchain_hana.utils import (
    DEFAULT_VECTOR_COLUMN_TYPE,
    VECTOR_COLUMN_SQL_TYPES,
    _datatype_support_error,
    _deserialize_vector,
    _get_min_supported_version,
    _serialize_vector,
    _to_target_vector_type,
    _validate_datatype_support,
    _validate_vector_column_type,
)


# --------------------------------------------------------------------------- #
# validate / defaults
# --------------------------------------------------------------------------- #
def test_default_vector_column_type_is_real_vector() -> None:
    assert DEFAULT_VECTOR_COLUMN_TYPE == "REAL_VECTOR"
    assert set(VECTOR_COLUMN_SQL_TYPES) == {"REAL_VECTOR", "HALF_VECTOR"}


def test_validate_vector_column_type_uppercases_and_accepts() -> None:
    assert _validate_vector_column_type("real_vector") == "REAL_VECTOR"
    assert _validate_vector_column_type("Half_Vector") == "HALF_VECTOR"


def test_validate_vector_column_type_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Unsupported vector_column_type"):
        _validate_vector_column_type("BFLOAT16")


# --------------------------------------------------------------------------- #
# serialize / deserialize round trip
# --------------------------------------------------------------------------- #
def test_serialize_real_vector_is_4_bytes_per_element() -> None:
    values = [0.5, -1.25, 3.0]
    blob = _serialize_vector(values, "REAL_VECTOR")
    assert len(blob) == 4 + 3 * 4
    assert struct.unpack_from("<I", blob, 0)[0] == 3
    assert _deserialize_vector(memoryview(blob), "REAL_VECTOR") == pytest.approx(values)


def test_serialize_half_vector_is_2_bytes_per_element() -> None:
    values = [0.5, -1.25, 3.0]
    blob = _serialize_vector(values, "HALF_VECTOR")
    assert len(blob) == 4 + 3 * 2
    assert struct.unpack_from("<I", blob, 0)[0] == 3
    assert _deserialize_vector(memoryview(blob), "HALF_VECTOR") == pytest.approx(values)


def test_serialize_rejects_unknown_type() -> None:
    with pytest.raises(ValueError, match="Unsupported vector column type"):
        _serialize_vector([1.0], "BOGUS")


# --------------------------------------------------------------------------- #
# SQL conversion
# --------------------------------------------------------------------------- #
def test_to_target_vector_type_wraps_expression() -> None:
    assert _to_target_vector_type("?", "REAL_VECTOR") == "TO_REAL_VECTOR(?)"
    assert _to_target_vector_type("?", "HALF_VECTOR") == "TO_HALF_VECTOR(?)"


def test_to_target_vector_type_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Unsupported vector type"):
        _to_target_vector_type("?", "BOGUS")


# --------------------------------------------------------------------------- #
# version / datatype support
# --------------------------------------------------------------------------- #
def test_get_min_supported_version() -> None:
    assert "2025.15" in _get_min_supported_version("HALF_VECTOR")
    assert "2024.2" in _get_min_supported_version("REAL_VECTOR")
    with pytest.raises(ValueError, match="Unknown datatype"):
        _get_min_supported_version("BOGUS")


def test_datatype_support_error_includes_version_when_known() -> None:
    msg = _datatype_support_error("HALF_VECTOR", "2024.4")
    assert "HALF_VECTOR" in msg
    assert "Instance version: 2024.4" in msg
    assert "2025.15" in msg


def test_datatype_support_error_omits_version_when_unknown() -> None:
    msg = _datatype_support_error("HALF_VECTOR", None)
    assert "Instance version:" not in msg
    assert "2025.15" in msg


class _FakeCursor:
    """Minimal hdbcli-cursor stand-in driven by a {sql_substring: rows} script."""

    def __init__(self, script: dict[str, list[tuple]]) -> None:
        self._script = script
        self._rows: list[tuple] = []

    def execute(self, sql: str, params: object = None) -> None:
        self._rows = []
        for key, rows in self._script.items():
            if key in sql:
                self._rows = rows
                return

    def has_result_set(self) -> bool:
        return True

    def fetchall(self) -> list[tuple]:
        return self._rows

    def fetchone(self) -> tuple | None:
        return self._rows[0] if self._rows else None

    def close(self) -> None:
        pass


class _FakeConn:
    def __init__(self, script: dict[str, list[tuple]]) -> None:
        self._script = script

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self._script)


def test_validate_datatype_support_passes_when_available() -> None:
    conn = _FakeConn({"SYS.DATA_TYPES": [("REAL_VECTOR",), ("HALF_VECTOR",)]})
    assert _validate_datatype_support(conn, "HALF_VECTOR") is True  # type: ignore[arg-type]


def test_validate_datatype_support_raises_with_version_when_missing() -> None:
    conn = _FakeConn(
        {"SYS.DATA_TYPES": [("REAL_VECTOR",)], "SYS.M_DATABASE": [("2024.4",)]}
    )
    with pytest.raises(ValueError) as exc:
        _validate_datatype_support(conn, "HALF_VECTOR")  # type: ignore[arg-type]
    msg = str(exc.value)
    assert "HALF_VECTOR" in msg
    assert "2024.4" in msg
    assert "2025.15" in msg

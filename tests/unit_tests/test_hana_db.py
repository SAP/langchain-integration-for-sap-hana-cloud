"""Test HanaVector functionality."""

from unittest.mock import Mock, patch

import pytest

from langchain_hana import HanaDB, HanaInternalEmbeddings
from langchain_hana.utils import _validate_k, _validate_k_and_fetch_k
from langchain_hana.vectorstores.utils import _validate_identifier


def test_int_sanitation_with_illegal_value() -> None:
    """Test sanitization of int with illegal value"""
    successful = True
    try:
        HanaDB._sanitize_int("HUGO")
        successful = False
    except ValueError:
        pass

    assert successful


def test_int_sanitation_with_legal_values() -> None:
    """Test sanitization of int with legal values"""
    assert HanaDB._sanitize_int(42) == 42

    assert HanaDB._sanitize_int("21") == 21


def test_int_sanitation_with_negative_values() -> None:
    """Test sanitization of int with legal values"""
    assert HanaDB._sanitize_int(-1) == -1

    assert HanaDB._sanitize_int("-1") == -1


def test_int_sanitation_with_illegal_negative_value() -> None:
    """Test sanitization of int with illegal value"""
    successful = True
    try:
        HanaDB._sanitize_int(-2)
        successful = False
    except ValueError:
        pass

    assert successful


def dummy_similarity_search(query: str, k: int = 4) -> str:
    _validate_k(k)
    return f"Query: {query}, k={k}"


@pytest.mark.parametrize(
    "query, k, expected",
    [
        ("apple", None, "Query: apple, k=4"),
        ("banana", 3, "Query: banana, k=3"),
        ("cherry", 2, "Query: cherry, k=2"),
    ],
)
def test_similarity_search_valid(query: str, k: int | None, expected: str) -> None:
    if k is None:
        result = dummy_similarity_search(query)
    else:
        result = dummy_similarity_search(query, k)
    assert result == expected


@pytest.mark.parametrize(
    "query, k",
    [
        ("orange", 0),
        ("mango", -1),
    ],
)
def test_similarity_search_invalid(query: str, k: int) -> None:
    with pytest.raises(ValueError, match="must be an integer greater than 0"):
        dummy_similarity_search(query, k=k)


def dummy_max_marginal_relevance_search(
    query: str, k: int = 4, fetch_k: int = 10
) -> str:
    _validate_k_and_fetch_k(k, fetch_k)
    return f"Query: {query}, k={k}, fetch_k={fetch_k}"


@pytest.mark.parametrize(
    "query, k, fetch_k, expected",
    [
        ("apple", None, None, "Query: apple, k=4, fetch_k=10"),
        ("banana", 3, 5, "Query: banana, k=3, fetch_k=5"),
        ("cherry", 2, 2, "Query: cherry, k=2, fetch_k=2"),
    ],
)
def test_max_marginal_relevance_search_valid(
    query: str, k: int | None, fetch_k: int | None, expected: str
) -> None:
    if k is None and fetch_k is None:
        result = dummy_max_marginal_relevance_search(query)
    elif fetch_k is None:
        result = dummy_max_marginal_relevance_search(query, k)  # type: ignore[arg-type]
    else:
        result = dummy_max_marginal_relevance_search(query, k, fetch_k)  # type: ignore[arg-type]
    assert result == expected


@pytest.mark.parametrize(
    "query, k, fetch_k, match",
    [
        ("orange", 0, 5, "must be an integer greater than 0"),
        ("mango", -1, 5, "must be an integer greater than 0"),
        ("grape", 5, 3, "greater than or equal to 'k'"),
    ],
)
def test_max_marginal_relevance_search_invalid(
    query: str, k: int, fetch_k: int, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        dummy_max_marginal_relevance_search(query, k=k, fetch_k=fetch_k)


# ---------------------------------------------------------------------------
# SQL injection guard: remote_source and remote_source_schema
# ---------------------------------------------------------------------------

_BAD_IDENTIFIERS = [
    "schema'; DROP TABLE users--",
    "schema OR 1=1",
    "schema name",
    "1schema",
    "schema-name",
    "schema.name",
    "col\n",
    "",
]

_VALID_IDENTIFIERS = [
    "valid_col",
    "_leading_underscore",
    "Col123",
    "a",
]


@pytest.mark.parametrize("name", _VALID_IDENTIFIERS)
def test_validate_identifier_accepts_valid_names(name: str) -> None:
    _validate_identifier(name)  # must not raise


@pytest.mark.parametrize("name", _BAD_IDENTIFIERS)
def test_validate_identifier_rejects_invalid_names(name: str) -> None:
    with pytest.raises(ValueError):
        _validate_identifier(name)


def _make_hana_db(embedding: HanaInternalEmbeddings) -> HanaDB:
    with (
        patch.object(HanaDB, "_initialize_table"),
        patch.object(HanaDB, "_validate_internal_embedding_function"),
        patch.object(
            HanaDB, "_sanitize_vector_column_type", return_value="REAL_VECTOR"
        ),
    ):
        return HanaDB(connection=Mock(), embedding=embedding)


@pytest.mark.parametrize("bad_value", [v for v in _BAD_IDENTIFIERS if v])
def test_invalid_remote_source_schema_raises(bad_value: str) -> None:
    embedding = HanaInternalEmbeddings(
        internal_embedding_model_id="model",
        remote_source_schema=bad_value,
        remote_source="valid_source",
    )
    with pytest.raises(ValueError, match="Invalid identifier"):
        _make_hana_db(embedding)


@pytest.mark.parametrize("bad_value", [v for v in _BAD_IDENTIFIERS if v])
def test_invalid_remote_source_raises(bad_value: str) -> None:
    embedding = HanaInternalEmbeddings(
        internal_embedding_model_id="model",
        remote_source_schema="valid_schema",
        remote_source=bad_value,
    )
    with pytest.raises(ValueError, match="Invalid identifier"):
        _make_hana_db(embedding)


@pytest.mark.parametrize(
    "schema, source",
    [(v, v) for v in _VALID_IDENTIFIERS],
)
def test_valid_remote_source_and_schema_accepted(schema: str, source: str) -> None:
    embedding = HanaInternalEmbeddings(
        internal_embedding_model_id="model",
        remote_source_schema=schema,
        remote_source=source,
    )
    db = _make_hana_db(embedding)
    assert db.internal_embedding_remote_source_schema == schema
    assert db.internal_embedding_remote_source == source

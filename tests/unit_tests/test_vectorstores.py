"""Test HanaVector functionality."""

import pytest

from langchain_hana.utils import _validate_k, _validate_k_and_fetch_k
from langchain_hana.vectorstores import HanaDB


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


def dummy_similarity_search(query, k=4):
    _validate_k(k)
    return f"Query: {query}, k={k}"


@pytest.mark.parametrize(
    "query, k, expected",
    [
        ("apple", None, "Query: apple, k=4"),
        ("banana", 3, "Query: banana, k=3"),
        ("cherry", 2, "Query: cherry, k=2"),
    ]
)
def test_similarity_search_valid(query, k, expected):
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
    ]
)
def test_similarity_search_invalid(query, k):
    with pytest.raises(ValueError, match="must be an integer greater than 0"):
        dummy_similarity_search(query, k=k)


def dummy_max_marginal_relevance_search(query, k=4, fetch_k=10):
    _validate_k_and_fetch_k(k, fetch_k)
    return f"Query: {query}, k={k}, fetch_k={fetch_k}"


@pytest.mark.parametrize(
    "query, k, fetch_k, expected",
    [
        ("apple", None, None, "Query: apple, k=4, fetch_k=10"),
        ("banana", 3, 5, "Query: banana, k=3, fetch_k=5"),
        ("cherry", 2, 2, "Query: cherry, k=2, fetch_k=2"),
    ]
)
def test_max_marginal_relevance_search_valid(query, k, fetch_k, expected):
    if k is None and fetch_k is None:
        result = dummy_max_marginal_relevance_search(query)
    elif fetch_k is None:
        result = dummy_max_marginal_relevance_search(query, k)
    else:
        result = dummy_max_marginal_relevance_search(query, k, fetch_k)
    assert result == expected


@pytest.mark.parametrize(
    "query, k, fetch_k, match",
    [
        ("orange", 0, 5, "must be an integer greater than 0"),
        ("mango", -1, 5, "must be an integer greater than 0"),
        ("grape", 5, 3, "greater than or equal to 'k'"),
    ]
)
def test_max_marginal_relevance_search_invalid(query, k, fetch_k, match):
    with pytest.raises(ValueError, match=match):
        dummy_max_marginal_relevance_search(query, k=k, fetch_k=fetch_k)


def test_parse_float_array_from_string() -> None:
    array_as_string = "[0.1, 0.2, 0.3]"
    assert HanaDB._parse_float_array_from_string(array_as_string) == [0.1, 0.2, 0.3]

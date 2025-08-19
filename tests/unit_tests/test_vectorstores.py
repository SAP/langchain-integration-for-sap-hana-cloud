"""Test HanaVector functionality."""

import pytest

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


@HanaDB._validate_k
def dummy_similarity_search(query, k=4):
    return f"Query: {query}, k={k}"


def test_similarity_search_valid_defaults():
    assert dummy_similarity_search("apple") == "Query: apple, k=4"


def test_similarity_search_valid_positional():
    assert dummy_similarity_search("banana", 3) == "Query: banana, k=3"


def test_similarity_search_valid_keyword():
    assert dummy_similarity_search("cherry", k=2) == "Query: cherry, k=2"


def test_similarity_search_invalid_k_zero():
    with pytest.raises(ValueError, match="must be an integer greater than 0"):
        dummy_similarity_search("orange", k=0)


def test_similarity_search_invalid_k_negative():
    with pytest.raises(ValueError, match="must be an integer greater than 0"):
        dummy_similarity_search("mango", k=-1)


@HanaDB._validate_k_and_fetch_k
def dummy_max_marginal_relevance_search(query, k=4, fetch_k=10):
    return f"Query: {query}, k={k}, fetch_k={fetch_k}"


def test_max_marginal_relevance_search_valid_defaults():
    assert (
        dummy_max_marginal_relevance_search("apple") == "Query: apple, k=4, fetch_k=10"
    )


def test_max_marginal_relevance_search_valid_positional():
    assert (
        dummy_max_marginal_relevance_search("banana", 3, 5)
        == "Query: banana, k=3, fetch_k=5"
    )


def test_max_marginal_relevance_search_valid_keyword():
    assert (
        dummy_max_marginal_relevance_search("cherry", k=2, fetch_k=2)
        == "Query: cherry, k=2, fetch_k=2"
    )


def test_max_marginal_relevance_search_invalid_k_zero():
    with pytest.raises(ValueError, match="must be an integer greater than 0"):
        dummy_max_marginal_relevance_search("orange", k=0, fetch_k=5)


def test_max_marginal_relevance_search_invalid_k_negative():
    with pytest.raises(ValueError, match="must be an integer greater than 0"):
        dummy_max_marginal_relevance_search("mango", k=-1, fetch_k=5)


def test_max_marginal_relevance_search_invalid_fetch_k_less_than_k():
    with pytest.raises(ValueError, match="greater than or equal to 'k'"):
        dummy_max_marginal_relevance_search("grape", k=5, fetch_k=3)


def test_parse_float_array_from_string() -> None:
    array_as_string = "[0.1, 0.2, 0.3]"
    assert HanaDB._parse_float_array_from_string(array_as_string) == [0.1, 0.2, 0.3]

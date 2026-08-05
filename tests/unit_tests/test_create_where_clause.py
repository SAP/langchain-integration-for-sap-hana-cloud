import re
from typing import Any, Dict, List

import pytest

from langchain_hana.vectorstores.create_where_clause import CreateWhereClause
from langchain_hana.vectorstores.hana_db import default_metadata_column
from tests.fixtures.filtering_test_cases import (
    ERROR_FILTERING_TEST_CASES,
    FILTERING_TEST_CASES,
)


class MockHanaDb:
    def __init__(self) -> None:
        self.metadata_column = default_metadata_column
        self.specific_metadata_columns: list[str] = []


def test_create_where_clause_empty_filter() -> None:
    where_clause, parameters = CreateWhereClause(MockHanaDb())({})
    assert where_clause == ""
    assert parameters == ()


@pytest.mark.parametrize(
    "test_filter, expected_ids, expected_where_clause, "
    "expected_where_clause_parameters",
    FILTERING_TEST_CASES,
)
def test_create_where_clause(
    test_filter: Dict[str, Any],
    expected_ids: List[int],
    expected_where_clause: str,
    expected_where_clause_parameters: List[Any],
) -> None:
    where_clause, parameters = CreateWhereClause(MockHanaDb())(test_filter)
    assert expected_where_clause == where_clause
    assert expected_where_clause_parameters == parameters


@pytest.mark.parametrize(
    "test_filter, expected_exception_message",
    ERROR_FILTERING_TEST_CASES,
)
def test_create_where_clause_invalid_filters(
    test_filter: Dict[str, Any],
    expected_exception_message: str,
) -> None:
    with pytest.raises(ValueError, match=re.escape(expected_exception_message)):
        CreateWhereClause(MockHanaDb())(test_filter)


# ---------------------------------------------------------------------------
# SQL injection guard tests
# ---------------------------------------------------------------------------

_INJECTION_COLUMN_NAMES = [
    "col') FROM DUAL UNION SELECT * FROM SYS.USERS--",
    'col"evil',
    "col; DROP TABLE users--",
    "col OR 1=1",
    "col/*comment*/",
    "1col",  # starts with digit
    "col name",  # space
    "col-name",  # hyphen
    "col.name",  # dot
]


@pytest.mark.parametrize("bad_column", _INJECTION_COLUMN_NAMES)
def test_sql_injection_via_filter_key_raises(bad_column: str) -> None:
    """Malicious filter keys must raise ValueError, not reach the query string."""
    with pytest.raises(ValueError, match="Invalid identifier"):
        CreateWhereClause(MockHanaDb())({bad_column: "value"})


@pytest.mark.parametrize("bad_column", _INJECTION_COLUMN_NAMES)
def test_sql_injection_via_contains_filter_key_raises(bad_column: str) -> None:
    """Malicious column names in $contains must raise ValueError."""
    with pytest.raises(ValueError, match="Invalid identifier"):
        CreateWhereClause(MockHanaDb())({bad_column: {"$contains": "search term"}})


@pytest.mark.parametrize(
    "good_column",
    ["name", "my_column", "_private", "col123", "CamelCase"],
)
def test_valid_column_names_accepted(good_column: str) -> None:
    """Valid identifier column names must not raise."""
    clause, params = CreateWhereClause(MockHanaDb())({good_column: "value"})
    assert good_column in clause

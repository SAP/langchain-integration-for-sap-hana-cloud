import pytest
import re
from typing import List, Dict, Any
from langchain_hana.vectorstores.create_where_clause import CreateWhereClause 
from langchain_hana.vectorstores.hana_db import default_metadata_column
from tests.integration_tests.fixtures.filtering_test_cases import FILTERING_TEST_CASES, ERROR_FILTERING_TEST_CASES

class MockHanaDb:
    def __init__(self):
        self.metadata_column = default_metadata_column
        self.specific_metadata_columns = []


def test_create_where_clause_empty_filter() -> None:
    where_clause, parameters = CreateWhereClause(MockHanaDb())({})
    assert where_clause == ""
    assert parameters == []

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

@pytest.mark.parametrize(
    "test_filter, expected_ids, expected_where_clause, expected_where_clause_parameters",
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

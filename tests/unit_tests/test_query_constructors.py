from typing import Any, Dict, List, Tuple

import pytest as pytest
from langchain_core.structured_query import (
    Comparator,
    Comparison,
    Operation,
    Operator,
    StructuredQuery,
)

from langchain_hana.query_constructors import CreateWhereClause, HanaTranslator
from langchain_hana.vectorstores import default_metadata_column
from tests.integration_tests.fixtures.filtering_test_cases import FILTERING_TEST_CASES

DEFAULT_TRANSLATOR = HanaTranslator()


class MockHanaDb:
    def __init__(self):
        self.metadata_column = default_metadata_column
        self.specific_metadata_columns = []


def test_visit_comparison() -> None:
    comp = Comparison(comparator=Comparator.LT, attribute="foo", value=1)
    expected = {"foo": {"$lt": 1}}
    actual = DEFAULT_TRANSLATOR.visit_comparison(comp)
    assert expected == actual


def test_visit_operation() -> None:
    op = Operation(
        operator=Operator.AND,
        arguments=[
            Comparison(comparator=Comparator.LT, attribute="foo", value=2),
            Comparison(comparator=Comparator.EQ, attribute="bar", value="baz"),
            Comparison(comparator=Comparator.GT, attribute="abc", value=2.0),
        ],
    )
    expected = {
        "$and": [{"foo": {"$lt": 2}}, {"bar": {"$eq": "baz"}}, {"abc": {"$gt": 2.0}}]
    }
    actual = DEFAULT_TRANSLATOR.visit_operation(op)
    assert expected == actual


def test_visit_structured_query() -> None:
    query = "What is the capital of France?"
    structured_query = StructuredQuery(
        query=query,
        filter=None,
    )
    expected: Tuple[str, Dict] = (query, {})
    actual = DEFAULT_TRANSLATOR.visit_structured_query(structured_query)
    assert expected == actual

    comp = Comparison(comparator=Comparator.LT, attribute="foo", value=1)
    structured_query = StructuredQuery(
        query=query,
        filter=comp,
    )
    expected = (query, {"filter": {"foo": {"$lt": 1}}})
    actual = DEFAULT_TRANSLATOR.visit_structured_query(structured_query)
    assert expected == actual

    op = Operation(
        operator=Operator.AND,
        arguments=[
            Comparison(comparator=Comparator.LT, attribute="foo", value=2),
            Comparison(comparator=Comparator.EQ, attribute="bar", value="baz"),
            Comparison(comparator=Comparator.GT, attribute="abc", value=2.0),
        ],
    )
    structured_query = StructuredQuery(
        query=query,
        filter=op,
    )
    expected = (
        query,
        {
            "filter": {
                "$and": [
                    {"foo": {"$lt": 2}},
                    {"bar": {"$eq": "baz"}},
                    {"abc": {"$gt": 2.0}},
                ]
            }
        },
    )
    actual = DEFAULT_TRANSLATOR.visit_structured_query(structured_query)
    assert expected == actual


def test_create_where_clause_empty_filter() -> None:
    where_clause, parameters = CreateWhereClause(MockHanaDb())({})
    assert where_clause == ""
    assert parameters == []


def test_create_where_clause_unexpected_operator() -> None:
    invalid_filter = {"$eq": [{"key": "value"}]}
    with pytest.raises(ValueError, match="Unexpected operator"):
        CreateWhereClause(MockHanaDb())(invalid_filter)


def test_create_where_clause_unsupported_filter_value_type() -> None:
    unsupported_filter = {"key": [1, 2, 3]}
    with pytest.raises(ValueError, match="Unsupported filter value type"):
        CreateWhereClause(MockHanaDb())(unsupported_filter)


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

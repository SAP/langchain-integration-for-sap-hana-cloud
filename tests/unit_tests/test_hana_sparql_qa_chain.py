"""Unit tests for HanaSparqlQAChain."""

import pytest
from langchain_hana import HanaSparqlQAChain


@pytest.mark.parametrize(
    "input_query,expected_result,test_case_name",
    [
        (
            "```sparql\nSELECT * WHERE { ?s ?p ?o }\n```",
            "SELECT * WHERE { ?s ?p ?o }",
            "lowercase_sparql",
        ),
        (
            "```SPARQL\nSELECT * WHERE { ?s ?p ?o }\n```",
            "SELECT * WHERE { ?s ?p ?o }",
            "uppercase_SPARQL",
        ),
        (
            "```Sparql\nSELECT * WHERE { ?s ?p ?o }\n```",
            "SELECT * WHERE { ?s ?p ?o }",
            "mixed_case_Sparql",
        ),
        (
            "```SparQL\nSELECT * WHERE { ?s ?p ?o }\n```",
            "SELECT * WHERE { ?s ?p ?o }",
            "mixed_case_SparQL",
        ),
        (
            "```\nSELECT * WHERE { ?s ?p ?o }\n```",
            "SELECT * WHERE { ?s ?p ?o }",
            "fenced_no_language",
        ),
        (
            "<sparql>\nSELECT * WHERE { ?s ?p ?o }\n</sparql>",
            "SELECT * WHERE { ?s ?p ?o }",
            "xml_tags",
        ),
        (
            "SELECT * WHERE { ?s ?p ?o }",
            "SELECT * WHERE { ?s ?p ?o }",
            "plain_query",
        ),
        ("", "", "empty_string"),
        ("   \n\t  ", "", "whitespace_only"),
    ],
)
def test_extract_sparql_parameterized(
    input_query: str, expected_result: str, test_case_name: str
) -> None:
    """Parameterized test for extract_sparql method covering various input formats.

    This test verifies the fix for issue #47 and ensures all supported
    formats work correctly.
    """
    result = HanaSparqlQAChain.extract_sparql(input_query)
    assert (
        result.strip() == expected_result.strip()
    ), f"Failed for test case: {test_case_name}"

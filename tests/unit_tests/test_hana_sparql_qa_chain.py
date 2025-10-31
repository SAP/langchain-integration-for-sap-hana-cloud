import pytest
from langchain_hana.chains.graph_qa.hana_sparql_qa_chain import HanaSparqlQAChain


@pytest.mark.xfail(reason="Issue #47: uppercase 'SPARQL' is not properly extracted from fenced code blocks")
def test_extract_sparql_uppercase_issue_47() -> None:
    """Test that demonstrates issue #47: uppercase 'SPARQL' in fenced code blocks fails to extract properly.
    
    The issue occurs when LLM returns SPARQL code with uppercase 'SPARQL' language identifier
    in the fenced code block. The current implementation only handles lowercase 'sparql'.
    """
    # This is the exact example from issue #47
    uppercase_query = """```SPARQL
PREFIX schema: <http://schema.org/>

SELECT DISTINCT ?personName
FROM <teched2025_devkeynote>
WHERE {
    ?person a schema:Person .
    ?person schema:name ?personName .
}
```"""
    
    expected_sparql = """PREFIX schema: <http://schema.org/>

SELECT DISTINCT ?personName
FROM <teched2025_devkeynote>
WHERE {
    ?person a schema:Person .
    ?person schema:name ?personName .
}"""
    
    result = HanaSparqlQAChain.extract_sparql(uppercase_query)
    # This will fail with current implementation - it returns "SPARQL\nPREFIX..." instead of "PREFIX..."
    # causing HANA db execution to fail
    assert result.strip() == expected_sparql.strip()
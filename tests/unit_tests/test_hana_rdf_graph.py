"""Test HANA RDF Graph unit tests."""

from unittest.mock import Mock, patch
import rdflib
from langchain_hana import HanaRdfGraph


def test_get_schema_returns_graph_not_string_issue_45():
    """Test that verifies the fix for GitHub issue #45"""
    # Mock the database connection since this is a unit test
    mock_connection = Mock()

    # Create a minimal schema graph for testing
    test_schema = rdflib.Graph()
    test_schema.add((
        rdflib.URIRef("http://example.org/Person"),
        rdflib.RDF.type,
        rdflib.RDFS.Class
    ))

    # Mock the HanaRdfGraph to avoid database dependency
    with patch.object(
        HanaRdfGraph, '_load_ontology_schema_graph_from_query', return_value=test_schema
    ), patch.object(HanaRdfGraph, '_validate_construct_query'):

        graph = HanaRdfGraph(
            connection=mock_connection,
            auto_extract_ontology=True,
        )

        schema_graph = graph.get_schema
        assert isinstance(schema_graph, rdflib.Graph), (
            "graph.get_schema returns a string instead of a rdflib.Graph object"
        )

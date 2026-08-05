"""Test HANA RDF Graph unit tests."""

from typing import Any, Optional
from unittest.mock import Mock, patch

import pytest
import rdflib

from langchain_hana import HanaRdfGraph


def _mock_graph(
    mock_connection: Mock,
    graph_uri: str = "",
    ontology_uri: Optional[str] = None,
    auto_extract_ontology: bool = True,
) -> HanaRdfGraph:
    """Return a HanaRdfGraph with all DB calls patched out."""
    test_schema = rdflib.Graph()
    kwargs: dict[str, Any] = dict(connection=mock_connection)
    if graph_uri:
        kwargs["graph_uri"] = graph_uri
    if ontology_uri is not None:
        kwargs["ontology_uri"] = ontology_uri
        auto_extract_ontology = False
    kwargs["auto_extract_ontology"] = auto_extract_ontology

    with patch.object(
        HanaRdfGraph, "_load_ontology_schema_graph_from_query", return_value=test_schema
    ), patch.object(HanaRdfGraph, "_validate_construct_query"):
        return HanaRdfGraph(**kwargs)


def test_get_schema_returns_graph_not_string_issue_45() -> None:
    """Test that verifies the fix for GitHub issue #45"""
    graph = _mock_graph(Mock())
    schema_graph = graph.get_schema
    assert isinstance(
        schema_graph, rdflib.Graph
    ), "graph.get_schema returns a string instead of a rdflib.Graph object"


# ---------------------------------------------------------------------------
# IRI injection guard tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_uri",
    [
        "http://example.com/graph> WHERE {?s ?p ?o} #",
        'http://example.com/graph"evil',
        "http://example.com/{graph}",
        "http://example.com/graph|pipe",
        "http://example.com/graph\\backslash",
        "http://example.com/graph^caret",
        "http://example.com/graph`tick",
        "http://example.com/graph\x00null",
        "http://example.com/graph\x01control",
    ],
)
def test_validate_iri_raises_on_forbidden_chars(bad_uri: str) -> None:
    with pytest.raises(ValueError, match="Invalid IRI"):
        HanaRdfGraph._validate_iri(bad_uri)


@pytest.mark.parametrize(
    "good_uri",
    [
        "http://example.com/graph",
        "http://example.com/ontology#Class",
        "urn:example:graph",
        "https://dbpedia.org/ontology/Person",
    ],
)
def test_validate_iri_accepts_valid_uris(good_uri: str) -> None:
    HanaRdfGraph._validate_iri(good_uri)  # must not raise


def test_graph_uri_with_injection_raises_on_init() -> None:
    with pytest.raises(ValueError, match="Invalid IRI"):
        HanaRdfGraph(
            connection=Mock(),
            graph_uri="http://x.com/g> UNION SELECT * WHERE {?s ?p ?o}",
            auto_extract_ontology=True,
        )


def test_graph_uri_valid_sets_from_clause() -> None:
    graph = _mock_graph(Mock(), graph_uri="http://example.com/mygraph")
    assert graph.from_clause == "FROM <http://example.com/mygraph>"


def test_ontology_uri_with_injection_raises() -> None:
    with pytest.raises(ValueError, match="Invalid IRI"):
        _mock_graph(Mock(), ontology_uri="http://x.com/ont> INJECT")


def test_ontology_uri_valid_does_not_raise() -> None:
    _mock_graph(Mock(), ontology_uri="http://example.com/ontology")  # must not raise


# ---------------------------------------------------------------------------
# HTTP header injection guard tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_content_type",
    [
        "text/csv\r\nX-Injected: evil",
        "text/csv\nX-Injected: evil",
        "text/csv\rX-Injected: evil",
    ],
)
def test_query_raises_on_crlf_in_content_type(bad_content_type: str) -> None:
    graph = _mock_graph(Mock())
    with pytest.raises(ValueError, match="CR/LF"):
        graph.query("SELECT ?s WHERE { ?s ?p ?o }", content_type=bad_content_type)


def test_query_accepts_valid_content_type() -> None:
    mock_connection = Mock()
    mock_cursor = Mock()
    mock_cursor.callproc.return_value = (None, None, "s\nhttp://example.com/a\n")
    mock_connection.cursor.return_value = mock_cursor

    graph = _mock_graph(mock_connection)
    result = graph.query(
        "SELECT ?s WHERE { ?s ?p ?o }",
        content_type="application/sparql-results+csv",
        inject_from_clause=False,
    )
    assert isinstance(result, str)

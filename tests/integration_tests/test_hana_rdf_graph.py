"""Test HANA Rdf Graph functionality."""

import os
import pytest
import rdflib
from rdflib.compare import isomorphic
from pathlib import Path
import textwrap
from hdbcli import dbapi
from langchain_hana import HanaRdfGraph
from tests.integration_tests.hana_test_utils import HanaTestUtils

class Config:
    def __init__(self):  # type: ignore[no-untyped-def]
        self.conn = None

config = Config()
        

def setup_module(module):  # type: ignore[no-untyped-def]
    
    config.conn = dbapi.connect(
        address=os.environ["HANA_DB_ADDRESS"],
        port=os.environ["HANA_DB_PORT"],
        user=os.environ["HANA_DB_USER"],
        password=os.environ["HANA_DB_PASSWORD"],
        autocommit=True,
        sslValidateCertificate=False,
        # encrypt=True
    )
    
def teardown_module(module):  # type: ignore[no-untyped-def]
    config.conn.close()
    config.conn = None

@pytest.fixture
def default_graph():
    return HanaRdfGraph(
        connection=config.conn,
        auto_extract_ontology=True,
    )

@pytest.fixture(params=["DEFAULT", None])
def default_graph_with_graph_uri(request):
    return HanaRdfGraph(
        connection=config.conn,
        graph_uri=request.param,
        auto_extract_ontology=True,
    )

@pytest.fixture
def default_graph_with_ontology_uri():
    ontology_uri = "http://example.com/ontology"
    query = f"""
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
    PREFIX ex: <http://example.com/>

    INSERT DATA {{
    GRAPH ex:ontology {{

        # Define class ex:Puppet
        ex:Puppet a owl:Class ;
                rdfs:label "Puppet" .

        # Define property ex:name
        ex:name a owl:DatatypeProperty ;
                rdfs:label "name" ;
                rdfs:domain ex:Puppet ;
                rdfs:range xsd:string .

        # Define property ex:show
        ex:show a owl:DatatypeProperty ;
                rdfs:label "show" ;
                rdfs:domain ex:Puppet ;
                rdfs:range xsd:string .
    }}
    }}
    """
    HanaTestUtils.execute_sparql_query(config.conn, query, '')

    graph = HanaRdfGraph(
        connection=config.conn,
        ontology_uri=ontology_uri,
    )

    yield graph

    query = f"""
    DROP GRAPH <{ontology_uri}>
    """

    HanaTestUtils.execute_sparql_query(config.conn, query, '')

@pytest.fixture
def default_graph_with_ontology_file():
    ontology_local_file_path = Path(__file__).parent / "fixtures" / "hana_rdf_graph_sample_schema.ttl"
    return HanaRdfGraph(
        connection=config.conn,
        ontology_local_file=ontology_local_file_path,
        ontology_local_file_format="turtle"
    )

@pytest.fixture
def expected_schema_graph():
    expected_schema_file_path = Path(__file__).parent / "fixtures" / "hana_rdf_graph_sample_schema.ttl"
    expected_schema_graph = rdflib.Graph()
    expected_schema_graph.parse(expected_schema_file_path, format="turtle")
    return expected_schema_graph

@pytest.fixture
def example_graph():
    graph_uri = "http://example.com/graph"
    query = f"""
    PREFIX ex: <http://example.com/>
    INSERT DATA {{
    GRAPH <{graph_uri}> {{
        <P1> a ex:Puppet; ex:name "Ernie"; ex:show "Sesame Street".
        <P2> a ex:Puppet; ex:name "Bert"; ex:show "Sesame Street" .
        }}
    }}
    """
    HanaTestUtils.execute_sparql_query(config.conn, query, '')

    graph = HanaRdfGraph(
        connection=config.conn,
        graph_uri=graph_uri,
        auto_extract_ontology=True,
    )
    yield graph

    query = f"""
    DROP GRAPH <{graph_uri}>
    """
    HanaTestUtils.execute_sparql_query(config.conn, query, '')


def test_hana_rdf_default_graph_creation(default_graph):
    """Test default graph creation with no graph uri given."""

    assert default_graph
    assert isinstance(default_graph, HanaRdfGraph)

def test_hana_rdf_default_graph_creation_with_graph_uri(default_graph_with_graph_uri):
    """Test default graph creation with default graph URI."""

    assert default_graph_with_graph_uri
    assert isinstance(default_graph_with_graph_uri, HanaRdfGraph)
    assert default_graph_with_graph_uri.from_clause == "FROM DEFAULT"

def test_hana_rdf_graph_creation_with_graph_uri(example_graph):
    """Test graph creation with graph URI."""

    assert example_graph
    assert isinstance(example_graph, HanaRdfGraph)
    assert example_graph.from_clause == "FROM <http://example.com/graph>"

def test_hana_rdf_graph_creation_with_ontology_uri(default_graph_with_ontology_uri, expected_schema_graph):
    """Test graph creation with ontology URI."""

    assert default_graph_with_ontology_uri
    assert isinstance(default_graph_with_ontology_uri, HanaRdfGraph)
    assert isomorphic(default_graph_with_ontology_uri.get_schema, expected_schema_graph)

def test_hana_graph_creation_with_ontology_file(default_graph_with_ontology_file, expected_schema_graph):
    """Test graph creation with ontology file."""

    assert default_graph_with_ontology_file
    assert isinstance(default_graph_with_ontology_file, HanaRdfGraph)
    assert isomorphic(default_graph_with_ontology_file.get_schema, expected_schema_graph)

def test_hana_rdf_graph_query(example_graph):
    """Test graph query."""

    query = """
    PREFIX ex: <http://example.com/>
    SELECT ?s ?name ?show
    FROM NAMED ex:graph
    WHERE {
        GRAPH ex:graph {
        ?s a ex:Puppet ;
            ex:name ?name ;
            ex:show ?show .
        } 
    }
    ORDER BY ?s
    """

    expected_csv = textwrap.dedent("""
        s,name,show
        P1,Ernie,Sesame Street
        P2,Bert,Sesame Street
    """
    )
    response = example_graph.query(query)
    response = response.replace('\r\n', '\n')
    assert response == expected_csv.lstrip()

"""Test HANA Rdf Graph functionality."""

import os
import pytest
from pathlib import Path
import textwrap
from hdbcli import dbapi
from langchain_hana.graphs import HanaRdfGraph
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

@pytest.fixture
def ontology_graph():
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


def test_hana_rdf_graph_creation(default_graph):
    """Test graph creation."""

    assert default_graph is not None
    assert isinstance(default_graph, HanaRdfGraph)

def test_hana_rdf_graph_creation_with_graph_uri(example_graph):
    """Test graph creation with graph URI."""

    assert example_graph is not None
    assert isinstance(example_graph, HanaRdfGraph)
    assert example_graph.graph_uri == "http://example.com/graph"

def test_hana_rdf_graph_creation_with_ontology_uri(ontology_graph):
    """Test graph creation with ontology URI."""

    expected_schema = textwrap.dedent("""
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

    <http://example.com/name> a owl:DatatypeProperty ;
        rdfs:label "name" ;
        rdfs:domain <http://example.com/Puppet> ;
        rdfs:range xsd:string .

    <http://example.com/show> a owl:DatatypeProperty ;
        rdfs:label "show" ;
        rdfs:domain <http://example.com/Puppet> ;
        rdfs:range xsd:string .

    <http://example.com/Puppet> a owl:Class ;
        rdfs:label "Puppet" .
    """)

    assert ontology_graph is not None
    assert isinstance(ontology_graph, HanaRdfGraph)
    assert ontology_graph.schema.strip() == expected_schema.strip()

def test_hana_graph_creation_with_ontology_file():
    """Test graph creation with ontology file."""

    ontology_local_file_path = Path(__file__).parent / "fixtures" / "hana_rdf_graph_sample_schema.ttl"

    graph = HanaRdfGraph(
        connection=config.conn,
        ontology_local_file=ontology_local_file_path,
        ontology_local_file_format="turtle"
    )

    assert graph is not None
    assert isinstance(graph, HanaRdfGraph)
    assert graph.schema is not None

def test_hana_rdf_graph_query(example_graph):
    """Test graph query."""

    query = """
    PREFIX ex: <http://example.com/>
    SELECT ?s ?name ?show
    FROM ex:graph
    WHERE {
        ?s a ex:Puppet ;
           ex:name ?name ;
           ex:show ?show .
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
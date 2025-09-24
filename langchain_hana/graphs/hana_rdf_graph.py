from __future__ import annotations

import os
import re
from typing import Optional

import rdflib
from hdbcli import dbapi
from rdflib.plugins.sparql import prepareQuery


class HanaRdfGraph:
    """
    SAP HANA CLOUD Knowledge Graph Engine Wrapper

    This class connects to a SAP HANA Graph SPARQL endpoint, executes queries,
    and loads or generates ontology/schema data via one of four methods:

    1. ontology_query: Provide a SPARQL CONSTRUCT query to extract the schema
       directly from the graph.
    2. ontology_uri: Specify a remote ontology graph URI; the schema is loaded
       via a default CONSTRUCT that copies all triples.
    3. ontology_local_file: Load the schema from a local RDF file (e.g., Turtle,
       RDF/XML) on disk.
    4. auto_extract_ontology: When enabled (and no other source given), run a
       built-in generic CONSTRUCT query that reverse-engineers classes, properties,
       domains, and ranges from your instance data.

    Args:
        connection (dbapi.Connection): A HANA database connection instance.
        ontology_query (Optional[str]): A SPARQL CONSTRUCT query to load the schema.
        ontology_uri (Optional[str]): The URI of the ontology graph
            containing the RDF schema.
        ontology_local_file (Optional[str]): Path to a local ontology file to load.
        ontology_local_file_format (Optional[str]): RDF format of the local file
            (e.g., 'turtle').
        graph_uri (Optional[str]): The URI of the target graph; uses 'DEFAULT' if None.
        auto_extract_ontology (bool): If True and no schema source provided,
            automatically extract a generic ontology via SPARQL.

    Example:
        from hdbcli import dbapi
        # Establish a database connection
        connection = dbapi.connect(
            address="<hostname>",
            port=30015,
            user="<username>",
            password="<password>"
        )
        rdf_graph = HanaRdfGraph(
            connection=connection,
            graph_uri="http://example.com/graph",
            ontology_uri="http://example.com/ontology"
        )
        # Execute a SPARQL query:
        sparql_query = "SELECT ?s ?p ?o WHERE { ?s ?p ?o . }"
        response = rdf_graph.query(sparql_query, inject_from_clause=True)
        print(response)

    *Security note*: Make sure that the database connection uses credentials
        that are narrowly-scoped to only include necessary permissions.
        Failure to do so may result in data corruption or loss, since the calling
        code may attempt commands that would result in deletion, mutation
        of data if appropriately prompted or reading sensitive data if such
        data is present in the database.
        The best way to guard against such negative outcomes is to (as appropriate)
        limit the permissions granted to the credentials used with this tool.

        See https://python.langchain.com/docs/security for more information.
    """

    def __init__(
        self,
        connection: dbapi.Connection,
        graph_uri: Optional[str] = "DEFAULT",  # use default graph if None was provided as graph_uri
        ontology_query: Optional[str] = None,
        ontology_uri: Optional[str] = None,
        ontology_local_file: Optional[str] = None,
        ontology_local_file_format: Optional[str] = None,
        auto_extract_ontology: bool = False,
    ) -> None:
        self.connection = connection
        self.graph_uri = graph_uri   #if graph_uri else "DEFAULT"

        self.refresh_schema(
            ontology_query,
            ontology_uri,
            ontology_local_file,
            ontology_local_file_format,
            auto_extract_ontology,
        )

    def inject_from_clause(self, query: str) -> str:
        """
        Injects a FROM clause into the SPARQL query if one is not already present..

        If self.graph_uri is provided, it inserts FROM <graph_uri>.
        If self.graph_uri is None, it inserts FROM DEFAULT.

        Args:
            query: The SPARQL query string.

        Returns:
            The modified SPARQL query with the appropriate FROM clause.

        Raises:
            ValueError: If the query does not contain a 'WHERE' clause.
        """
        # Determine the appropriate FROM clause.
        if self.graph_uri is None or self.graph_uri.upper() == "DEFAULT":
            from_clause = "FROM DEFAULT"
        else:
            from_clause = f"FROM <{self.graph_uri}>"

        # Check if a FROM clause is already present.
        from_pattern = re.compile(r"\bFROM\b", flags=re.IGNORECASE)
        if from_pattern.search(query):
            # FROM clause already exists, return query unchanged.
            return query

        # Use regex to match the first occurrence of 'WHERE' with word boundaries
        pattern = re.compile(r"\bWHERE\b", flags=re.IGNORECASE)
        match = pattern.search(query)
        if match:
            index = match.start()
            # Insert the FROM clause before the matched WHERE clause.
            query = query[:index] + f"\n{from_clause}\n" + query[index:]
        else:
            raise ValueError("The SPARQL query does not contain a 'WHERE' clause.")

        return query

    def query(
        self,
        query: str,
        content_type: Optional[str] = None,
        inject_from_clause: bool = True,  # If True , inject a FROM clause into query.
    ) -> str:
        """Executes SPARQL query and returns response as a string."""

        if content_type is None:
            content_type = "application/sparql-results+csv"

        request_headers = (
            f"Accept: {content_type}\r\nContent-Type: application/sparql-query"
        )

        if inject_from_clause and self.graph_uri:
            query = self.inject_from_clause(query)

        cursor = self.connection.cursor()
        try:
            result = cursor.callproc(
                "SYS.SPARQL_EXECUTE", (query, request_headers, "?", "?")
            )
            response = result[2]
        except dbapi.Error as db_error:
            raise RuntimeError(
                f'The database query "{query}" failed: '
                f'{db_error.errortext.split("; Server Connection")[0]}'
            )

        finally:
            cursor.close()

        return response

    def _load_ontology_schema_graph_from_query(self, ontology_query) -> rdflib.Graph:
        """
        Load an ontology schema by executing a SPARQL CONSTRUCT query.
        """
        self._validate_construct_query(ontology_query)

        response = self.query(ontology_query, content_type="", inject_from_clause=False)

        graph = rdflib.Graph()

        # Parse the string into the graph
        graph.parse(data=response, format="turtle")

        return graph

    @staticmethod
    def _load_ontology_schema_from_file(local_file: str, local_file_format: str = None):  # type: ignore[no-untyped-def, assignment]
        """
        Parse the ontology schema statements from the provided file
        """
        if not os.path.exists(local_file):
            raise FileNotFoundError(f"File {local_file} does not exist.")
        if not os.access(local_file, os.R_OK):
            raise PermissionError(f"Read permission for {local_file} is restricted")
        graph = rdflib.Graph()
        try:
            graph.parse(local_file, format=local_file_format)
        except Exception as e:
            raise ValueError(f"Invalid file format for {local_file} : ", e)
        return graph

    def refresh_schema(
        self,
        ontology_query: Optional[str] = None,
        ontology_uri: Optional[str] = None,
        ontology_local_file: Optional[str] = None,
        ontology_local_file_format: Optional[str] = None,
        auto_extract_ontology: bool = False,
    ) -> None:
        """
        Load or generate the graph's ontology schema.

        Args:
            ontology_query: SPARQL CONSTRUCT to load schema.
            ontology_uri: URI of schema graph to load.
            ontology_local_file: Local file path for schema.
            ontology_local_file_format: Format of local file.
            auto_extract_ontology: If True and no other source, use generic extractor.

        Raises:
            ValueError: If multiple or no schema sources are specified.
        """
        # Count provided schema sources
        schema_sources = sum(
            1
            for source in [ontology_query, ontology_uri, ontology_local_file]
            if source is not None
        )

        if schema_sources == 0 and auto_extract_ontology:
            ontology_query = HanaRdfGraph.get_generic_ontology_query(self.graph_uri)
            schema_sources = 1

        if schema_sources > 1:
            raise ValueError(
                "Multiple ontology/schema sources provided. Use only one of: "
                "ontology_query, ontology_uri, or ontology_local_file."
            )
        elif schema_sources == 0:
            raise ValueError(
                "No ontology/schema sources provided. Use only one of: "
                "ontology_query, ontology_uri, or ontology_local_file."
            )

        if ontology_local_file:
            ontology_schema_graph = self._load_ontology_schema_from_file(
                ontology_local_file,
                ontology_local_file_format,  # type: ignore[arg-type]
            )
        else:
            if ontology_uri:
                ontology_query = (
                    f"CONSTRUCT {{?s ?p ?o}} FROM <{ontology_uri}> WHERE"
                    + "{?s ?p ?o .}"
                )
            ontology_schema_graph = self._load_ontology_schema_graph_from_query(
                ontology_query
            )

        self.schema = ontology_schema_graph.serialize(format="turtle")

    @staticmethod
    def _validate_construct_query(construct_query: str) -> None:
        """
        Validate the query is a valid SPARQL CONSTRUCT query.

        Args:
            construct_query: The SPARQL query string to validate.

        Raises:
            TypeError: If query is not a string.
            ValueError: If query is not a valid CONSTRUCT query.
        """
        if not isinstance(construct_query, str):
            raise TypeError("Schema query must be provided as string.")

        parsed_query = prepareQuery(construct_query)
        if parsed_query.algebra.name != "ConstructQuery":
            raise ValueError(
                "Invalid query type. Only CONSTRUCT queries are supported for schema."
            )

    @staticmethod
    def get_generic_ontology_query(graph_uri):
        """
        Return a generic SPARQL CONSTRUCT that extracts
            a minimal OWL schema from the graph.

        Args:
            graph_uri: URI of the named graph to query.

        Returns:
            A SPARQL CONSTRUCT query string.
        """
        ontology_query = f"""
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
        CONSTRUCT {{ ?cls rdf:type owl:Class . ?cls rdfs:label ?clsLabel . ?rel rdf:type ?propertyType . ?rel rdfs:label ?relLabel . ?rel rdfs:domain ?domain . ?rel rdfs:range ?range .}}
        {f"FROM <{graph_uri}>" if graph_uri and graph_uri != "DEFAULT" else ""}
        WHERE {{ # get properties
            {{SELECT DISTINCT ?domain ?rel ?relLabel ?propertyType ?range
            WHERE {{
                ?subj ?rel ?obj .
                ?subj a ?domain .
                OPTIONAL{{?obj a ?rangeClass .}}
                FILTER(?rel != rdf:type) #I think we should remove type
                BIND(IF(isIRI(?obj) = true, owl:ObjectProperty, owl:DatatypeProperty) AS ?propertyType)
                BIND(COALESCE(?rangeClass, DATATYPE(?obj)) AS ?range)
                BIND(STR(?rel) AS ?uriStr)       # Convert URI to string
                BIND(REPLACE(?uriStr, "^.*[/#]", "") AS ?relLabel)
            }}}}
            UNION {{ # get classes
                SELECT DISTINCT ?cls ?clsLabel
                WHERE {{
                    ?instance a/rdfs:subClassOf* ?cls .
                    FILTER (isIRI(?cls)) .
                    BIND(STR(?cls) AS ?uriStr)       # Convert URI to string
                    BIND(REPLACE(?uriStr, "^.*[/#]", "") AS ?clsLabel)
                }}
            }}
        }}
        """
        return ontology_query

    @property
    def get_schema(self) -> str:
        """
        Return the schema of the graph in turtle format.
        """
        return self.schema

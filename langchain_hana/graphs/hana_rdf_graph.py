from __future__ import annotations

import csv
import io
from typing import Optional
import rdflib
from hdbcli import dbapi
import re


class HanaRdfGraph:
    """
    SAP HANA CLOUD Knowledge Graph Engine Wrapper

    This class connects to a SAP HANA Graph SPARQL endpoint, executes queries,
    and loads ontology schema data via a SELECT query.

    Args:
        connection (dbapi.Connection): A HANA database connection object obtained from hdbcli.dbapi.
        ontology_uri (str): The URI of the ontology containing the RDF schema.
        graph_uri (Optional[str]): The URI of the target graph. If None is provided, the default graph is used.

    Example:
        from hdbcli import dbapi
        # Establish a database connection (customize connection parameters as needed)
        connection = dbapi.connect(
            address="<hostname>",
            port=3<NN>MM,
            user="<username>",
            password="<password>"
        )
        ontology_uri = "http://example.com/ontology"
        graph_uri = "http://example.com/graph"  # Use None to select the default graph
        rdf_graph = HanaRdfGraph(
            connection=connection,
            ontology_uri=ontology_uri,
            graph_uri=graph_uri
        )
        # To execute a SPARQL query:
        sparql_query = "SELECT ?s ?p ?o WHERE { ?s ?p ?o . }"
        response = rdf_graph.query(sparql_query, add_from_clause=True)
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
        ontology_uri: str,
        graph_uri: Optional[str],  # use default graph if None was provided as graph_uri
    ) -> None:
        self.connection = connection
        self.ontology_uri = ontology_uri
        self.graph_uri = graph_uri

        ontology_schema_graph = self._load_ontology_schema_graph()
        self.schema = ontology_schema_graph.serialize(format="turtle")

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
        if self.graph_uri is None:
            from_clause = "FROM DEFAULT"
        else:
            from_clause = f"FROM <{self.graph_uri}>"

        # Check if a FROM clause is already present.
        from_pattern = re.compile(r'\bFROM\b', flags=re.IGNORECASE)
        if from_pattern.search(query):
            # FROM clause already exists, return query unchanged.
            return query

        # Use regex to match the first occurrence of 'WHERE' with word boundaries, case-insensitive.
        pattern = re.compile(r'\bWHERE\b', flags=re.IGNORECASE)
        match = pattern.search(query)
        if match:
            index = match.start()
            # Insert the FROM clause before the matched WHERE clause.
            query = query[:index] + f'\n{from_clause}\n' + query[index:]
        else:
            raise ValueError("The SPARQL query does not contain a 'WHERE' clause.")

        return query

    def query(
        self,
        query: str,
        content_type: Optional[str] = None,
        inject_from_clause: bool = True,  # If True , inject a FROM clause into the query.
    ) -> str:
        """Executes SPARQL query and returns response as a string."""

        self._validate_sparql_query(query)

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
                "SYS.SPARQL_EXECUTE", (query, request_headers, "?", None)
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

    def _load_ontology_schema_graph(self) -> rdflib.Graph:
        """
        Execute the query for collecting the ontology schema statements
        and store as rdblib.Graph
        """
        ontology_query = (
            f"SELECT   ?s ?o ?p FROM <{self.ontology_uri}> WHERE" + "{?s ?o ?p .}"
        )
        response = self.query(ontology_query, inject_from_clause=False)
        ontology_triples = self.convert_csv_response_to_list(response)

        graph = rdflib.Graph()

        for s_raw, p_raw, o_raw in ontology_triples:
            # Subject could be URI or bnode (blank node)
            if s_raw.startswith("http"):
                subject = rdflib.URIRef(s_raw)
            elif s_raw.startswith("_:"):
                subject = rdflib.BNode(s_raw)
            else:
                subject = None

            # Predicate (usually a URI)
            if p_raw.startswith("http"):
                predicate = rdflib.URIRef(p_raw)
            else:
                predicate = None

            # Object could be a URI, bnode, or literal
            if o_raw.startswith("http"):
                obj = rdflib.URIRef(o_raw)
            elif o_raw.startswith("_:"):
                obj = rdflib.BNode(o_raw)
            else:
                obj = rdflib.Literal(o_raw)

            # Validate RDF triple rules
            if subject is None:
                raise ValueError(f"Invalid RDF: Subject cannot be a literal ({s_raw})")
            if predicate is None:
                raise ValueError(f"Invalid RDF: Predicate must be a URI, not a blank node or literal ({p_raw})")

            graph.add((subject, predicate, obj))

        return graph

    @staticmethod
    def _validate_sparql_query(query: str) -> None:
        """Validate the generated SPARQL query structure."""
        required_keywords = ["SELECT", "WHERE"]
        if not all(keyword in query.upper() for keyword in required_keywords):
            raise ValueError(
                f"SPARQL query is invalid. "
                f"The query must contains the following keywords: {required_keywords}."
                f"Only SELECT queries are supported now!"
            )

    @staticmethod
    def convert_csv_response_to_list(csv_string: str, header: bool = False) -> list[list[str]]:
        """Convert CSV string response to a list of lists."""
        with io.StringIO(csv_string) as csv_file:
            reader = csv.reader(csv_file)
            if not header:
                next(reader, None)
            return [row for row in reader]

    @staticmethod
    def convert_csv_response_to_dataframe(result):  # type: ignore[no-untyped-def]
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "The 'pandas' library is required for this function. "
                "Please install it using 'pip install pandas'."
            )

        result_df = pd.read_csv(io.StringIO(result))
        return result_df.fillna("")

    def refresh_schema(self) -> None:
        """Reload and update the ontology schema."""
        ontology_schema_graph = self._load_ontology_schema_graph()
        self.schema = ontology_schema_graph.serialize(format="turtle")

    @property
    def get_schema(self) -> str:
        """
        Return the schema of the graph in turtle format.
        """
        return self.schema

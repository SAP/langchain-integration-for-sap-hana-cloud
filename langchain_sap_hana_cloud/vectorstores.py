"""SAP HANA Cloud Vector Engine"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import (
    Any,
    Callable,
    Iterable,
    Optional,
    Pattern,
    Type,
)

import numpy as np
from hdbcli import dbapi  # type: ignore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables.config import run_in_executor
from langchain_core.vectorstores import VectorStore
from langchain_core.vectorstores.utils import maximal_marginal_relevance

from langchain_sap_hana_cloud.embeddings import HanaInternalEmbeddings
from langchain_sap_hana_cloud.utils import DistanceStrategy

HANA_DISTANCE_FUNCTION: dict = {
    DistanceStrategy.COSINE: ("COSINE_SIMILARITY", "DESC"),
    DistanceStrategy.EUCLIDEAN_DISTANCE: ("L2DISTANCE", "ASC"),
}

COMPARISONS_TO_SQL = {
    "$eq": "=",
    "$ne": "<>",
    "$lt": "<",
    "$lte": "<=",
    "$gt": ">",
    "$gte": ">=",
}

IN_OPERATORS_TO_SQL = {
    "$in": "IN",
    "$nin": "NOT IN",
}

BETWEEN_OPERATOR = "$between"

LIKE_OPERATOR = "$like"

CONTAINS_OPERATOR = "$contains"

LOGICAL_OPERATORS_TO_SQL = {"$and": "AND", "$or": "OR"}

INTERMEDIATE_TABLE_NAME = "intermediate_result"

default_distance_strategy = DistanceStrategy.COSINE
default_table_name: str = "EMBEDDINGS"
default_content_column: str = "VEC_TEXT"
default_metadata_column: str = "VEC_META"
default_vector_column: str = "VEC_VECTOR"
default_vector_column_length: int = -1  # -1 means dynamic length


class HanaDB(VectorStore, ABC):
    """SAP HANA Cloud Vector Engine base class and factory.

    This class serves two roles:
    1. Factory: Creates and returns the appropriate implementation based on the
        embedding type:
       - Returns _HanaDBWithInternalEmbeddings for HanaInternalEmbeddings
       - Returns _HanaDBWithExternalEmbeddings for all other embedding types
    2. Base Class: Provides common functionality and interface for both implementations:
       - Table and index management
        - Query building and execution
        - Metadata handling
        - Connection management

    The factory pattern is implemented through __new__, which examines the
    embedding type and returns the appropriate concrete implementation.

    Key Features:
        - Automatic selection of appropriate implementation
        - Common interface for both internal and external embeddings
        - Shared functionality through inheritance
        - Type-safe implementation selection

    Example:
        >>> from langchain_sap_hana_cloud import HanaDB, HanaInternalEmbeddings
        >>> from langchain_openai import OpenAIEmbeddings
        >>>
        >>> # Using internal embeddings (returns _HanaDBWithInternalEmbeddings)
        >>> internal_embedding = HanaInternalEmbeddings(model_id="<model_id>")
        >>> db1 = HanaDB(
        ...     connection=your_connection,
        ...     embedding=internal_embedding
        ... )
        >>>
        >>> # Using external embeddings (returns _HanaDBWithExternalEmbeddings)
        >>> external_embedding = OpenAIEmbeddings()
        >>> db2 = HanaDB(
        ...     connection=your_connection,
        ...     embedding=external_embedding
        ... )
    """

    def __new__(
        cls,
        connection: dbapi.Connection,
        embedding: Embeddings,
        distance_strategy: DistanceStrategy = default_distance_strategy,
        table_name: str = default_table_name,
        content_column: str = default_content_column,
        metadata_column: str = default_metadata_column,
        vector_column: str = default_vector_column,
        vector_column_length: int = default_vector_column_length,
        *,
        specific_metadata_columns: Optional[list[str]] = None,
    ):
        if cls is HanaDB:
            # Create an instance based on the embedding type
            if isinstance(embedding, HanaInternalEmbeddings):
                subclass = _HanaDBWithInternalEmbeddings
            else:
                subclass = _HanaDBWithExternalEmbeddings
            return super().__new__(subclass)

        else:
            return super().__new__(cls)

    """Base class for SAP HANA Cloud Vector Engine implementations.

    The HanaDB vectorstore can be created by providing an embedding function and
    an existing database connection. Optionally, the names of the table and the
    columns to use.
    """

    def __init__(
        self,
        connection: dbapi.Connection,
        embedding: Embeddings,
        distance_strategy: DistanceStrategy = default_distance_strategy,
        table_name: str = default_table_name,
        content_column: str = default_content_column,
        metadata_column: str = default_metadata_column,
        vector_column: str = default_vector_column,
        vector_column_length: int = default_vector_column_length,
        *,
        specific_metadata_columns: Optional[list[str]] = None,
    ):
        valid_distance = False
        for key in HANA_DISTANCE_FUNCTION.keys():
            if key is distance_strategy:
                valid_distance = True
        if not valid_distance:
            raise ValueError(
                "Unsupported distance_strategy: {}".format(distance_strategy)
            )

        self.connection = connection
        self.embedding = embedding
        self.distance_strategy = distance_strategy
        self.table_name = HanaDB._sanitize_name(table_name)
        self.content_column = HanaDB._sanitize_name(content_column)
        self.metadata_column = HanaDB._sanitize_name(metadata_column)
        self.vector_column = HanaDB._sanitize_name(vector_column)
        self.vector_column_length = HanaDB._sanitize_int(vector_column_length)
        self.specific_metadata_columns = HanaDB._sanitize_specific_metadata_columns(
            specific_metadata_columns or []
        )

        # Initialize the table if it doesn't exist
        self._initialize_table()

    def _initialize_table(self) -> None:
        """Initialize the table if it doesn't exist and validate columns."""
        if not self._table_exists(self.table_name):
            sql_str = (
                f'CREATE TABLE "{self.table_name}"('
                f'"{self.content_column}" NCLOB, '
                f'"{self.metadata_column}" NCLOB, '
                f'"{self.vector_column}" REAL_VECTOR '
            )
            if self.vector_column_length in [-1, 0]:
                sql_str += ");"
            else:
                sql_str += f"({self.vector_column_length}));"

            try:
                cur = self.connection.cursor()
                cur.execute(sql_str)
            finally:
                cur.close()

        # Check if the needed columns exist and have the correct type
        self._check_column(self.table_name, self.content_column, ["NCLOB", "NVARCHAR"])
        self._check_column(self.table_name, self.metadata_column, ["NCLOB", "NVARCHAR"])
        self._check_column(
            self.table_name,
            self.vector_column,
            ["REAL_VECTOR"],
            self.vector_column_length,
        )
        for column_name in self.specific_metadata_columns:
            self._check_column(self.table_name, column_name)

    def _table_exists(self, table_name) -> bool:  # type: ignore[no-untyped-def]
        sql_str = (
            "SELECT COUNT(*) FROM SYS.TABLES WHERE SCHEMA_NAME = CURRENT_SCHEMA"
            " AND TABLE_NAME = ?"
        )
        try:
            cur = self.connection.cursor()
            cur.execute(sql_str, (table_name))
            if cur.has_result_set():
                rows = cur.fetchall()
                if rows[0][0] == 1:
                    return True
        finally:
            cur.close()
        return False

    def _check_column(  # type: ignore[no-untyped-def]
        self, table_name, column_name, column_type=None, column_length=None
    ):
        sql_str = (
            "SELECT DATA_TYPE_NAME, LENGTH FROM SYS.TABLE_COLUMNS WHERE "
            "SCHEMA_NAME = CURRENT_SCHEMA "
            "AND TABLE_NAME = ? AND COLUMN_NAME = ?"
        )
        try:
            cur = self.connection.cursor()
            cur.execute(sql_str, (table_name, column_name))
            if cur.has_result_set():
                rows = cur.fetchall()
                if len(rows) == 0:
                    raise AttributeError(f"Column {column_name} does not exist")
                # Check data type
                if column_type:
                    if rows[0][0] not in column_type:
                        raise AttributeError(
                            f"Column {column_name} has the wrong type: {rows[0][0]}"
                        )
                # Check length, if parameter was provided
                # Length can either be -1 (QRC01+02-24) or 0 (QRC03-24 onwards)
                # to indicate no length constraint being present.
                if column_length is not None and column_length > 0:
                    if rows[0][1] != column_length:
                        raise AttributeError(
                            f"Column {column_name} has the wrong length: {rows[0][1]} "
                            f"expected: {column_length}"
                        )
            else:
                raise AttributeError(f"Column {column_name} does not exist")
        finally:
            cur.close()

    @property
    def embeddings(self) -> Embeddings:
        return self.embedding

    @staticmethod
    def _sanitize_name(input_str: str) -> str:  # type: ignore[misc]
        # Remove characters that are not alphanumeric or underscores
        return re.sub(r"[^a-zA-Z0-9_]", "", input_str)

    @staticmethod
    def _sanitize_int(input_int: any) -> int:  # type: ignore[valid-type]
        value = int(str(input_int))
        if value < -1:
            raise ValueError(f"Value ({value}) must not be smaller than -1")
        return int(str(input_int))

    @staticmethod
    def _sanitize_list_float(embedding: list[float]) -> list[float]:
        for value in embedding:
            if not isinstance(value, float):
                raise ValueError(f"Value ({value}) does not have type float")
        return embedding

    # Compile pattern only once, for better performance
    _compiled_pattern: Pattern = re.compile("^[_a-zA-Z][_a-zA-Z0-9]*$")

    @staticmethod
    def _sanitize_metadata_keys(metadata: dict) -> dict:
        for key in metadata.keys():
            if not HanaDB._compiled_pattern.match(key):
                raise ValueError(f"Invalid metadata key {key}")

        return metadata

    @staticmethod
    def _sanitize_specific_metadata_columns(
        specific_metadata_columns: list[str],
    ) -> list[str]:
        metadata_columns = []
        for c in specific_metadata_columns:
            sanitized_name = HanaDB._sanitize_name(c)
            metadata_columns.append(sanitized_name)
        return metadata_columns

    def _split_off_special_metadata(self, metadata: dict) -> tuple[dict, list]:
        # Use provided values by default or fallback
        special_metadata = []

        if not metadata:
            return {}, []

        for column_name in self.specific_metadata_columns:
            special_metadata.append(metadata.get(column_name, None))

        return metadata, special_metadata

    def _get_specific_metadata_columns_string(self) -> str:
        """
        Helper function to generate the specific metadata columns as a SQL string.
        Returns:
            str: SQL string for specific metadata columns.
        """
        if not self.specific_metadata_columns:
            return ""
        return ', "' + '", "'.join(self.specific_metadata_columns) + '"'

    def create_hnsw_index(
        self,
        m: Optional[int] = None,  # Optional M parameter
        ef_construction: Optional[int] = None,  # Optional efConstruction parameter
        ef_search: Optional[int] = None,  # Optional efSearch parameter
        index_name: Optional[str] = None,  # Optional custom index name
    ) -> None:
        """
        Creates an HNSW vector index on a specified table and vector column with
        optional build and search configurations. If no configurations are provided,
        default parameters from the database are used. If provided values exceed the
        valid ranges, an error will be raised.
        The index is always created in ONLINE mode.

        Args:
            m: (Optional) Maximum number of neighbors per graph node
                (Valid Range: [4, 1000])
            ef_construction: (Optional) Maximal candidates to consider when building
                                the graph (Valid Range: [1, 100000])
            ef_search: (Optional) Minimum candidates for top-k-nearest neighbor
                                queries (Valid Range: [1, 100000])
            index_name: (Optional) Custom index name. Defaults to
                        <table_name>_<distance_strategy>_idx
        """
        # Set default index name if not provided
        distance_func_name = HANA_DISTANCE_FUNCTION[self.distance_strategy][0]
        default_index_name = f"{self.table_name}_{distance_func_name}_idx"
        # Use provided index_name or default
        index_name = (
            HanaDB._sanitize_name(index_name) if index_name else default_index_name
        )
        # Initialize build_config and search_config as empty dictionaries
        build_config = {}
        search_config = {}

        # Validate and add m parameter to build_config if provided
        if m is not None:
            m = HanaDB._sanitize_int(m)
            if not (4 <= m <= 1000):
                raise ValueError("M must be in the range [4, 1000]")
            build_config["M"] = m

        # Validate and add ef_construction to build_config if provided
        if ef_construction is not None:
            ef_construction = HanaDB._sanitize_int(ef_construction)
            if not (1 <= ef_construction <= 100000):
                raise ValueError("efConstruction must be in the range [1, 100000]")
            build_config["efConstruction"] = ef_construction

        # Validate and add ef_search to search_config if provided
        if ef_search is not None:
            ef_search = HanaDB._sanitize_int(ef_search)
            if not (1 <= ef_search <= 100000):
                raise ValueError("efSearch must be in the range [1, 100000]")
            search_config["efSearch"] = ef_search

        # Convert build_config and search_config to JSON strings if they contain values
        build_config_str = json.dumps(build_config) if build_config else ""
        search_config_str = json.dumps(search_config) if search_config else ""

        # Create the index SQL string with the ONLINE keyword
        sql_str = (
            f'CREATE HNSW VECTOR INDEX {index_name} ON "{self.table_name}" '
            f'("{self.vector_column}") '
            f"SIMILARITY FUNCTION {distance_func_name} "
        )

        # Append build_config to the SQL string if provided
        if build_config_str:
            sql_str += f"BUILD CONFIGURATION '{build_config_str}' "

        # Append search_config to the SQL string if provided
        if search_config_str:
            sql_str += f"SEARCH CONFIGURATION '{search_config_str}' "

        # Always add the ONLINE option
        sql_str += "ONLINE "
        cur = self.connection.cursor()
        try:
            cur.execute(sql_str)
        finally:
            cur.close()

    def similarity_search(  # type: ignore[override]
        self, query: str, k: int = 4, filter: Optional[dict] = None
    ) -> list[Document]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: A dictionary of metadata fields and values to filter by.
                    Defaults to None.

        Returns:
            List of Documents most similar to the query
        """
        docs_and_scores = self.similarity_search_with_score(
            query=query, k=k, filter=filter
        )
        return [doc for doc, _ in docs_and_scores]

    def _extract_keyword_search_columns(
        self, filter: Optional[dict] = None
    ) -> list[str]:
        """
        Extract metadata columns used with `$contains` in the filter.
        Scans the filter to find unspecific metadata columns used
        with the `$contains` operator.
        Args:
            filter: A dictionary of filter criteria.
        Returns:
            list of metadata column names for keyword searches.
        Example:
            filter = {"$or": [
                {"title": {"$contains": "barbie"}},
                {"VEC_TEXT": {"$contains": "fred"}}]}
            Result: ["title"]
        """
        keyword_columns = set()

        def recurse_filters(
            f: Optional[dict[Any, Any]], parent_key: Optional[str] = None
        ) -> None:
            if isinstance(f, dict):
                for key, value in f.items():
                    if key == CONTAINS_OPERATOR:
                        # Add the parent key as it's the metadata column being filtered
                        if parent_key and not (
                            parent_key == self.content_column
                            or parent_key in self.specific_metadata_columns
                        ):
                            keyword_columns.add(parent_key)
                    elif (
                        key in LOGICAL_OPERATORS_TO_SQL.keys()
                    ):  # Handle logical operators
                        for subfilter in value:
                            recurse_filters(subfilter)
                    else:
                        recurse_filters(value, parent_key=key)

        recurse_filters(filter)
        return list(keyword_columns)

    def _create_metadata_projection(self, projected_metadata_columns: list[str]) -> str:
        """
        Generate a SQL `WITH` clause to project metadata columns for keyword search.
        Args:
            projected_metadata_columns: list of metadata column names for projection.
        Returns:
            A SQL `WITH` clause string.
        Example:
            Input: ["title", "author"]
            Output:
            WITH intermediate_result AS (
                SELECT *,
                JSON_VALUE(metadata_column, '$.title') AS "title",
                JSON_VALUE(metadata_column, '$.author') AS "author"
                FROM "table_name"
            )
        """

        metadata_columns = [
            f"JSON_VALUE({self.metadata_column}, '$.{col}') AS \"{col}\""
            for col in projected_metadata_columns
        ]
        return (
            f"WITH {INTERMEDIATE_TABLE_NAME} AS ("
            f"SELECT *, {', '.join(metadata_columns)} "
            f"FROM \"{self.table_name}\")"
        )

    def _create_where_by_filter(self, filter):  # type: ignore[no-untyped-def]
        query_tuple = []
        where_str = ""
        if filter:
            where_str, query_tuple = self._process_filter_object(filter)
            where_str = " WHERE " + where_str
        return where_str, query_tuple

    def _process_filter_object(self, filter):  # type: ignore[no-untyped-def]
        query_tuple = []
        where_str = ""
        if filter:
            for i, key in enumerate(filter.keys()):
                filter_value = filter[key]
                if i != 0:
                    where_str += " AND "

                # Handling of 'special' boolean operators "$and", "$or"
                if key in LOGICAL_OPERATORS_TO_SQL:
                    logical_operator = LOGICAL_OPERATORS_TO_SQL[key]
                    logical_operands = filter_value
                    for j, logical_operand in enumerate(logical_operands):
                        if j != 0:
                            where_str += f" {logical_operator} "
                        (
                            where_str_logical,
                            query_tuple_logical,
                        ) = self._process_filter_object(logical_operand)
                        where_str += "(" + where_str_logical + ")"
                        query_tuple += query_tuple_logical
                    continue

                operator = "="
                sql_param = "?"

                if isinstance(filter_value, bool):
                    query_tuple.append("true" if filter_value else "false")
                elif isinstance(filter_value, int) or isinstance(filter_value, str):
                    query_tuple.append(filter_value)
                elif isinstance(filter_value, dict):
                    # Handling of 'special' operators starting with "$"
                    special_op = next(iter(filter_value))
                    special_val = filter_value[special_op]
                    # "$eq", "$ne", "$lt", "$lte", "$gt", "$gte"
                    if special_op in COMPARISONS_TO_SQL:
                        operator = COMPARISONS_TO_SQL[special_op]
                        if isinstance(special_val, bool):
                            query_tuple.append("true" if special_val else "false")
                        elif isinstance(special_val, float):
                            sql_param = "CAST(? as float)"
                            query_tuple.append(special_val)
                        elif (
                            isinstance(special_val, dict)
                            and "type" in special_val
                            and special_val["type"] == "date"
                        ):
                            # Date type
                            sql_param = "CAST(? as DATE)"
                            query_tuple.append(special_val["date"])
                        else:
                            query_tuple.append(special_val)
                    # "$between"
                    elif special_op == BETWEEN_OPERATOR:
                        between_from = special_val[0]
                        between_to = special_val[1]
                        operator = "BETWEEN"
                        sql_param = "? AND ?"
                        query_tuple.append(between_from)
                        query_tuple.append(between_to)
                    # "$like"
                    elif special_op == LIKE_OPERATOR:
                        operator = "LIKE"
                        query_tuple.append(special_val)
                    # "$contains"
                    elif special_op == CONTAINS_OPERATOR:
                        operator = CONTAINS_OPERATOR
                        query_tuple.append(special_val)
                    # "$in", "$nin"
                    elif special_op in IN_OPERATORS_TO_SQL:
                        operator = IN_OPERATORS_TO_SQL[special_op]
                        if isinstance(special_val, list):
                            for i, list_entry in enumerate(special_val):
                                if i == 0:
                                    sql_param = "("
                                sql_param = sql_param + "?"
                                if i == (len(special_val) - 1):
                                    sql_param = sql_param + ")"
                                else:
                                    sql_param = sql_param + ","
                                query_tuple.append(list_entry)
                        else:
                            raise ValueError(
                                f"Unsupported value for {operator}: {special_val}"
                            )
                    else:
                        raise ValueError(f"Unsupported operator: {special_op}")
                else:
                    raise ValueError(
                        f"Unsupported filter data-type: {type(filter_value)}"
                    )

                if operator == CONTAINS_OPERATOR:
                    where_str += f"SCORE(? IN (\"{key}\" EXACT SEARCH MODE 'text')) > 0"
                else:
                    selector = (
                        f' "{key}"'
                        if key in self.specific_metadata_columns
                        else f"JSON_VALUE({self.metadata_column}, '$.{key}')"
                    )
                    where_str += f"{selector} " f"{operator} {sql_param}"

        return where_str, query_tuple

    def delete(  # type: ignore[override]
        self, ids: Optional[list[str]] = None, filter: Optional[dict] = None
    ) -> Optional[bool]:
        """Delete entries by filter with metadata values

        Args:
            ids: Deletion with ids is not supported! A ValueError will be raised.
            filter: A dictionary of metadata fields and values to filter by.
                    An empty filter ({}) will delete all entries in the table.

        Returns:
            Optional[bool]: True, if deletion is technically successful.
            Deletion of zero entries, due to non-matching filters is a success.
        """

        if ids is not None:
            raise ValueError("Deletion via ids is not supported")

        if filter is None:
            raise ValueError("Parameter 'filter' is required when calling 'delete'")

        where_str, query_tuple = self._create_where_by_filter(filter)
        sql_str = f'DELETE FROM "{self.table_name}" {where_str}'

        try:
            cur = self.connection.cursor()
            cur.execute(sql_str, query_tuple)
        finally:
            cur.close()

        return True

    async def adelete(  # type: ignore[override]
        self, ids: Optional[list[str]] = None, filter: Optional[dict] = None
    ) -> Optional[bool]:
        """Delete by vector ID or other criteria.

        Args:
            ids: List of ids to delete.

        Returns:
            Optional[bool]: True if deletion is successful,
            False otherwise, None if not implemented.
        """
        return await run_in_executor(None, self.delete, ids=ids, filter=filter)

    async def amax_marginal_relevance_search_by_vector(  # type: ignore[override]
        self,
        embedding: list[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs selected using the maximal marginal relevance."""
        return await run_in_executor(
            None,
            self.max_marginal_relevance_search_by_vector,
            embedding=embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
        )

    @staticmethod
    def _parse_float_array_from_string(array_as_string: str) -> list[float]:  # type: ignore[misc]
        array_wo_brackets = array_as_string[1:-1]
        return [float(x) for x in array_wo_brackets.split(",")]

    @staticmethod
    def _cosine_relevance_score_fn(distance: float) -> float:
        return distance

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """
        The 'correct' relevance function
        may differ depending on a few things, including:
        - the distance / similarity metric used by the VectorStore
        - the scale of your embeddings (OpenAI's are unit normed. Many others are not!)
        - embedding dimensionality
        - etc.

        Vectorstores should define their own selection based method of relevance.
        """
        if self.distance_strategy == DistanceStrategy.COSINE:
            return HanaDB._cosine_relevance_score_fn
        elif self.distance_strategy == DistanceStrategy.EUCLIDEAN_DISTANCE:
            return HanaDB._euclidean_relevance_score_fn
        else:
            raise ValueError(
                "Unsupported distance_strategy: {}".format(self.distance_strategy)
            )

    def _similarity_search_with_score_and_vector(
        self,
        embedding_expr: str,
        k: int = 4,
        filter: Optional[dict] = None,
        vector_embedding_params: list[str, str] = None,
    ) -> list[tuple[Document, float, list[float]]]:
        """Perform similarity search and return documents with scores and vectors.

        Args:
            embedding_expr: SQL expression that generates or represents the embedding
                vector. For internal embeddings, this would be
                "VECTOR_EMBEDDING(?, 'QUERY', ?)",
                where ? will be replaced by parameters from query_params.
                For external embeddings, this would be "TO_REAL_VECTOR('...')".
            k: Number of documents to return. Defaults to 4.
            filter: Optional dictionary of metadata fields and values to filter by.
            query_params: Optional parameters for the embedding_expr SQL expression.
                For VECTOR_EMBEDDING function: [text, model_id] for the placeholders.
                For TO_REAL_VECTOR: None as the vector is included in the expression.

        Returns:
            List of tuples, each containing:
            - Document: The matched document with its content and metadata
            - float: The similarity score
            - list[float]: The document's embedding vector
        """
        result = []
        k = HanaDB._sanitize_int(k)
        distance_func_name = HANA_DISTANCE_FUNCTION[self.distance_strategy][0]

        # Generate metadata projection for filtered results
        projected_metadata_columns = self._extract_keyword_search_columns(filter)
        metadata_projection = ""
        if projected_metadata_columns:
            metadata_projection = self._create_metadata_projection(
                projected_metadata_columns
            )

        from_clause = (
            INTERMEDIATE_TABLE_NAME if metadata_projection else f'"{self.table_name}"'
        )

        order_str = f" order by CS {HANA_DISTANCE_FUNCTION[self.distance_strategy][1]}"
        where_str, query_tuple = self._create_where_by_filter(filter)

        if vector_embedding_params:
            query_tuple = vector_embedding_params + list(query_tuple)

        sql_str = (
            f"{metadata_projection} "
            f"SELECT TOP {k}"
            f'  "{self.content_column}", '  # row[0]
            f'  "{self.metadata_column}", '  # row[1]
            f'  TO_NVARCHAR("{self.vector_column}"), '  # row[2]
            f'  {distance_func_name}("{self.vector_column}", '
            f"  {embedding_expr}) AS CS "  # row[3]
            f"FROM {from_clause}"
        )
        sql_str = sql_str + where_str
        sql_str = sql_str + order_str
        try:
            cur = self.connection.cursor()
            cur.execute(sql_str, query_tuple)
            if cur.has_result_set():
                rows = cur.fetchall()
                for row in rows:
                    js = json.loads(row[1])
                    doc = Document(page_content=row[0], metadata=js)
                    result_vector = HanaDB._parse_float_array_from_string(row[2])
                    result.append((doc, row[3], result_vector))
        finally:
            cur.close()
        return result

    @classmethod
    def from_texts(  # type: ignore[no-untyped-def, override]
        cls: Type[HanaDB],
        texts: list[str],
        embedding: Embeddings,
        metadatas: Optional[list[dict]] = None,
        connection: dbapi.Connection = None,
        distance_strategy: DistanceStrategy = default_distance_strategy,
        table_name: str = default_table_name,
        content_column: str = default_content_column,
        metadata_column: str = default_metadata_column,
        vector_column: str = default_vector_column,
        vector_column_length: int = default_vector_column_length,
        *,
        specific_metadata_columns: Optional[list[str]] = None,
    ):
        """Create a HanaDB instance from raw documents.
        This is a user-friendly interface that:
            1. Embeds documents.
            2. Creates a table if it does not yet exist.
            3. Adds the documents to the table.
        This is intended to be a quick way to get started.
        """

        instance = cls(
            connection=connection,
            embedding=embedding,
            distance_strategy=distance_strategy,
            table_name=table_name,
            content_column=content_column,
            metadata_column=metadata_column,
            vector_column=vector_column,
            vector_column_length=vector_column_length,  # -1 means dynamic length
            specific_metadata_columns=specific_metadata_columns,
        )
        instance.add_texts(texts, metadatas)
        return instance

    @abstractmethod
    def add_texts(  # type: ignore[override]
        self,
        texts: Iterable[str],
        metadatas: Optional[list[dict]] = None,
        embeddings: Optional[list[list[float]]] = None,
        **kwargs: Any,
    ) -> list[str]:
        """Add texts to the vectorstore.

        Each implementation should handle the specific way texts are added
        based on whether internal or external embeddings are used.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            embeddings: Optional list of pre-computed embeddings (not used
                with internal embeddings).

        Returns:
            list[str]: empty list
        """

    @abstractmethod
    def similarity_search_with_score(  # type: ignore[override]
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
    ) -> list[tuple[Document, float]]:
        """Return documents and score values most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: A dictionary of metadata fields and values to filter by.
                    Defaults to None.

        Returns:
            List of tuples (containing a Document and a score) that are
            most similar to the query
        """

    @abstractmethod
    def similarity_search_by_vector(  # type: ignore[override]
        self, embedding: list[float], k: int = 4, filter: Optional[dict] = None
    ) -> list[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: A dictionary of metadata fields and values to filter by.
                    Defaults to None.

        Returns:
            List of Documents most similar to the query vector.
        """

    @abstractmethod
    def max_marginal_relevance_search(  # type: ignore[override]
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[dict] = None,
    ) -> list[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            query: search query text.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
            filter: Filter on metadata properties, e.g.
                            {
                                "str_property": "foo",
                                "int_property": 123
                            }
        Returns:
            List of Documents selected by maximal marginal relevance.
        """

    @abstractmethod
    def max_marginal_relevance_search_by_vector(  # type: ignore[override]
        self,
        embedding: list[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[dict] = None,
    ) -> list[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Returns:
            List of Documents selected by maximal marginal relevance.
        """


class _HanaDBWithExternalEmbeddings(HanaDB):
    """SAP HANA Cloud Vector Engine implementation using external embeddings.

    This implementation enables the use of any LangChain-compatible embedding model
    (except HanaInternalEmbeddings) with SAP HANA's vector store capabilities.
    It handles the generation of embeddings externally and manages their storage and
    retrieval within the database.

    Implementation Details:
        - Embeddings are generated externally using the provided embedding model
        - Vectors are stored in SAP HANA using TO_REAL_VECTOR for efficient storage
        - Similarity search uses SAP HANA's vector operations for fast retrieval
        - Supports both synchronous and asynchronous operations

    Note:
        This class is not meant to be instantiated directly. Instead, use HanaDB with
        any embedding model except HanaInternalEmbeddings:
        >>> from langchain_sap_hana_cloud import HanaDB
        >>> from langchain_openai import OpenAIEmbeddings
        >>> embedding = OpenAIEmbeddings()
        >>> # Returns this implementation
        >>> db = HanaDB(connection=conn, embedding=embedding)
    """

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[list[dict]] = None,
        embeddings: Optional[list[list[float]]] = None,
        **kwargs: Any,
    ) -> list[str]:
        """Add more texts using external embeddings.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            embeddings: Optional list of pre-computed embeddings.

        Returns:
            List[str]: empty list
        """
        # Create all embeddings of the texts beforehand to improve performance
        if embeddings is None:
            embeddings = self.embedding.embed_documents(list(texts))

        # Create sql parameters array
        sql_params = []
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas else {}
            metadata, extracted_special_metadata = self._split_off_special_metadata(
                metadata
            )
            sql_params.append(
                (
                    text,
                    json.dumps(HanaDB._sanitize_metadata_keys(metadata)),
                    str(embeddings[i]),
                    *extracted_special_metadata,
                )
            )

        specific_metadata_columns_string = self._get_specific_metadata_columns_string()
        sql_str = (
            f'INSERT INTO "{self.table_name}" ("{self.content_column}", '
            f'"{self.metadata_column}", '
            f'"{self.vector_column}"{specific_metadata_columns_string}) '
            f"VALUES (?, ?, TO_REAL_VECTOR (?)"
            f"{', ?' * len(self.specific_metadata_columns)});"
        )

        # Insert data into the table
        cur = self.connection.cursor()
        try:
            cur.executemany(sql_str, sql_params)
        finally:
            cur.close()
        return []

    def similarity_search_with_score(  # type: ignore[override]
        self, query: str, k: int = 4, filter: Optional[dict] = None
    ) -> list[tuple[Document, float]]:
        """Return documents and score values most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: A dictionary of metadata fields and values to filter by.
                    Defaults to None.

        Returns:
            List of tuples (containing a Document and a score) that are
            most similar to the query
        """
        embedding = self.embedding.embed_query(query)
        return self.similarity_search_with_score_by_vector(
            embedding=embedding, k=k, filter=filter
        )

    def similarity_search_with_score_by_vector(
        self, embedding: list[float], k: int = 4, filter: Optional[dict] = None
    ) -> list[tuple[Document, float]]:
        """Return docs most similar to the given embedding.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: A dictionary of metadata fields and values to filter by.
                    Defaults to None.

        Returns:
            List of Documents most similar to the query and score for each
        """
        whole_result = self.similarity_search_with_score_and_vector_by_vector(
            embedding=embedding, k=k, filter=filter
        )
        return [(result_item[0], result_item[1]) for result_item in whole_result]

    def similarity_search_with_score_and_vector_by_vector(
        self, embedding: list[float], k: int = 4, filter: Optional[dict] = None
    ) -> list[tuple[Document, float, list[float]]]:
        """Return docs most similar to the given embedding.

        Args:
            embedding: Pre-computed embedding vector to search with.
            k: Number of Documents to return. Defaults to 4.
            filter: A dictionary of metadata fields and values to filter by.
                    Defaults to None.

        Returns:
            List of tuples, each containing:
            - Document: The matched document with its content and metadata
            - float: The similarity score
            - list[float]: The document's embedding vector
        """
        embedding_expr = f"TO_REAL_VECTOR ('{str(embedding)}')"

        return self._similarity_search_with_score_and_vector(
            embedding_expr, k=k, filter=filter
        )

    def similarity_search_by_vector(  # type: ignore[override]
        self, embedding: list[float], k: int = 4, filter: Optional[dict] = None
    ) -> list[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: A dictionary of metadata fields and values to filter by.
                    Defaults to None.

        Returns:
            List of Documents most similar to the query vector.
        """
        docs_and_scores = self.similarity_search_with_score_by_vector(
            embedding=embedding, k=k, filter=filter
        )
        return [doc for doc, _ in docs_and_scores]

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[dict] = None,
    ) -> list[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            query: search query text.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
            filter: Filter on metadata properties, e.g.
                            {
                                "str_property": "foo",
                                "int_property": 123
                            }
        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        embedding = self.embedding.embed_query(query)
        return self.max_marginal_relevance_search_by_vector(
            embedding=embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
        )

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[dict] = None,
    ) -> list[Document]:
        whole_result = self.similarity_search_with_score_and_vector_by_vector(
            embedding=embedding, k=fetch_k, filter=filter
        )
        embeddings = [result_item[2] for result_item in whole_result]
        mmr_doc_indexes = maximal_marginal_relevance(
            np.array(embedding), embeddings, lambda_mult=lambda_mult, k=k
        )

        return [whole_result[i][0] for i in mmr_doc_indexes]


class _HanaDBWithInternalEmbeddings(HanaDB):
    """SAP HANA Cloud Vector Engine implementation using internal embeddings.

    This implementation leverages SAP HANA's built-in VECTOR_EMBEDDING function for
    generating embeddings directly within the database. This approach provides several
    key advantages:
        - Superior Performance: Embeddings are generated inside the database,
          eliminating data transfer overhead
        - Reduced Network Traffic: No need to send large text data to external services
        - Lower Latency: Direct integration with the database engine

    Key Features:
        - In-database embedding generation using VECTOR_EMBEDDING function
        - Optimized for SAP HANA's internal embedding models
        - Automatic validation of embedding function availability
        - Support for document and query embedding types

    Requirements:
        - SAP HANA instance with Vector Engine capability
        - Appropriate permissions for VECTOR_EMBEDDING function
        - Valid model ID for the internal embedding model

    Note:
        This class is not meant to be instantiated directlyInstead, use HanaDB with
        a HanaInternalEmbeddings instance, and the factory will create this
        implementation:
        >>> from langchain_sap_hana_cloud import HanaDB, HanaInternalEmbeddings
        >>> embedding = HanaInternalEmbeddings(model_id="<model_id>")
        >>> # Returns this implementation
        >>> db = HanaDB(connection=conn, embedding=embedding)
    """

    def __init__(
        self,
        connection: dbapi.Connection,
        embedding: HanaInternalEmbeddings,
        distance_strategy: DistanceStrategy = default_distance_strategy,
        table_name: str = default_table_name,
        content_column: str = default_content_column,
        metadata_column: str = default_metadata_column,
        vector_column: str = default_vector_column,
        vector_column_length: int = default_vector_column_length,
        *,
        specific_metadata_columns: Optional[list[str]] = None,
    ):
        super().__init__(
            connection=connection,
            embedding=embedding,
            distance_strategy=distance_strategy,
            table_name=table_name,
            content_column=content_column,
            metadata_column=metadata_column,
            vector_column=vector_column,
            vector_column_length=vector_column_length,
            specific_metadata_columns=specific_metadata_columns,
        )
        self.internal_embedding_model_id = embedding.get_model_id()
        self._validate_internal_embedding_function()

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[list[dict]] = None,
        embeddings: Optional[list[list[float]]] = None,
        **kwargs: Any,
    ) -> list[str]:
        """Add more texts using internal embeddings.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            embeddings: When using internal embedding function, embeddings will be "
            "ignored even if they are provided

        Returns:
            List[str]: empty list
        """
        sql_params = []
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas else {}
            metadata, extracted_special_metadata = self._split_off_special_metadata(
                metadata
            )
            parameters = {
                "content": text,  # Replace `content_value` with the actual value
                "metadata": json.dumps(
                    HanaDB._sanitize_metadata_keys(metadata)
                ),  # Replace `metadata_value` with the actual value
                "model_version": self.internal_embedding_model_id,
            }
            parameters.update(
                {
                    col: value
                    for col, value in zip(
                        self.specific_metadata_columns, extracted_special_metadata
                    )
                }
            )  # specific_metadata_values must align with the columns
            sql_params.append(parameters)

        specific_metadata_str = ", ".join(
            f":{col}" for col in self.specific_metadata_columns
        )
        specific_metadata_columns_string = self._get_specific_metadata_columns_string()

        sql_str = (
            f'INSERT INTO "{self.table_name}" ("{self.content_column}", '
            f'"{self.metadata_column}", '
            f'"{self.vector_column}"{specific_metadata_columns_string}) '
            f"VALUES (:content, :metadata, VECTOR_EMBEDDING"
            f"(:content, 'DOCUMENT', :model_version) "
            f"{(', ' + specific_metadata_str) if specific_metadata_str else ''});"
        )

        # Insert data into the table
        cur = self.connection.cursor()
        try:
            cur.executemany(sql_str, sql_params)
        finally:
            cur.close()
        return []

    def _validate_internal_embedding_function(self) -> None:
        """
        Ping the database to check if the in-database embedding function
            exists and works.
        Raises:
            RuntimeError: If the embedding function does not exist or fails.
        """
        cur = self.connection.cursor()
        try:
            # Test the VECTOR_EMBEDDING function by executing a simple query
            cur.execute(
                (
                    "SELECT TO_NVARCHAR("
                    "VECTOR_EMBEDDING('test', 'QUERY', :model_version))"
                    "FROM sys.DUMMY;"
                ),
                model_version=self.internal_embedding_model_id,
            )
            cur.fetchall()  # Ensure the query runs successfully

        except Exception as e:  # Catch all database-related exceptions
            raise RuntimeError(
                f"Validation of the internal embedding function failed: {str(e)}. "
            )
        finally:
            cur.close()

    def similarity_search_with_score(
        self, query: str, k: int = 4, filter: Optional[dict] = None
    ) -> list[tuple[Document, float]]:
        """Return documents and score values most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: A dictionary of metadata fields and values to filter by.
                    Defaults to None.

        Returns:
            List of tuples (containing a Document and a score) that are
            most similar to the query
        """
        return self.similarity_search_with_score_by_query(
            query=query, k=k, filter=filter
        )

    def similarity_search_by_vector(
        self, embedding: list[float], k: int = 4, filter: Optional[dict] = None
    ) -> list[Document]:
        raise NotImplementedError(
            "The method 'similarity_search_by_vector' is not supported "
            "when using HANA's internal embeddings"
        )

    def similarity_search_by_query(
        self, query: str, k: int = 4, filter: Optional[dict] = None
    ) -> list[Document]:
        """Return docs most similar to the given query.
        The query is vectorized in the database.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: A dictionary of metadata fields and values to filter by.
                    Defaults to None.

        Returns:
            List of Documents most similar to the query vector.
        """
        docs_and_scores = self.similarity_search_with_score_by_query(
            query=query, k=k, filter=filter
        )
        return [doc for doc, _ in docs_and_scores]

    def similarity_search_with_score_by_query(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
    ) -> list[tuple[Document, float]]:
        """Return docs most similar to the given embedding or query.
        The query is vectorized in the database.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: A dictionary of metadata fields and values to filter by.
                    Defaults to None.

        Returns:
            list of Documents most similar to the query and score for each
        """
        whole_result = self.similarity_search_with_score_and_vector_by_query(
            query=query, k=k, filter=filter
        )
        return [(result_item[0], result_item[1]) for result_item in whole_result]

    def similarity_search_with_score_and_vector_by_query(
        self, query: str, k: int = 4, filter: Optional[dict] = None
    ) -> list[tuple[Document, float, list[float]]]:
        """Return docs most similar to the given embedding.
        The query is vectorized in the database.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: A dictionary of metadata fields and values to filter by.
                    Defaults to None.

        Returns:
            List of tuples, each containing:
            - Document: The matched document with its content and metadata
            - float: The similarity score
            - list[float]: The document's embedding vector
        """
        embedding_expr = "VECTOR_EMBEDDING(?, 'QUERY', ?)"
        vector_embedding_params = [query, self.internal_embedding_model_id]
        return self._similarity_search_with_score_and_vector(
            embedding_expr,
            vector_embedding_params=vector_embedding_params,
            k=k,
            filter=filter,
        )

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[dict] = None,
    ) -> list[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            query: search query text.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
            filter: Filter on metadata properties, e.g.
                            {
                                "str_property": "foo",
                                "int_property": 123
                            }
        Returns:
            List of Documents selected by maximal marginal relevance.
        """

        return self.max_marginal_relevance_search_by_query(
            query=query,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
        )

    def max_marginal_relevance_search_by_query(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[dict] = None,
    ) -> list[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Similar documents with their embeddings are retrived
        using query which is vectorized in database.
        Additionally, the query is embedded using internal embedding function
        """
        sql_str = (
            "SELECT TO_NVARCHAR("
            "VECTOR_EMBEDDING(:content, 'QUERY', :model_version)) FROM sys.DUMMY;"
        )
        embedding = None
        cur = self.connection.cursor()
        try:
            cur.execute(
                sql_str,
                content=query,
                model_version=self.internal_embedding_model_id,
            )
            if cur.has_result_set():
                res = cur.fetchall()
                embedding = json.loads(res[0][0])
        finally:
            cur.close()

        whole_result = self.similarity_search_with_score_and_vector_by_query(
            query=query, k=fetch_k, filter=filter
        )
        embeddings = [result_item[2] for result_item in whole_result]
        mmr_doc_indexes = maximal_marginal_relevance(
            np.array(embedding), embeddings, lambda_mult=lambda_mult, k=k
        )

        return [whole_result[i][0] for i in mmr_doc_indexes]

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[dict] = None,
    ) -> list[Document]:
        raise NotImplementedError(
            "The method 'max_marginal_relevance_search_by_vector' is not supported "
            "when using HANA's internal embeddings"
        )

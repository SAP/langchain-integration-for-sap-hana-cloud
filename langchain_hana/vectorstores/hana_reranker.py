import json
import logging
from typing import Any, Sequence

from hdbcli import dbapi
from langchain_core.documents import BaseDocumentCompressor, Document
from pydantic import ConfigDict, Field, model_validator
from langchain_hana.vectorstores.utils import (
    _generate_cross_encode_sql_and_params,
    _sanitize_metadata_keys
)

logger = logging.getLogger(__name__)

class HanaReranker(BaseDocumentCompressor):
    """Document compressor that uses Internal SAP Models to Rerank."""

    connection: dbapi.Connection

    model_id: str = Field(
        description="Model to use for reranking.",
    )
    """Model to use for reranking."""

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    @model_validator(mode="after")
    def validate_model_supported(self) -> Any:
        """Validate that the provided model is supported by SAP HANA for reranking."""
        with self.connection.cursor() as cur:
            try:
                cur.execute(
                    # CROSS_ENCODE IS A WINDOW FUNCTION
                    # passing a single text through a "'test'""
                    f"SELECT {_generate_cross_encode_sql_and_params("'test'", '', 'test', [], self.model_id)[0]} FROM SYS.DUMMY",
                    ["test", self.model_id],
                )
            except dbapi.Error as e:
                logger.error(f"Database error while validating rerank model ID: {e}")
                raise
        return self

    def rerank(
        self,
        documents: Sequence[Document],
        query: str,
        top_n: int = 3,
        return_documents: bool = True,
        rank_fields: list[str] = [],
    ) -> list[tuple[int, Document, float]]:
        """Reranks documents based on relevance to the query using SAP HANA's CROSS_ENCODE function.
        Args:
            documents: A sequence of Document objects to be reranked.
            query: The query string to compare the documents against.
            top_n: Optional number of top results to return. If not provided, uses the default top_n.
            return_documents: Whether to return the documents in the reranking results.
            rank_fields: additional list of metadata fields to include in the reranking along with the page_content. Defaults to empty.
        Returns:
            A list of tuples containing the index, document, and score, ordered by relevance.
        """

        if top_n <= 0 or top_n > len(documents):
            raise ValueError(
                "top_n must be greater than 0 and less than or equal to the number of documents"
            )

        _sanitize_metadata_keys(rank_fields)  # Validate rank_fields

        document_idx_with_scores = []

        with self.connection.cursor() as cur:
            temp_table_name = "#RERANK_DOCS"
            create_temp_table_sql = f"""
            CREATE LOCAL TEMPORARY TABLE {temp_table_name} (
                ID NVARCHAR(5000),
                TEXT NCLOB,
                METADATA NCLOB
            );
            """
            try:
                cur.execute(create_temp_table_sql)
            except Exception as e:
                raise RuntimeError(f"Error creating temporary table for reranking: {e}")

            try:
                insert_sql = f"INSERT INTO {temp_table_name} (ID, TEXT, METADATA) VALUES (?, ?, ?)"
                insert_sql_params = []
                for doc in documents:
                    _sanitize_metadata_keys(list(doc.metadata.keys()))
                    insert_sql_params.append(
                        (
                            doc.id,
                            doc.page_content,
                            json.dumps(doc.metadata),
                        )
                    )
                cur.executemany(insert_sql, insert_sql_params)

                cross_encode_sql, cross_encode_params = (
                    _generate_cross_encode_sql_and_params(
                        "TEXT", "METADATA", query, rank_fields, self.model_id
                    )
                )

                reranking_sql = f"""
                SELECT
                    TOP {top_n}
                    ROW_NUMBER() OVER () - 1 AS INDEX,
                    ID,
                    TEXT,
                    METADATA,
                    {cross_encode_sql} AS SCORE
                FROM {temp_table_name}
                ORDER BY SCORE DESC
                """
                cur.execute(reranking_sql, cross_encode_params)
                rows = cur.fetchall()
                for row in rows:
                    idx, doc_id, text, metadata_json, score = row
                    if return_documents:
                        metadata = json.loads(metadata_json)
                        document = Document(
                            id=doc_id, page_content=text, metadata=metadata
                        )
                        document_idx_with_scores.append((idx, score, document))
                    else:
                        document_idx_with_scores.append((idx, score))
            finally:
                cur.execute(
                    f"DROP TABLE {temp_table_name}"
                )  # Ensure temp table is dropped

        return document_idx_with_scores

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
    ) -> list[Document]:
        """Compress documents using the rerank method.
        Args:
            documents: A sequence of Document objects to be compressed.
            query: The query string to compare the documents against for relevance.
        Returns:
            A list of Document objects reranked according to relevance to the query.
            Only the top 5 documents are returned, or fewer if there are less than 5 documents.
            The scores are added to the metadata of each Document under the key "relevance_score".
        """

        compressed = []

        reranked_results = self.rerank(documents=documents, query=query, top_n=min(5, len(documents)))

        for idx, score, doc in reranked_results:
            doc.metadata["relevance_score"] = score
            compressed.append(doc)

        return compressed

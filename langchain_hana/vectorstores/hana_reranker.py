# from __future__ import annotations

# import logging
from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Union

from langchain_core.callbacks.base import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document
from langchain_core.utils import secret_from_env
from hdbcli import dbapi
from pydantic import AliasChoices, ConfigDict, Field, SecretStr, model_validator


# logger = logging.getLogger(__name__)


class HanaReranker(BaseDocumentCompressor):
    """Document compressor that uses Internal SAP Models to Rerank."""

    connection: dbapi.Connection
    # top_n: Optional[int] = 3
    # """Number of documents to return."""
    model: str = Field(
        default="SAP_CER.20250701",
        description="Model to use for reranking. Default is 'SAP_CER.20250701'.",
    )
    """Model to use for reranking."""

    rank_fields: Optional[List[str]] = None
    """Fields to use for reranking when documents are dictionaries."""
    # return_documents: bool = True
    # """Whether to return the documents in the reranking results."""

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    # @model_validator(mode="after")
    # def validate_model_supported(self) -> Any:
    #     """Validate that the provided model is supported by SAP HANA for reranking."""
    #     supported = self.list_supported_models()
    #     supported_names = [m["model"] for m in supported]
    #     if self.model not in supported_names:
    #         raise ValueError(
    #             f"Model '{self.model}' is not a supported SAP HANA reranker model. Supported: {supported_names}"
    #         )
    #     return self

    # def list_supported_models(self, vector_type: Optional[str] = None) -> list:
    #     """Return a list of supported embedding models from SAP HANA."""
    #     api_key = self.pinecone_api_key.get_secret_value()
    #     return get_pinecone_supported_models(
    #         api_key, model_type="rerank", vector_type=vector_type
    #     )

    # async def alist_supported_models(self, vector_type: Optional[str] = None) -> list:
    #     """Return a list of supported reranker models from Pinecone asynchronously."""
    #     api_key = self.pinecone_api_key.get_secret_value()
    #     return await aget_pinecone_supported_models(
    #         api_key, model_type="rerank", vector_type=vector_type
    #     )

    # def _document_to_dict(
    #     self,
    #     document: Union[str, Document, dict],
    #     index: int,
    # ) -> dict:
    #     if isinstance(document, Document):
    #         doc_id_from_meta = document.metadata.get("id")
    #         if isinstance(doc_id_from_meta, str) and doc_id_from_meta:
    #             doc_id = doc_id_from_meta
    #         else:  # Generate ID if not valid
    #             doc_id = f"doc_{index}"

    #         doc_data = {
    #             "id": doc_id,
    #             "text": document.page_content,
    #             **document.metadata,
    #         }
    #         return doc_data
    #     elif isinstance(document, dict):
    #         current_id = document.get("id")
    #         if not isinstance(current_id, str) or not current_id:
    #             document["id"] = f"doc_{index}"  # Generate and set ID if not valid
    #         return document
    #     else:
    #         return {"id": f"doc_{index}", "text": str(document)}

    # def _rerank_params(self, model: str, truncate: str) -> dict:
    #     """Returns the parameters for the rerank API call."""
    #     parameters = {}
    #     # Only include truncate parameter for models that support it
    #     if model != "cohere-rerank-3.5":
    #         parameters["truncate"] = truncate
    #     return parameters
    
    def rerank(
        self,
        documents : Sequence[Document],
        query: str,
        *,
        top_n: int = 3,
    ) -> list[tuple[int, Document, float]]:
        """Reranks documents based on relevance to the query using SAP HANA's CROSS_ENCODE function.
        Args:
            documents: A sequence of Document objects to be reranked.
            query: The query string to compare the documents against.
            top_n: Optional number of top results to return. If not provided, uses the default top_n.
        Returns:
            A list of tuples containing the index, document, and score, ordered by relevance.
        """

        document_idx_with_scores = []

        for idx, document in enumerate(documents):
            with self.connection.cursor() as cur:
                try:
                    cur.execute("SELECT CROSS_ENCODE(?, ?, ?) OVER() AS SCORE FROM SYS.DUMMY", [document.page_content, query, self.model])
                    score = cur.fetchone()[0]
                    document_idx_with_scores.append((idx, document, score))
                except Exception as e:
                    raise RuntimeError(f"Error during reranking document at index {idx}: {e}")
        
        sorted_docs = sorted(document_idx_with_scores, key=lambda x: x[2], reverse=True)
        return sorted_docs[:top_n]

    # def rerank(
    #     self,
    #     documents: Sequence[Union[str, Document, dict]],
    #     query: str,
    #     *,
    #     rank_fields: Optional[List[str]] = None,
    #     model: Optional[str] = None,
    #     top_n: Optional[int] = None,
    #     truncate: str = "END",
    # ) -> List[Dict[str, Any]]:
    #     """Returns an ordered list of documents ordered by their relevance to the provided query."""
    #     if len(documents) == 0:  # to avoid empty API call
    #         return []

    #     # Convert documents to dict format
    #     docs = [
    #         self._document_to_dict(document=doc, index=i)
    #         for i, doc in enumerate(documents)
    #     ]

    #     try:
    #         client = self._get_sync_client()

    #         # Use self.model if model is None
    #         model_to_use = model if model is not None else self.model
    #         if model_to_use is None:  # This should never happen due to validator
    #             raise ValueError("No model specified for reranking")

    #         rerank_result = client.inference.rerank(
    #             model=model_to_use,
    #             query=query,
    #             documents=docs,
    #             rank_fields=rank_fields or self.rank_fields or ["text"],
    #             top_n=top_n or self.top_n,
    #             return_documents=self.return_documents,
    #             parameters=self._rerank_params(model=model_to_use, truncate=truncate),
    #         )

    #         result_dicts = []
    #         for result_item_data in rerank_result.data:
    #             result_dict = {
    #                 "id": result_item_data.document.id,
    #                 "index": result_item_data.index,
    #                 "score": result_item_data.score,
    #             }

    #             if self.return_documents:
    #                 result_dict["document"] = result_item_data.document.to_dict()

    #             result_dicts.append(result_dict)

    #         return result_dicts

    #     except Exception as e:
    #         logger.error(f"Rerank error: {e}")
    #         return []

    # async def arerank(
    #     self,
    #     documents: Sequence[Union[str, Document, dict]],
    #     query: str,
    #     *,
    #     rank_fields: Optional[List[str]] = None,
    #     model: Optional[str] = None,
    #     top_n: Optional[int] = None,
    #     truncate: str = "END",
    # ) -> List[Dict[str, Any]]:
    #     """Async rerank documents using Pinecone's reranking API."""
    #     if len(documents) == 0:  # to avoid empty API call
    #         return []

    #     docs = [
    #         self._document_to_dict(document=doc, index=i)
    #         for i, doc in enumerate(documents)
    #     ]

    #     try:
    #         client = await self._get_async_client()

    #         # Use self.model if model is None
    #         model_to_use = model if model is not None else self.model
    #         if model_to_use is None:  # This should never happen due to validator
    #             raise ValueError("No model specified for reranking")

    #         rerank_result = await client.inference.rerank(
    #             model=model_to_use,
    #             query=query,
    #             documents=docs,
    #             rank_fields=rank_fields or self.rank_fields or ["text"],
    #             top_n=top_n or self.top_n,
    #             return_documents=self.return_documents,
    #             parameters=self._rerank_params(model=model_to_use, truncate=truncate),
    #         )

    #         result_dicts = []
    #         for result_item_data in rerank_result.data:
    #             result_dict = {
    #                 "id": result_item_data.document.id,
    #                 "index": result_item_data.index,
    #                 "score": result_item_data.score,
    #             }

    #             if self.return_documents:
    #                 result_dict["document"] = result_item_data.document.to_dict()

    #             result_dicts.append(result_dict)

    #         return result_dicts
    #     except Exception as e:
    #         logger.error(f"Async rerank error: {e}")
    #         return []

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
            Only the top 5 documents are returned.
            The scores are added to the metadata of each Document under the key "relevance_score".
        """

        compressed = []

        reranked_results = self.rerank(documents=documents, query=query, top_n = 5)

        for idx, doc, score in reranked_results:
            doc_copy = Document(doc.page_content, metadata=deepcopy(doc.metadata))
            doc_copy.metadata["relevance_score"] = score
            compressed.append(doc_copy)

        return compressed
    

    # def compress_documents(
    #     self,
    #     documents: Sequence[Document],
    #     query: str,
    #     callbacks: Optional[Callbacks] = None,
    # ) -> Sequence[Document]:
    #     """Compress documents using Pinecone's rerank API."""
    #     if not documents:
    #         return []

    #     compressed = []
    #     reranked_results = self.rerank(documents=documents, query=query)

    #     if not reranked_results:
    #         return []

    #     for res in reranked_results:
    #         if res["index"] is not None:
    #             doc_index = res["index"]
    #             if 0 <= doc_index < len(documents):
    #                 doc = documents[doc_index]
    #                 doc_copy = Document(
    #                     doc.page_content, metadata=deepcopy(doc.metadata)
    #                 )
    #                 doc_copy.metadata["relevance_score"] = res["score"]
    #                 compressed.append(doc_copy)

    #     return compressed

    # async def acompress_documents(
    #     self,
    #     documents: Sequence[Document],
    #     query: str,
    #     callbacks: Optional[Callbacks] = None,
    # ) -> Sequence[Document]:
    #     """Async compress documents using Pinecone's rerank API."""
    #     if not documents:
    #         return []

    #     compressed = []
    #     reranked_results = await self.arerank(documents=documents, query=query)

    #     if not reranked_results:
    #         return []

    #     for res in reranked_results:
    #         if res["index"] is not None:
    #             doc_index = res["index"]
    #             if 0 <= doc_index < len(documents):
    #                 doc = documents[doc_index]
    #                 doc_copy = Document(
    #                     doc.page_content, metadata=deepcopy(doc.metadata)
    #                 )
    #                 doc_copy.metadata["relevance_score"] = res["score"]
    #                 compressed.append(doc_copy)

    #     return compressed

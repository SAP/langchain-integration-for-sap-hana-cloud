import os
from typing import Any, Generator

import pytest
from hdbcli import dbapi
from langchain_core.documents import Document

from langchain_hana import HanaReranker
from tests.integration_tests.hana_test_constants import HanaTestConstants


class Config:
    def __init__(self) -> None:
        self.conn: dbapi.Connection


config = Config()


def setup_module(module: Any) -> None:
    config.conn = dbapi.connect(
        address=os.environ["HANA_DB_ADDRESS"],
        port=int(os.environ["HANA_DB_PORT"]),
        user=os.environ["HANA_DB_USER"],
        password=os.environ["HANA_DB_PASSWORD"],
    )


def teardown_module(module: Any) -> None:
    config.conn.close()


@pytest.fixture(scope="module")
def reranker() -> Generator[HanaReranker, None, None]:
    yield HanaReranker(
        connection=config.conn,
        model_id=os.environ["HANA_DB_RERANK_MODEL_ID"],
    )


@pytest.fixture
def documents() -> list[Document]:
    return [
        Document(page_content=text, metadata=metadata)
        for text, metadata in zip(HanaTestConstants.TEXTS, HanaTestConstants.METADATAS)
    ]


@pytest.mark.parametrize(
    "query, top_n, return_documents, rank_fields, expected_idx",
    [
        (HanaTestConstants.TEXTS[0], 3, True, [], 0),
        (HanaTestConstants.TEXTS[1], 2, False, [], 1),
        (HanaTestConstants.TEXTS[2], 4, True, ["quality"], 2),
        (HanaTestConstants.TEXTS[3], 1, False, ["Owner", "quality"], 3),
    ],
)
def test_rerank(
    reranker: HanaReranker,
    documents: list[Document],
    query: str,
    top_n: int,
    return_documents: bool,
    rank_fields: list[str],
    expected_idx: int,
) -> None:
    result = reranker.rerank(  # type: ignore[call-overload]
        documents, query, top_n, return_documents, rank_fields
    )
    assert result
    assert len(result) == top_n
    assert result[0][0] == expected_idx

    prev_score = 1.0
    for item in result:
        if return_documents:
            assert isinstance(item[2], Document)
        assert item[1] <= prev_score
        prev_score = item[1]


@pytest.mark.parametrize("invalid_top_n", [-1, 0, len(HanaTestConstants.TEXTS) + 1])
def test_rerank_with_invalid_top_n(
    reranker: HanaReranker, documents: list[Document], invalid_top_n: int
) -> None:
    with pytest.raises(
        ValueError,
        match="top_n must be greater than 0 and less than or equal to "
        "the number of documents",
    ):
        reranker.rerank(documents, HanaTestConstants.TEXTS[0], invalid_top_n)


def test_rerank_with_invalid_metadata_key(
    reranker: HanaReranker, documents: list[Document]
) -> None:
    with pytest.raises(ValueError, match="Invalid metadata key invalid-key"):
        reranker.rerank(
            documents, HanaTestConstants.TEXTS[0], rank_fields=["invalid-key"]
        )


def test_compress_documents(
    reranker: HanaReranker, documents: list[Document]
) -> None:
    documents.append(
        Document(
            page_content="abc",
            metadata={"start": 400, "quality": "ugly", "Owner": "Bob"},
        )
    )
    compressed_docs = reranker.compress_documents(
        query=HanaTestConstants.TEXTS[0], documents=documents
    )
    assert compressed_docs
    assert len(compressed_docs) == 5

    for doc in compressed_docs:
        assert isinstance(doc, Document)
        assert "relevance_score" in doc.metadata
        assert isinstance(doc.metadata["relevance_score"], float)
    assert compressed_docs[0].page_content == HanaTestConstants.TEXTS[0]

    prev_score = compressed_docs[0].metadata["relevance_score"]
    for doc in compressed_docs:
        assert doc.metadata["relevance_score"] <= prev_score
        prev_score = doc.metadata["relevance_score"]

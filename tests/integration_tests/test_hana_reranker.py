import os
from hdbcli import dbapi
from langchain_core.documents import Document
import pytest
from langchain_hana import HanaReranker
from tests.integration_tests.hana_test_constants import HanaTestConstants

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
    )
    
def teardown_module(module):  # type: ignore[no-untyped-def]
    config.conn.close()
    config.conn = None


@pytest.fixture
def reranker():
    return HanaReranker(
        connection=config.conn,
        model_id=os.environ["HANA_DB_RERANK_MODEL_ID"],
    )


@pytest.fixture
def documents():
    return [Document(page_content=text, metadata=metadata) for text, metadata in zip(HanaTestConstants.TEXTS, HanaTestConstants.METADATAS)]


@pytest.mark.parametrize("query, top_n, return_documents, rank_fields, expected_idx", [
    (HanaTestConstants.TEXTS[0], 3, True, [], 0),
    (HanaTestConstants.TEXTS[1], 2, False, [], 1),
    (HanaTestConstants.TEXTS[2], 4, True, ["quality"], 2),
    (HanaTestConstants.TEXTS[3], 1, False, ["Owner", "quality"], 3),
])
def test_rerank(reranker, documents, query, top_n, return_documents, rank_fields, expected_idx):
    result = reranker.rerank(documents, query, top_n, return_documents, rank_fields)
    assert result
    assert len(result) == top_n
    assert result[0][0] == expected_idx

    prev_score = 1.0
    for item in result:
        if return_documents:
            assert isinstance(item[2], Document)
        assert item[1] <= prev_score
        prev_score = item[1]


def test_compress_documents(reranker, documents):
    documents.append(Document(page_content="abc", metadata={"start": 400, "quality": "ugly", "Owner": "Bob"}))
    compressed_docs = reranker.compress_documents(query=HanaTestConstants.TEXTS[0], documents=documents)
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

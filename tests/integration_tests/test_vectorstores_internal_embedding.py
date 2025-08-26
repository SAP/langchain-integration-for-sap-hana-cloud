"""Test HANA vectorstore's internal embedding functionality."""

import os

import pytest
from hdbcli import dbapi

from langchain_hana.embeddings import HanaInternalEmbeddings
from langchain_hana.vectorstores import HanaDB
from tests.integration_tests.hana_test_utils import HanaTestUtils


class ConfigData:
    def __init__(self):  # type: ignore[no-untyped-def]
        self.conn = None
        self.schema_name = ""


test_setup = ConfigData()

embedding = None


def is_internal_embedding_available(connection, embedding) -> bool:
    """
    Check if the internal embedding function is available in HANA DB.
    Returns:
        bool: True if available, False otherwise.
    """
    if embedding.model_id is None:
        return False
    try:
        cur = connection.cursor()
        # Test the VECTOR_EMBEDDING function by executing a simple query
        cur.execute(
            (
                "SELECT TO_NVARCHAR("
                "VECTOR_EMBEDDING('test', 'QUERY', :model_version))"
                "FROM sys.DUMMY;"
            ),
            model_version=embedding.model_id,
        )
        cur.fetchall()  # Ensure the query executes successfully
        return True
    except Exception:
        return False
    finally:
        cur.close()


def setup_module(module):  # type: ignore[no-untyped-def]
    test_setup.conn = dbapi.connect(
        address=os.environ.get("HANA_DB_ADDRESS"),
        port=os.environ.get("HANA_DB_PORT"),
        user=os.environ.get("HANA_DB_USER"),
        password=os.environ.get("HANA_DB_PASSWORD"),
        autocommit=True,
        sslValidateCertificate=False,
        # encrypt=True
    )

    global embedding
    embedding_model_id = os.environ.get("HANA_DB_EMBEDDING_MODEL_ID")
    embedding = HanaInternalEmbeddings(internal_embedding_model_id=embedding_model_id)

    if not is_internal_embedding_available(test_setup.conn, embedding):
        pytest.fail(
            f"Internal embedding function is not available "
            f"or the model id {embedding.model_id} is wrong"
        )

    schema_prefix = "LANGCHAIN_INT_EMB_TEST"
    HanaTestUtils.drop_old_test_schemas(test_setup.conn, schema_prefix)
    test_setup.schema_name = HanaTestUtils.generate_schema_name(
        test_setup.conn, schema_prefix
    )
    HanaTestUtils.create_and_set_schema(test_setup.conn, test_setup.schema_name)


def teardown_module(module):  # type: ignore[no-untyped-def]
    HanaTestUtils.drop_schema_if_exists(test_setup.conn, test_setup.schema_name)


@pytest.fixture
def texts() -> list[str]:
    return ["foo", "bar", "baz", "bak", "cat"]


@pytest.fixture
def metadatas() -> list[str]:
    return [
        {"start": 0, "end": 100, "quality": "good", "ready": True},  # type: ignore[list-item]
        {"start": 100, "end": 200, "quality": "bad", "ready": False},  # type: ignore[list-item]
        {"start": 200, "end": 300, "quality": "ugly", "ready": True},  # type: ignore[list-item]
        {"start": 200, "quality": "ugly", "ready": True, "Owner": "Steve"},  # type: ignore[list-item]
        {"start": 300, "quality": "ugly", "Owner": "Steve"},  # type: ignore[list-item]
    ]


def test_hanavector_add_texts(texts: list[str], metadatas: list[dict]) -> None:
    table_name = "TEST_TABLE_ADD_TEXTS"

    # Check if table is created
    vectordb = HanaDB(
        connection=test_setup.conn, embedding=embedding, table_name=table_name
    )

    vectordb.add_texts(texts=texts, metadatas=metadatas)

    # check that embeddings have been created in the table
    number_of_texts = len(texts)
    number_of_rows = -1
    sql_str = f"SELECT COUNT(*) FROM {table_name}"
    cur = test_setup.conn.cursor()
    cur.execute(sql_str)
    if cur.has_result_set():
        rows = cur.fetchall()
        number_of_rows = rows[0][0]
    assert number_of_rows == number_of_texts


def test_hanavector_similarity_search_with_metadata_filter(
    texts: list[str], metadatas: list[dict]
) -> None:
    table_name = "TEST_TABLE_FILTER"

    # Check if table is created
    vectorDB = HanaDB.from_texts(
        connection=test_setup.conn,
        texts=texts,
        metadatas=metadatas,
        embedding=embedding,
        table_name=table_name,
    )

    search_result = vectorDB.similarity_search(texts[0], 3, filter={"start": 100})

    assert len(search_result) == 1
    assert texts[1] == search_result[0].page_content
    assert metadatas[1]["start"] == search_result[0].metadata["start"]
    assert metadatas[1]["end"] == search_result[0].metadata["end"]

    search_result = vectorDB.similarity_search(
        texts[0], 3, filter={"start": 100, "end": 150}
    )
    assert len(search_result) == 0

    search_result = vectorDB.similarity_search(
        texts[0], 3, filter={"start": 100, "end": 200}
    )
    assert len(search_result) == 1
    assert texts[1] == search_result[0].page_content
    assert metadatas[1]["start"] == search_result[0].metadata["start"]
    assert metadatas[1]["end"] == search_result[0].metadata["end"]


@pytest.mark.parametrize("k", [0, -4])
def test_hanavector_similarity_search_simple_invalid(texts: list[str], k: int) -> None:
    table_name = "TEST_TABLE_SEARCH_SIMPLE_INVALID"

    # Check if table is created
    vectorDB = HanaDB.from_texts(
        connection=test_setup.conn,
        texts=texts,
        embedding=embedding,
        table_name=table_name,
    )

    with pytest.raises(ValueError, match="must be an integer greater than 0"):
        vectorDB.similarity_search(texts[0], k)


def test_hanavector_max_marginal_relevance_search(texts: list[str]) -> None:
    table_name = "TEST_TABLE_MAX_RELEVANCE"

    # Check if table is created
    vectorDB = HanaDB.from_texts(
        connection=test_setup.conn,
        texts=texts,
        embedding=embedding,
        table_name=table_name,
    )

    search_result = vectorDB.max_marginal_relevance_search(texts[0], k=2, fetch_k=20)

    assert len(search_result) == 2
    assert search_result[0].page_content == texts[0]
    assert search_result[1].page_content != texts[0]


@pytest.mark.parametrize(
    "k, fetch_k, error_msg",
    [
        (0, 20, "must be an integer greater than 0"),
        (-4, 20, "must be an integer greater than 0"),
        (2, 0, "greater than or equal to 'k'"),
    ],
)
def test_hanavector_max_marginal_relevance_search_invalid(
    texts: list[str], k: int, fetch_k: int, error_msg: str
) -> None:
    table_name = "TEST_TABLE_MAX_RELEVANCE_INVALID"

    # Check if table is created
    vectorDB = HanaDB.from_texts(
        connection=test_setup.conn,
        texts=texts,
        embedding=embedding,
        table_name=table_name,
    )

    with pytest.raises(ValueError, match=error_msg):
        vectorDB.max_marginal_relevance_search(texts[0], k, fetch_k)


@pytest.mark.parametrize(
    "k, fetch_k, error_msg",
    [
        (0, 20, "must be an integer greater than 0"),
        (-4, 20, "must be an integer greater than 0"),
        (2, 0, "greater than or equal to 'k'"),
    ],
)
async def test_hanavector_max_marginal_relevance_search_async_invalid(
    texts: list[str], k: int, fetch_k: int, error_msg: str
) -> None:
    table_name = "TEST_TABLE_MAX_RELEVANCE_ASYNC_INVALID"

    # Check if table is created
    vectorDB = HanaDB.from_texts(
        connection=test_setup.conn,
        texts=texts,
        embedding=embedding,
        table_name=table_name,
    )

    with pytest.raises(ValueError, match=error_msg):
        await vectorDB.amax_marginal_relevance_search(texts[0], k, fetch_k)

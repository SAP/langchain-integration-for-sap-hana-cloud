"""Test HANA vectorstore's internal embedding functionality."""

import os

import pytest
from hdbcli import dbapi

from langchain_hana.embeddings import HanaInternalEmbeddings
from langchain_hana.vectorstores import HanaDB
from tests.integration_tests.hana_test_utils import HanaTestUtils


class Config:
    def __init__(self):  # type: ignore[no-untyped-def]
        self.conn = None
        self.schema_name = ""


config = Config()

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
    config.conn = dbapi.connect(
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

    if not is_internal_embedding_available(config.conn, embedding):
        pytest.fail(
            f"Internal embedding function is not available "
            f"or the model id {embedding.model_id} is wrong"
        )

    schema_prefix = "LANGCHAIN_INT_EMB_TEST"
    HanaTestUtils.drop_old_test_schemas(config.conn, schema_prefix)
    config.schema_name = HanaTestUtils.generate_schema_name(config.conn, schema_prefix)
    HanaTestUtils.create_and_set_schema(config.conn, config.schema_name)


def teardown_module(module):  # type: ignore[no-untyped-def]
    HanaTestUtils.drop_schema_if_exists(config.conn, config.schema_name)


@pytest.fixture
def vectorDB_empty():
    table_name = "TEST_TABLE_EMPTY"
    vectorDB = HanaDB(
        connection=config.conn,
        embedding=embedding,
        table_name=table_name,
    )

    yield vectorDB

    HanaTestUtils.drop_table(config.conn, table_name)


@pytest.fixture
def vectorDB_with_texts():
    table_name = "TEST_TABLE_WITH_TEXTS"
    vectorDB = HanaDB.from_texts(
        connection=config.conn,
        embedding=embedding,
        texts=HanaTestUtils.TEXTS,
        table_name=table_name,
    )

    yield vectorDB

    HanaTestUtils.drop_table(config.conn, table_name)


@pytest.fixture
def vectorDB_with_texts_and_metadatas():
    table_name = "TEST_TABLE_WITH_TEXTS_AND_METADATAS"

    vectorDB = HanaDB.from_texts(
        connection=config.conn,
        embedding=embedding,
        texts=HanaTestUtils.TEXTS,
        metadatas=HanaTestUtils.METADATAS,
        table_name=table_name,
    )

    yield vectorDB

    HanaTestUtils.drop_table(config.conn, table_name)


def test_hanavector_add_texts(vectorDB_empty) -> None:
    table_name = "TEST_TABLE_EMPTY"

    vectorDB_empty.add_texts(
        texts=HanaTestUtils.TEXTS, metadatas=HanaTestUtils.METADATAS
    )

    # check that embeddings have been created in the table
    number_of_texts = len(HanaTestUtils.TEXTS)
    number_of_rows = -1
    sql_str = f"SELECT COUNT(*) FROM {table_name}"
    cur = config.conn.cursor()
    cur.execute(sql_str)
    if cur.has_result_set():
        rows = cur.fetchall()
        number_of_rows = rows[0][0]
    assert number_of_rows == number_of_texts


def test_hanavector_similarity_search_with_metadata_filter(
    vectorDB_with_texts_and_metadatas,
) -> None:
    search_result = vectorDB_with_texts_and_metadatas.similarity_search(
        HanaTestUtils.TEXTS[0], 3, filter={"start": 100}
    )

    assert len(search_result) == 1
    assert HanaTestUtils.TEXTS[1] == search_result[0].page_content
    assert HanaTestUtils.METADATAS[1]["start"] == search_result[0].metadata["start"]
    assert HanaTestUtils.METADATAS[1]["end"] == search_result[0].metadata["end"]

    search_result = vectorDB_with_texts_and_metadatas.similarity_search(
        HanaTestUtils.TEXTS[0], 3, filter={"start": 100, "end": 150}
    )
    assert len(search_result) == 0

    search_result = vectorDB_with_texts_and_metadatas.similarity_search(
        HanaTestUtils.TEXTS[0], 3, filter={"start": 100, "end": 200}
    )
    assert len(search_result) == 1
    assert HanaTestUtils.TEXTS[1] == search_result[0].page_content
    assert HanaTestUtils.METADATAS[1]["start"] == search_result[0].metadata["start"]
    assert HanaTestUtils.METADATAS[1]["end"] == search_result[0].metadata["end"]


@pytest.mark.parametrize("k", [0, -4])
def test_hanavector_similarity_search_simple_invalid(vectorDB, k: int) -> None:

    with pytest.raises(ValueError, match="must be an integer greater than 0"):
        vectorDB.similarity_search(HanaTestUtils.TEXTS[0], k)


def test_hanavector_max_marginal_relevance_search(
    texts: list[str], vectorDB_with_texts
) -> None:
    search_result = vectorDB_with_texts.max_marginal_relevance_search(
        HanaTestUtils.TEXTS[0], k=2, fetch_k=20
    )

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
    vectorDB, k: int, fetch_k: int, error_msg: str
) -> None:

    with pytest.raises(ValueError, match=error_msg):
        vectorDB.max_marginal_relevance_search(HanaTestUtils.TEXTS[0], k, fetch_k)


@pytest.mark.parametrize(
    "k, fetch_k, error_msg",
    [
        (0, 20, "must be an integer greater than 0"),
        (-4, 20, "must be an integer greater than 0"),
        (2, 0, "greater than or equal to 'k'"),
    ],
)
async def test_hanavector_max_marginal_relevance_search_async_invalid(
    vectorDB, k: int, fetch_k: int, error_msg: str
) -> None:

    with pytest.raises(ValueError, match=error_msg):
        await vectorDB.amax_marginal_relevance_search(HanaTestUtils.TEXTS[0], k, fetch_k)

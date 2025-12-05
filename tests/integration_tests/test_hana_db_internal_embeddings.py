"""Test HANA vectorstore's internal embedding functionality."""

import os

import pytest
from hdbcli import dbapi

from langchain_hana import HanaInternalEmbeddings
from langchain_hana import HanaDB
from tests.integration_tests.hana_test_constants import HanaTestConstants
from tests.integration_tests.hana_test_utils import HanaTestUtils


class Config:
    def __init__(self):  # type: ignore[no-untyped-def]
        self.conn = None
        self.schema_name = ""
        self.embedding = None


config = Config()


def is_internal_embedding_available(connection, embedding) -> bool:
    """
    Check if the internal embedding function is available in HANA DB.
    Returns:
        bool: True if available, False otherwise.
    """
    if embedding.model_id is None:
        return False
    sql_params = [embedding.model_id]
    if not embedding.remote_source:
        sql_str = """VECTOR_EMBEDDING('test', 'QUERY', ?)"""
    else:
        sql_str = f"""VECTOR_EMBEDDING('test', 'QUERY', ?, "{embedding.remote_source}")"""
    try:
        cur = connection.cursor()
        # Test the VECTOR_EMBEDDING function by executing a simple query
        cur.execute(
            (
                "SELECT TO_NVARCHAR("
                f"{sql_str})"
                "FROM sys.DUMMY;"
            ),
            sql_params,
        )
        cur.fetchall()  # Ensure the query executes successfully
        return True
    except Exception as e:
        print(f"Error checking internal embedding availability: {e}")
        return False
    finally:
        cur.close()

@pytest.fixture(scope="module", params=[None, "memoryview", "list", "tuple"], 
                ids=["default", "memoryview", "list", "tuple"])
def vectoroutputtype_param(request):  # type: ignore[no-untyped-def]
    """Parametrize vectoroutputtype values for testing."""
    return request.param

@pytest.fixture(scope="module", params=[{
    "internal_embedding_model_id": os.environ["HANA_DB_EMBEDDING_MODEL_ID"],
}, {
    "internal_embedding_model_id": os.environ["HANA_DB_EMBEDDING_REMOTE_MODEL_ID"],
    "remote_source": os.environ["HANA_DB_EMBEDDING_REMOTE_SOURCE"],
}], 
                ids=["without_remote_source", "with_remote_source"])
def embedding_param(request):  # type: ignore[no-untyped-def]
    """Parametrize embedding values for testing."""
    return request.param


@pytest.fixture(scope="module", autouse=True)
def setup_connection(vectoroutputtype_param, embedding_param):  # type: ignore[no-untyped-def]
    """Setup connection with specific vectoroutputtype parameter."""
    # Build connection parameters
    conn_params = {
        "address": os.environ["HANA_DB_ADDRESS"],
        "port": os.environ["HANA_DB_PORT"],
        "user": os.environ["HANA_DB_USER"],
        "password": os.environ["HANA_DB_PASSWORD"],
        "autocommit": True,
        "sslValidateCertificate": False,
    }
    
    # Only add vectoroutputtype if it's not None (to test default behavior)
    if vectoroutputtype_param is not None:
        conn_params["vectoroutputtype"] = vectoroutputtype_param
    
    config.conn = dbapi.connect(**conn_params)
    config.embedding = HanaInternalEmbeddings(**embedding_param)

    if not is_internal_embedding_available(config.conn, config.embedding):
        pytest.fail(
            f"Internal embedding function is not available "
            f"or the model id {config.embedding.model_id} is wrong"
        )

    schema_prefix = "LANGCHAIN_INT_EMB_TEST"
    HanaTestUtils.drop_old_test_schemas(config.conn, schema_prefix)
    config.schema_name = HanaTestUtils.generate_schema_name(config.conn, schema_prefix)
    HanaTestUtils.create_and_set_schema(config.conn, config.schema_name)
    
    yield
    
    HanaTestUtils.drop_schema_if_exists(config.conn, config.schema_name)
    config.conn.close()

@pytest.fixture(params=["REAL_VECTOR", "HALF_VECTOR"])
def vectorDB(request):
    vectorDB = HanaDB(
        connection=config.conn,
        embedding=config.embedding,
        table_name=HanaTestConstants.TABLE_NAME,
        vector_column_type=request.param,
    )

    yield vectorDB

    HanaTestUtils.drop_table(config.conn, HanaTestConstants.TABLE_NAME)


def test_hanavector_add_texts(vectorDB) -> None:
    vectorDB.add_texts(
        texts=HanaTestConstants.TEXTS, metadatas=HanaTestConstants.METADATAS
    )

    # check that embeddings have been created in the table
    number_of_texts = len(HanaTestConstants.TEXTS)
    number_of_rows = -1
    sql_str = f"SELECT COUNT(*) FROM {HanaTestConstants.TABLE_NAME}"
    cur = config.conn.cursor()
    cur.execute(sql_str)
    if cur.has_result_set():
        rows = cur.fetchall()
        number_of_rows = rows[0][0]
    assert number_of_rows == number_of_texts


def test_hanavector_similarity_search_with_metadata_filter(
    vectorDB,
) -> None:
    vectorDB.add_texts(
        texts=HanaTestConstants.TEXTS, metadatas=HanaTestConstants.METADATAS
    )

    search_result = vectorDB.similarity_search(
        HanaTestConstants.TEXTS[0], 3, filter={"start": 100}
    )

    assert len(search_result) == 1
    assert HanaTestConstants.TEXTS[1] == search_result[0].page_content
    assert HanaTestConstants.METADATAS[1]["start"] == search_result[0].metadata["start"]
    assert HanaTestConstants.METADATAS[1]["end"] == search_result[0].metadata["end"]

    search_result = vectorDB.similarity_search(
        HanaTestConstants.TEXTS[0], 3, filter={"start": 100, "end": 150}
    )
    assert len(search_result) == 0

    search_result = vectorDB.similarity_search(
        HanaTestConstants.TEXTS[0], 3, filter={"start": 100, "end": 200}
    )
    assert len(search_result) == 1
    assert HanaTestConstants.TEXTS[1] == search_result[0].page_content
    assert HanaTestConstants.METADATAS[1]["start"] == search_result[0].metadata["start"]
    assert HanaTestConstants.METADATAS[1]["end"] == search_result[0].metadata["end"]


def test_hanavector_similarity_search_simple(vectorDB) -> None:
    vectorDB.add_texts(texts=HanaTestConstants.TEXTS)

    assert (
        HanaTestConstants.TEXTS[0]
        == vectorDB.similarity_search(HanaTestConstants.TEXTS[0], 1)[0].page_content
    )
    assert (
        HanaTestConstants.TEXTS[1]
        != vectorDB.similarity_search(HanaTestConstants.TEXTS[0], 1)[0].page_content
    )


@pytest.mark.parametrize("k", [0, -4])
def test_hanavector_similarity_search_simple_invalid(vectorDB, k: int) -> None:
    with pytest.raises(ValueError, match="must be an integer greater than 0"):
        vectorDB.similarity_search(HanaTestConstants.TEXTS[0], k)


def test_hanavector_max_marginal_relevance_search(vectorDB) -> None:
    vectorDB.add_texts(texts=HanaTestConstants.TEXTS)

    search_result = vectorDB.max_marginal_relevance_search(
        HanaTestConstants.TEXTS[0], k=2, fetch_k=20
    )

    assert len(search_result) == 2
    assert search_result[0].page_content == HanaTestConstants.TEXTS[0]
    assert search_result[1].page_content != HanaTestConstants.TEXTS[0]


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
        vectorDB.max_marginal_relevance_search(HanaTestConstants.TEXTS[0], k, fetch_k)


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
        await vectorDB.amax_marginal_relevance_search(
            HanaTestConstants.TEXTS[0], k, fetch_k
        )

"""Test HANA vectorstore functionality."""

import os
from typing import Any, Dict, List

import numpy as np
import pytest

from langchain_hana.utils import DistanceStrategy
from langchain_hana import HanaDB
from tests.integration_tests.fake_embeddings import ConsistentFakeEmbeddings
from tests.integration_tests.fixtures.filtering_test_cases import (
    DOCUMENTS,
    FILTERING_TEST_CASES,
)
from tests.integration_tests.hana_test_constants import HanaTestConstants
from tests.integration_tests.hana_test_utils import HanaTestUtils

try:
    from hdbcli import dbapi  # type: ignore

    hanadb_installed = True
except ImportError:
    hanadb_installed = False


class NormalizedFakeEmbeddings(ConsistentFakeEmbeddings):
    """Fake embeddings with normalization. For testing purposes."""

    def normalize(self, vector: List[float]) -> List[float]:
        """Normalize vector."""
        return [float(v / np.linalg.norm(vector)) for v in vector]

    def embed_documents(self, texts: list[str]) -> List[List[float]]:
        return [self.normalize(v) for v in super().embed_documents(texts)]

    def embed_query(self, text: str) -> List[float]:
        return self.normalize(super().embed_query(text))


embedding = NormalizedFakeEmbeddings()


class Config:
    def __init__(self):  # type: ignore[no-untyped-def]
        self.conn = None
        self.schema_name = ""


config = Config()


def setup_module(module):  # type: ignore[no-untyped-def]
    config.conn = dbapi.connect(
        address=os.environ["HANA_DB_ADDRESS"],
        port=os.environ["HANA_DB_PORT"],
        user=os.environ["HANA_DB_USER"],
        password=os.environ["HANA_DB_PASSWORD"],
        autocommit=True,
        sslValidateCertificate=False,
        vectoroutputtype="memoryview",
        # encrypt=True
    )
    schema_prefix = "LANGCHAIN_TEST"
    HanaTestUtils.drop_old_test_schemas(config.conn, schema_prefix)
    config.schema_name = HanaTestUtils.generate_schema_name(config.conn, schema_prefix)
    HanaTestUtils.create_and_set_schema(config.conn, config.schema_name)


def teardown_module(module):  # type: ignore[no-untyped-def]
    HanaTestUtils.drop_schema_if_exists(config.conn, config.schema_name)


@pytest.fixture(params=["REAL_VECTOR", "HALF_VECTOR"])
def vectorDB(request):
    vectorDB = HanaDB(
        connection=config.conn,
        embedding=embedding,
        table_name=HanaTestConstants.TABLE_NAME,
        vector_column_type=request.param,
    )

    yield vectorDB

    HanaTestUtils.drop_table(config.conn, HanaTestConstants.TABLE_NAME)

@pytest.fixture
def table_name_with_cleanup():
    yield HanaTestConstants.TABLE_NAME_CUSTOM_DB
    HanaTestUtils.drop_table(config.conn, HanaTestConstants.TABLE_NAME_CUSTOM_DB)


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_hanavector_non_existing_table(table_name_with_cleanup) -> None:
    """Test end to end construction and search."""
    table_name = table_name_with_cleanup

    # Check if table is created
    vectordb = HanaDB(
        connection=config.conn,
        embedding=embedding,
        distance_strategy=DistanceStrategy.COSINE,
        table_name=table_name,
    )

    assert vectordb._table_exists(table_name)


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_hanavector_table_with_missing_columns(table_name_with_cleanup) -> None:
    table_name = table_name_with_cleanup
    try:
        cur = config.conn.cursor()
        sql_str = f"CREATE TABLE {table_name}(WRONG_COL NVARCHAR(500));"
        cur.execute(sql_str)
    finally:
        cur.close()

    # Check if table is created
    exception_occured = False
    try:
        HanaDB(
            connection=config.conn,
            embedding=embedding,
            distance_strategy=DistanceStrategy.COSINE,
            table_name=table_name,
        )
        exception_occured = False
    except AttributeError:
        exception_occured = True
    assert exception_occured


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_hanavector_table_with_nvarchar_content(table_name_with_cleanup) -> None:
    table_name = table_name_with_cleanup
    content_column = "TEST_TEXT"
    metadata_column = "TEST_META"
    vector_column = "TEST_VECTOR"
    try:
        cur = config.conn.cursor()
        sql_str = (
            f"CREATE TABLE {table_name}({content_column} NVARCHAR(2048), "
            f"{metadata_column} NVARCHAR(2048), {vector_column} REAL_VECTOR);"
        )
        cur.execute(sql_str)
    finally:
        cur.close()

    vectordb = HanaDB(
        connection=config.conn,
        embedding=embedding,
        distance_strategy=DistanceStrategy.COSINE,
        table_name=table_name,
        content_column=content_column,
        metadata_column=metadata_column,
        vector_column=vector_column,
    )

    vectordb.add_texts(texts=HanaTestConstants.TEXTS)

    # check that embeddings have been created in the table
    number_of_texts = len(HanaTestConstants.TEXTS)
    number_of_rows = -1
    sql_str = f"SELECT COUNT(*) FROM {table_name}"
    cur = config.conn.cursor()
    cur.execute(sql_str)
    if cur.has_result_set():
        rows = cur.fetchall()
        number_of_rows = rows[0][0]
    assert number_of_rows == number_of_texts


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_hanavector_table_with_wrong_typed_columns(table_name_with_cleanup) -> None:
    table_name = table_name_with_cleanup
    content_column = "DOC_TEXT"
    metadata_column = "DOC_META"
    vector_column = "DOC_VECTOR"
    try:
        cur = config.conn.cursor()
        sql_str = (
            f"CREATE TABLE {table_name}({content_column} INTEGER, "
            f"{metadata_column} INTEGER, {vector_column} INTEGER);"
        )
        cur.execute(sql_str)
    finally:
        cur.close()

    # Check if table is created
    exception_occured = False
    try:
        HanaDB(
            connection=config.conn,
            embedding=embedding,
            distance_strategy=DistanceStrategy.COSINE,
            table_name=table_name,
        )
        exception_occured = False
    except AttributeError as err:
        print(err)  # noqa: T201
        exception_occured = True
    assert exception_occured


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_hanavector_non_existing_table_fixed_vector_length(table_name_with_cleanup) -> None:
    """Test end to end construction and search."""
    table_name = table_name_with_cleanup
    vector_column = "MY_VECTOR"
    vector_column_length = 42

    # Check if table is created
    vectordb = HanaDB(
        connection=config.conn,
        embedding=embedding,
        distance_strategy=DistanceStrategy.COSINE,
        table_name=table_name,
        vector_column=vector_column,
        vector_column_length=vector_column_length,
    )

    assert vectordb._table_exists(table_name)
    vectordb._check_column(
        table_name, vector_column, "REAL_VECTOR", vector_column_length
    )


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_hanavector_add_texts(vectorDB) -> None:
    vectorDB.add_texts(texts=HanaTestConstants.TEXTS)

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


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_hanavector_from_texts(table_name_with_cleanup) -> None:
    table_name = table_name_with_cleanup
    vectorDB = HanaDB.from_texts(
        connection=config.conn,
        texts=HanaTestConstants.TEXTS,
        embedding=embedding,
        table_name=table_name,
    )

    # test if vectorDB is instance of HanaDB
    assert isinstance(vectorDB, HanaDB)

    # check that embeddings have been created in the table
    number_of_texts = len(HanaTestConstants.TEXTS)
    number_of_rows = -1
    sql_str = f"SELECT COUNT(*) FROM {table_name}"
    cur = config.conn.cursor()
    cur.execute(sql_str)
    if cur.has_result_set():
        rows = cur.fetchall()
        number_of_rows = rows[0][0]
    assert number_of_rows == number_of_texts


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
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


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
@pytest.mark.parametrize("k", [0, -4])
def test_hanavector_similarity_search_simple_invalid(vectorDB, k: int) -> None:
    with pytest.raises(ValueError, match="must be an integer greater than 0"):
        vectorDB.similarity_search(HanaTestConstants.TEXTS[0], k)


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_hanavector_similarity_search_by_vector_simple(vectorDB) -> None:
    vectorDB.add_texts(texts=HanaTestConstants.TEXTS)

    vector = embedding.embed_query(HanaTestConstants.TEXTS[0])
    assert (
        HanaTestConstants.TEXTS[0]
        == vectorDB.similarity_search_by_vector(vector, 1)[0].page_content
    )
    assert (
        HanaTestConstants.TEXTS[1]
        != vectorDB.similarity_search_by_vector(vector, 1)[0].page_content
    )


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
@pytest.mark.parametrize("k", [0, -4])
def test_hanavector_similarity_search_by_vector_simple_invalid(
    vectorDB, k: int
) -> None:
    with pytest.raises(ValueError, match="must be an integer greater than 0"):
        vectorDB.similarity_search_by_vector(HanaTestConstants.TEXTS[0], k)


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_hanavector_similarity_search_simple_euclidean_distance(table_name_with_cleanup) -> None:
    table_name = table_name_with_cleanup

    # Check if table is created
    vectorDB = HanaDB.from_texts(
        connection=config.conn,
        texts=HanaTestConstants.TEXTS,
        embedding=embedding,
        table_name=table_name,
        distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
    )

    assert (
        HanaTestConstants.TEXTS[0]
        == vectorDB.similarity_search(HanaTestConstants.TEXTS[0], 1)[0].page_content
    )
    assert (
        HanaTestConstants.TEXTS[1]
        != vectorDB.similarity_search(HanaTestConstants.TEXTS[0], 1)[0].page_content
    )


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_hanavector_similarity_search_with_metadata(
    vectorDB,
) -> None:
    vectorDB.add_texts(
        texts=HanaTestConstants.TEXTS, metadatas=HanaTestConstants.METADATAS
    )

    search_result = vectorDB.similarity_search(HanaTestConstants.TEXTS[0], 3)

    assert HanaTestConstants.TEXTS[0] == search_result[0].page_content
    assert HanaTestConstants.METADATAS[0]["start"] == search_result[0].metadata["start"]
    assert HanaTestConstants.METADATAS[0]["end"] == search_result[0].metadata["end"]
    assert HanaTestConstants.TEXTS[1] != search_result[0].page_content
    assert HanaTestConstants.METADATAS[1]["start"] != search_result[0].metadata["start"]
    assert HanaTestConstants.METADATAS[1]["end"] != search_result[0].metadata["end"]


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
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


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_hanavector_similarity_search_with_metadata_filter_string(
    vectorDB,
) -> None:
    vectorDB.add_texts(
        texts=HanaTestConstants.TEXTS, metadatas=HanaTestConstants.METADATAS
    )

    search_result = vectorDB.similarity_search(
        HanaTestConstants.TEXTS[0], 3, filter={"quality": "bad"}
    )

    assert len(search_result) == 1
    assert HanaTestConstants.TEXTS[1] == search_result[0].page_content


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_hanavector_similarity_search_with_metadata_filter_bool(
    vectorDB,
) -> None:
    vectorDB.add_texts(
        texts=HanaTestConstants.TEXTS, metadatas=HanaTestConstants.METADATAS
    )

    search_result = vectorDB.similarity_search(
        HanaTestConstants.TEXTS[0], 3, filter={"ready": False}
    )

    assert len(search_result) == 1
    assert HanaTestConstants.TEXTS[1] == search_result[0].page_content


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_hanavector_similarity_search_with_metadata_filter_invalid_type(
    vectorDB,
) -> None:
    vectorDB.add_texts(
        texts=HanaTestConstants.TEXTS, metadatas=HanaTestConstants.METADATAS
    )

    exception_occured = False
    try:
        vectorDB.similarity_search(
            HanaTestConstants.TEXTS[0], 3, filter={"wrong_type": 0.1}
        )
    except ValueError:
        exception_occured = True
    assert exception_occured


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_hanavector_similarity_search_with_score(vectorDB) -> None:
    vectorDB.add_texts(texts=HanaTestConstants.TEXTS)

    search_result = vectorDB.similarity_search_with_score(HanaTestConstants.TEXTS[0], 3)

    assert search_result[0][0].page_content == HanaTestConstants.TEXTS[0]
    assert search_result[0][1] == 1.0
    assert search_result[1][1] <= search_result[0][1]
    assert search_result[2][1] <= search_result[1][1]
    assert search_result[2][1] >= 0.0


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_hanavector_similarity_search_with_relevance_score(vectorDB) -> None:
    vectorDB.add_texts(texts=HanaTestConstants.TEXTS)

    search_result = vectorDB.similarity_search_with_relevance_scores(
        HanaTestConstants.TEXTS[0], 3
    )

    assert search_result[0][0].page_content == HanaTestConstants.TEXTS[0]
    assert search_result[0][1] == 1.0
    assert search_result[1][1] <= search_result[0][1]
    assert search_result[2][1] <= search_result[1][1]
    assert search_result[2][1] >= 0.0


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_hanavector_similarity_search_with_relevance_score_with_euclidian_distance(table_name_with_cleanup) -> (
    None
):
    table_name = table_name_with_cleanup

    # Check if table is created
    vectorDB = HanaDB.from_texts(
        connection=config.conn,
        texts=HanaTestConstants.TEXTS,
        embedding=embedding,
        table_name=table_name,
        distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
    )

    search_result = vectorDB.similarity_search_with_relevance_scores(
        HanaTestConstants.TEXTS[0], 3
    )

    assert search_result[0][0].page_content == HanaTestConstants.TEXTS[0]
    assert search_result[0][1] == 1.0
    assert search_result[1][1] <= search_result[0][1]
    assert search_result[2][1] <= search_result[1][1]
    assert search_result[2][1] >= 0.0


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_hanavector_similarity_search_with_score_with_euclidian_distance(table_name_with_cleanup) -> None:
    table_name = table_name_with_cleanup

    # Check if table is created
    vectorDB = HanaDB.from_texts(
        connection=config.conn,
        texts=HanaTestConstants.TEXTS,
        embedding=embedding,
        table_name=table_name,
        distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
    )

    search_result = vectorDB.similarity_search_with_score(HanaTestConstants.TEXTS[0], 3)

    assert search_result[0][0].page_content == HanaTestConstants.TEXTS[0]
    assert search_result[0][1] == 0.0
    assert search_result[1][1] >= search_result[0][1]
    assert search_result[2][1] >= search_result[1][1]


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_hanavector_delete_with_filter(vectorDB) -> None:
    vectorDB.add_texts(
        texts=HanaTestConstants.TEXTS, metadatas=HanaTestConstants.METADATAS
    )

    search_result = vectorDB.similarity_search(HanaTestConstants.TEXTS[0], 10)
    assert len(search_result) == 5

    # Delete one of the three entries
    assert vectorDB.delete(filter={"start": 100, "end": 200})

    search_result = vectorDB.similarity_search(HanaTestConstants.TEXTS[0], 10)
    assert len(search_result) == 4


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
async def test_hanavector_delete_with_filter_async(
    vectorDB,
) -> None:
    vectorDB.add_texts(
        texts=HanaTestConstants.TEXTS, metadatas=HanaTestConstants.METADATAS
    )

    search_result = vectorDB.similarity_search(HanaTestConstants.TEXTS[0], 10)
    assert len(search_result) == 5

    # Delete one of the three entries
    assert await vectorDB.adelete(filter={"start": 100, "end": 200})

    search_result = vectorDB.similarity_search(HanaTestConstants.TEXTS[0], 10)
    assert len(search_result) == 4


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_hanavector_delete_all_with_empty_filter(
    vectorDB,
) -> None:
    vectorDB.add_texts(
        texts=HanaTestConstants.TEXTS, metadatas=HanaTestConstants.METADATAS
    )

    search_result = vectorDB.similarity_search(HanaTestConstants.TEXTS[0], 3)
    assert len(search_result) == 3

    # Delete all entries
    assert vectorDB.delete(filter={})

    search_result = vectorDB.similarity_search(HanaTestConstants.TEXTS[0], 3)
    assert len(search_result) == 0


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_hanavector_delete_called_wrong(vectorDB) -> None:
    vectorDB.add_texts(
        texts=HanaTestConstants.TEXTS, metadatas=HanaTestConstants.METADATAS
    )

    # Delete without filter parameter
    exception_occured = False
    try:
        vectorDB.delete()
    except ValueError:
        exception_occured = True
    assert exception_occured

    # Delete with ids parameter
    exception_occured = False
    try:
        vectorDB.delete(ids=["id1", "id"], filter={"start": 100, "end": 200})
    except ValueError:
        exception_occured = True
    assert exception_occured


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_hanavector_max_marginal_relevance_search(vectorDB) -> None:
    vectorDB.add_texts(texts=HanaTestConstants.TEXTS)

    search_result = vectorDB.max_marginal_relevance_search(
        HanaTestConstants.TEXTS[0], k=2, fetch_k=20
    )

    assert len(search_result) == 2
    assert search_result[0].page_content == HanaTestConstants.TEXTS[0]
    assert search_result[1].page_content != HanaTestConstants.TEXTS[0]


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
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


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_hanavector_max_marginal_relevance_search_vector(vectorDB) -> None:
    vectorDB.add_texts(texts=HanaTestConstants.TEXTS)

    search_result = vectorDB.max_marginal_relevance_search_by_vector(
        embedding.embed_query(HanaTestConstants.TEXTS[0]), k=2, fetch_k=20
    )

    assert len(search_result) == 2
    assert search_result[0].page_content == HanaTestConstants.TEXTS[0]
    assert search_result[1].page_content != HanaTestConstants.TEXTS[0]


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
async def test_hanavector_max_marginal_relevance_search_async(
    vectorDB,
) -> None:
    vectorDB.add_texts(texts=HanaTestConstants.TEXTS)

    search_result = await vectorDB.amax_marginal_relevance_search(
        HanaTestConstants.TEXTS[0], k=2, fetch_k=20
    )

    assert len(search_result) == 2
    assert search_result[0].page_content == HanaTestConstants.TEXTS[0]
    assert search_result[1].page_content != HanaTestConstants.TEXTS[0]


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
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


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_hanavector_filter_prepared_statement_params(
    vectorDB,
) -> None:
    vectorDB.add_texts(
        texts=HanaTestConstants.TEXTS, metadatas=HanaTestConstants.METADATAS
    )

    cur = config.conn.cursor()
    sql_str = f"SELECT * FROM {HanaTestConstants.TABLE_NAME} WHERE JSON_VALUE(VEC_META, '$.start') = '100'"
    cur.execute(sql_str)
    rows = cur.fetchall()
    assert len(rows) == 1

    query_value = 100
    sql_str = f"SELECT * FROM {HanaTestConstants.TABLE_NAME} WHERE JSON_VALUE(VEC_META, '$.start') = ?"
    cur.execute(sql_str, (query_value))
    rows = cur.fetchall()
    assert len(rows) == 1

    sql_str = f"SELECT * FROM {HanaTestConstants.TABLE_NAME} WHERE JSON_VALUE(VEC_META, '$.quality') = 'good'"
    cur.execute(sql_str)
    rows = cur.fetchall()
    assert len(rows) == 1

    query_value = "good"  # type: ignore[assignment]
    sql_str = f"SELECT * FROM {HanaTestConstants.TABLE_NAME} WHERE JSON_VALUE(VEC_META, '$.quality') = ?"
    cur.execute(sql_str, (query_value))
    rows = cur.fetchall()
    assert len(rows) == 1

    sql_str = f"SELECT * FROM {HanaTestConstants.TABLE_NAME} WHERE JSON_VALUE(VEC_META, '$.ready') = false"
    cur.execute(sql_str)
    rows = cur.fetchall()
    assert len(rows) == 1

    # query_value = True
    query_value = "true"  # type: ignore[assignment]
    sql_str = f"SELECT * FROM {HanaTestConstants.TABLE_NAME} WHERE JSON_VALUE(VEC_META, '$.ready') = ?"
    cur.execute(sql_str, (query_value))
    rows = cur.fetchall()
    assert len(rows) == 3

    # query_value = False
    query_value = "false"  # type: ignore[assignment]
    sql_str = f"SELECT * FROM {HanaTestConstants.TABLE_NAME} WHERE JSON_VALUE(VEC_META, '$.ready') = ?"
    cur.execute(sql_str, (query_value))
    rows = cur.fetchall()
    assert len(rows) == 1


def test_invalid_metadata_keys(table_name_with_cleanup) -> None:
    table_name = table_name_with_cleanup

    invalid_metadatas = [
        {"sta rt": 0, "end": 100, "quality": "good", "ready": True},
    ]
    exception_occured = False
    try:
        HanaDB.from_texts(
            connection=config.conn,
            texts=HanaTestConstants.TEXTS,
            metadatas=invalid_metadatas,
            embedding=embedding,
            table_name=table_name,
        )
    except ValueError:
        exception_occured = True
    assert exception_occured

    invalid_metadatas = [
        {"sta/nrt": 0, "end": 100, "quality": "good", "ready": True},
    ]
    exception_occured = False
    try:
        HanaDB.from_texts(
            connection=config.conn,
            texts=HanaTestConstants.TEXTS,
            metadatas=invalid_metadatas,
            embedding=embedding,
            table_name=table_name,
        )
    except ValueError:
        exception_occured = True
    assert exception_occured


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_hanavector_table_mixed_case_names() -> None:
    table_name = "MyTableName"
    content_column = "TextColumn"
    metadata_column = "MetaColumn"
    vector_column = "VectorColumn"
    
    try:
        vectordb = HanaDB(
            connection=config.conn,
            embedding=embedding,
            distance_strategy=DistanceStrategy.COSINE,
            table_name=table_name,
            content_column=content_column,
            metadata_column=metadata_column,
            vector_column=vector_column,
        )

        vectordb.add_texts(texts=HanaTestConstants.TEXTS)

        # check that embeddings have been created in the table
        number_of_texts = len(HanaTestConstants.TEXTS)
        number_of_rows = -1
        sql_str = f'SELECT COUNT(*) FROM "{table_name}"'
        cur = config.conn.cursor()
        cur.execute(sql_str)
        if cur.has_result_set():
            rows = cur.fetchall()
            number_of_rows = rows[0][0]
        assert number_of_rows == number_of_texts

        # check results of similarity search
        assert (
            HanaTestConstants.TEXTS[0]
            == vectordb.similarity_search(HanaTestConstants.TEXTS[0], 1)[0].page_content
        )
    
    finally:
        HanaTestUtils.drop_table(config.conn, table_name)


@pytest.mark.parametrize(
    "test_filter, expected_ids, expected_where_clause, expected_where_clause_parameters",
    FILTERING_TEST_CASES,
)
@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_hanavector_with_with_metadata_filters(
    test_filter: Dict[str, Any],
    expected_ids: List[int],
    expected_where_clause: str,
    expected_where_clause_parameters: List[Any],
    vectorDB,  # Fixture
) -> None:
    # Delete already existing documents from the table
    vectorDB.delete(filter={})

    vectorDB.add_documents(DOCUMENTS)

    docs = vectorDB.similarity_search("meow", k=5, filter=test_filter)
    ids = [doc.metadata["id"] for doc in docs]
    assert sorted(ids) == sorted(expected_ids)


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_preexisting_specific_columns_for_metadata_fill(table_name_with_cleanup) -> None:
    table_name = table_name_with_cleanup

    sql_str = (
        f'CREATE TABLE "{table_name}" ('
        f'"VEC_TEXT" NCLOB, '
        f'"VEC_META" NCLOB, '
        f'"VEC_VECTOR" REAL_VECTOR, '
        f'"Owner" NVARCHAR(100), '
        f'"quality" NVARCHAR(100));'
    )
    try:
        cur = config.conn.cursor()
        cur.execute(sql_str)
    finally:
        cur.close()

    vectorDB = HanaDB.from_texts(
        connection=config.conn,
        texts=HanaTestConstants.TEXTS,
        metadatas=HanaTestConstants.METADATAS,
        embedding=embedding,
        table_name=table_name,
        specific_metadata_columns=["Owner", "quality"],
    )

    c = 0
    try:
        sql_str = f'SELECT COUNT(*) FROM {table_name} WHERE "quality"=' f"'ugly'"
        cur = config.conn.cursor()
        cur.execute(sql_str)
        if cur.has_result_set():
            rows = cur.fetchall()
            c = rows[0][0]
    finally:
        cur.close()
    assert c == 3

    docs = vectorDB.similarity_search("hello", k=5, filter={"quality": "good"})
    assert len(docs) == 1
    assert docs[0].page_content == "foo"

    docs = vectorDB.similarity_search("hello", k=5, filter={"start": 100})
    assert len(docs) == 1
    assert docs[0].page_content == "bar"

    docs = vectorDB.similarity_search(
        "hello", k=5, filter={"start": 100, "quality": "good"}
    )
    assert len(docs) == 0

    docs = vectorDB.similarity_search(
        "hello", k=5, filter={"start": 0, "quality": "good"}
    )
    assert len(docs) == 1
    assert docs[0].page_content == "foo"


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_preexisting_specific_columns_for_metadata_via_array(table_name_with_cleanup) -> None:
    table_name = table_name_with_cleanup

    sql_str = (
        f'CREATE TABLE "{table_name}" ('
        f'"VEC_TEXT" NCLOB, '
        f'"VEC_META" NCLOB, '
        f'"VEC_VECTOR" REAL_VECTOR, '
        f'"Owner" NVARCHAR(100), '
        f'"quality" NVARCHAR(100));'
    )
    try:
        cur = config.conn.cursor()
        cur.execute(sql_str)
    finally:
        cur.close()

    vectorDB = HanaDB.from_texts(
        connection=config.conn,
        texts=HanaTestConstants.TEXTS,
        metadatas=HanaTestConstants.METADATAS,
        embedding=embedding,
        table_name=table_name,
        specific_metadata_columns=["quality"],
    )

    c = 0
    try:
        sql_str = f'SELECT COUNT(*) FROM {table_name} WHERE "quality"=' f"'ugly'"
        cur = config.conn.cursor()
        cur.execute(sql_str)
        if cur.has_result_set():
            rows = cur.fetchall()
            c = rows[0][0]
    finally:
        cur.close()
    assert c == 3

    try:
        sql_str = f'SELECT COUNT(*) FROM {table_name} WHERE "Owner"=' f"'Steve'"
        cur = config.conn.cursor()
        cur.execute(sql_str)
        if cur.has_result_set():
            rows = cur.fetchall()
            c = rows[0][0]
    finally:
        cur.close()
    assert c == 0

    docs = vectorDB.similarity_search("hello", k=5, filter={"quality": "good"})
    assert len(docs) == 1
    assert docs[0].page_content == "foo"

    docs = vectorDB.similarity_search("hello", k=5, filter={"start": 100})
    assert len(docs) == 1
    assert docs[0].page_content == "bar"

    docs = vectorDB.similarity_search(
        "hello", k=5, filter={"start": 100, "quality": "good"}
    )
    assert len(docs) == 0

    docs = vectorDB.similarity_search(
        "hello", k=5, filter={"start": 0, "quality": "good"}
    )
    assert len(docs) == 1
    assert docs[0].page_content == "foo"


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_preexisting_specific_columns_for_metadata_multiple_columns(table_name_with_cleanup) -> None:
    table_name = table_name_with_cleanup

    sql_str = (
        f'CREATE TABLE "{table_name}" ('
        f'"VEC_TEXT" NCLOB, '
        f'"VEC_META" NCLOB, '
        f'"VEC_VECTOR" REAL_VECTOR, '
        f'"quality" NVARCHAR(100), '
        f'"start" INTEGER);'
    )
    try:
        cur = config.conn.cursor()
        cur.execute(sql_str)
    finally:
        cur.close()

    vectorDB = HanaDB.from_texts(
        connection=config.conn,
        texts=HanaTestConstants.TEXTS,
        metadatas=HanaTestConstants.METADATAS,
        embedding=embedding,
        table_name=table_name,
        specific_metadata_columns=["quality", "start"],
    )

    docs = vectorDB.similarity_search("hello", k=5, filter={"quality": "good"})
    assert len(docs) == 1
    assert docs[0].page_content == "foo"

    docs = vectorDB.similarity_search("hello", k=5, filter={"start": 100})
    assert len(docs) == 1
    assert docs[0].page_content == "bar"

    docs = vectorDB.similarity_search(
        "hello", k=5, filter={"start": 100, "quality": "good"}
    )
    assert len(docs) == 0

    docs = vectorDB.similarity_search(
        "hello", k=5, filter={"start": 0, "quality": "good"}
    )
    assert len(docs) == 1
    assert docs[0].page_content == "foo"


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_preexisting_specific_columns_for_metadata_empty_columns(table_name_with_cleanup) -> None:
    table_name = table_name_with_cleanup

    sql_str = (
        f'CREATE TABLE "{table_name}" ('
        f'"VEC_TEXT" NCLOB, '
        f'"VEC_META" NCLOB, '
        f'"VEC_VECTOR" REAL_VECTOR, '
        f'"quality" NVARCHAR(100), '
        f'"ready" BOOLEAN, '
        f'"start" INTEGER);'
    )
    try:
        cur = config.conn.cursor()
        cur.execute(sql_str)
    finally:
        cur.close()

    vectorDB = HanaDB.from_texts(
        connection=config.conn,
        texts=HanaTestConstants.TEXTS,
        metadatas=HanaTestConstants.METADATAS,
        embedding=embedding,
        table_name=table_name,
        specific_metadata_columns=["quality", "ready", "start"],
    )

    docs = vectorDB.similarity_search("hello", k=5, filter={"quality": "good"})
    assert len(docs) == 1
    assert docs[0].page_content == "foo"

    docs = vectorDB.similarity_search("hello", k=5, filter={"start": 100})
    assert len(docs) == 1
    assert docs[0].page_content == "bar"

    docs = vectorDB.similarity_search(
        "hello", k=5, filter={"start": 100, "quality": "good"}
    )
    assert len(docs) == 0

    docs = vectorDB.similarity_search(
        "hello", k=5, filter={"start": 0, "quality": "good"}
    )
    assert len(docs) == 1
    assert docs[0].page_content == "foo"

    docs = vectorDB.similarity_search("hello", k=5, filter={"ready": True})
    assert len(docs) == 3


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_preexisting_specific_columns_for_metadata_wrong_type_or_non_existing(table_name_with_cleanup) -> None:
    table_name = table_name_with_cleanup

    sql_str = (
        f'CREATE TABLE "{table_name}" ('
        f'"VEC_TEXT" NCLOB, '
        f'"VEC_META" NCLOB, '
        f'"VEC_VECTOR" REAL_VECTOR, '
        f'"quality" INTEGER); '
    )
    try:
        cur = config.conn.cursor()
        cur.execute(sql_str)
    finally:
        cur.close()

    # Check if table is created
    exception_occured = False
    try:
        HanaDB.from_texts(
            connection=config.conn,
            texts=HanaTestConstants.TEXTS,
            metadatas=HanaTestConstants.METADATAS,
            embedding=embedding,
            table_name=table_name,
            specific_metadata_columns=["quality"],
        )
        exception_occured = False
    except dbapi.Error:  # Nothing we should do here, hdbcli will throw an error
        exception_occured = True
    assert exception_occured  # Check if table is created

    exception_occured = False
    try:
        HanaDB.from_texts(
            connection=config.conn,
            texts=HanaTestConstants.TEXTS,
            metadatas=HanaTestConstants.METADATAS,
            embedding=embedding,
            table_name=table_name,
            specific_metadata_columns=["NonExistingColumn"],
        )
        exception_occured = False
    except AttributeError:  # Nothing we should do here, hdbcli will throw an error
        exception_occured = True
    assert exception_occured


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_preexisting_specific_columns_for_returned_metadata_completeness(table_name_with_cleanup) -> None:
    table_name = table_name_with_cleanup

    sql_str = (
        f'CREATE TABLE "{table_name}" ('
        f'"VEC_TEXT" NCLOB, '
        f'"VEC_META" NCLOB, '
        f'"VEC_VECTOR" REAL_VECTOR, '
        f'"quality" NVARCHAR(100), '
        f'"NonExisting" NVARCHAR(100), '
        f'"ready" BOOLEAN, '
        f'"start" INTEGER);'
    )
    try:
        cur = config.conn.cursor()
        cur.execute(sql_str)
    finally:
        cur.close()

    vectorDB = HanaDB.from_texts(
        connection=config.conn,
        texts=HanaTestConstants.TEXTS,
        metadatas=HanaTestConstants.METADATAS,
        embedding=embedding,
        table_name=table_name,
        specific_metadata_columns=["quality", "ready", "start", "NonExisting"],
    )

    docs = vectorDB.similarity_search("hello", k=5, filter={"quality": "good"})
    assert len(docs) == 1
    assert docs[0].page_content == "foo"
    assert docs[0].metadata["end"] == 100
    assert docs[0].metadata["start"] == 0
    assert docs[0].metadata["quality"] == "good"
    assert docs[0].metadata["ready"]
    assert "NonExisting" not in docs[0].metadata.keys()


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_create_hnsw_index_with_default_values(vectorDB) -> None:
    vectorDB.add_texts(texts=HanaTestConstants.TEXTS)

    # Test the creation of HNSW index
    try:
        vectorDB.create_hnsw_index()
    except Exception as e:
        pytest.fail(f"Failed to create HNSW index: {e}")

    # Perform a search using the index to confirm its correctness
    search_result = vectorDB.max_marginal_relevance_search(
        HanaTestConstants.TEXTS[0], k=2, fetch_k=20
    )

    assert len(search_result) == 2
    assert search_result[0].page_content == HanaTestConstants.TEXTS[0]
    assert search_result[1].page_content != HanaTestConstants.TEXTS[0]


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_create_hnsw_index_with_defined_values(table_name_with_cleanup) -> None:
    table_name = table_name_with_cleanup

    # Create table and insert data
    vectorDB = HanaDB.from_texts(
        connection=config.conn,
        texts=HanaTestConstants.TEXTS,
        embedding=embedding,
        table_name=table_name,
        distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
    )

    # Test the creation of HNSW index with specific values
    try:
        vectorDB.create_hnsw_index(
            index_name="my_L2_index", ef_search=500, m=100, ef_construction=200
        )
    except Exception as e:
        pytest.fail(f"Failed to create HNSW index with defined values: {e}")

    # Perform a search using the index to confirm its correctness
    search_result = vectorDB.max_marginal_relevance_search(
        HanaTestConstants.TEXTS[0], k=2, fetch_k=20
    )

    assert len(search_result) == 2
    assert search_result[0].page_content == HanaTestConstants.TEXTS[0]
    assert search_result[1].page_content != HanaTestConstants.TEXTS[0]


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_create_hnsw_index_after_initialization(vectorDB) -> None:
    # Create HNSW index before adding documents
    vectorDB.create_hnsw_index(
        index_name="index_pre_add", ef_search=400, m=50, ef_construction=150
    )

    # Add texts after index creation
    vectorDB.add_texts(texts=HanaTestConstants.TEXTS)

    # Perform similarity search using the index
    search_result = vectorDB.similarity_search(HanaTestConstants.TEXTS[0], k=3)

    # Assert that search result is valid and has expected length
    assert len(search_result) == 3
    assert search_result[0].page_content == HanaTestConstants.TEXTS[0]
    assert search_result[1].page_content != HanaTestConstants.TEXTS[0]


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_duplicate_hnsw_index_creation(vectorDB) -> None:
    vectorDB.add_texts(texts=HanaTestConstants.TEXTS)

    # Create HNSW index for the first time
    vectorDB.create_hnsw_index(
        index_name="index_cosine",
        ef_search=300,
        m=80,
        ef_construction=100,
    )

    with pytest.raises(Exception):
        vectorDB.create_hnsw_index(ef_search=300, m=80, ef_construction=100)


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_create_hnsw_index_invalid_m_value(vectorDB) -> None:
    vectorDB.add_texts(texts=HanaTestConstants.TEXTS)

    # Test invalid `m` value (too low)
    with pytest.raises(ValueError):
        vectorDB.create_hnsw_index(m=3)

    # Test invalid `m` value (too high)
    with pytest.raises(ValueError):
        vectorDB.create_hnsw_index(m=1001)


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_create_hnsw_index_invalid_ef_construction(vectorDB) -> None:
    vectorDB.add_texts(texts=HanaTestConstants.TEXTS)

    # Test invalid `ef_construction` value (too low)
    with pytest.raises(ValueError):
        vectorDB.create_hnsw_index(ef_construction=0)

    # Test invalid `ef_construction` value (too high)
    with pytest.raises(ValueError):
        vectorDB.create_hnsw_index(ef_construction=100001)


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_create_hnsw_index_invalid_ef_search(vectorDB) -> None:
    vectorDB.add_texts(texts=HanaTestConstants.TEXTS)

    # Test invalid `ef_search` value (too low)
    with pytest.raises(ValueError):
        vectorDB.create_hnsw_index(ef_search=0)

    # Test invalid `ef_search` value (too high)
    with pytest.raises(ValueError):
        vectorDB.create_hnsw_index(ef_search=100001)


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_hanavector_keyword_search(table_name_with_cleanup) -> None:
    table_name = table_name_with_cleanup

    sql_str = (
        f'CREATE TABLE "{table_name}" ('
        f'"VEC_TEXT" NCLOB, '
        f'"VEC_META" NCLOB, '
        f'"VEC_VECTOR" REAL_VECTOR, '
        f'"quality" NVARCHAR(100), '
        f'"start" INTEGER);'
    )

    try:
        cur = config.conn.cursor()
        cur.execute(sql_str)
    finally:
        cur.close()

    vectorDB = HanaDB.from_texts(
        connection=config.conn,
        texts=HanaTestConstants.TEXTS,
        metadatas=HanaTestConstants.METADATAS,
        embedding=embedding,
        table_name=table_name,
        specific_metadata_columns=["quality"],
    )

    # Perform keyword search on content column
    keyword = "foo"
    docs = vectorDB.similarity_search(
        query=keyword, k=3, filter={"VEC_TEXT": {"$contains": keyword}}
    )

    # Validate the results
    assert len(docs) == 1
    assert keyword in docs[0].page_content

    # Perform keyword search with non-existing keyword
    non_existing_keyword = "nonexistent"
    docs = vectorDB.similarity_search(
        query=non_existing_keyword,
        k=3,
        filter={"VEC_TEXT": {"$contains": non_existing_keyword}},
    )

    # Validate the results
    assert len(docs) == 0, "Expected no results for non-existing keyword"

    # Perform keyword search on specific metadata column
    keyword = "good"
    docs = vectorDB.similarity_search(
        query=keyword, k=3, filter={"quality": {"$contains": keyword}}
    )

    # Validate the results
    assert len(docs) == 1
    assert keyword in docs[0].metadata["quality"]

    # Perform keyword search with non-existing keyword
    non_existing_keyword = "terrible"
    docs = vectorDB.similarity_search(
        query=non_existing_keyword,
        k=3,
        filter={"quality": {"$contains": non_existing_keyword}},
    )

    # Validate the results
    assert len(docs) == 0, "Expected no results for non-existing keyword"


@pytest.mark.skipif(not hanadb_installed, reason="hanadb not installed")
def test_hanavector_keyword_search_unspecific_metadata_column(
    vectorDB,
) -> None:
    vectorDB.add_texts(
        texts=HanaTestConstants.TEXTS, metadatas=HanaTestConstants.METADATAS
    )

    keyword = "good"

    docs = vectorDB.similarity_search("hello", k=5, filter={"quality": keyword})
    assert len(docs) == 1
    assert "foo" in docs[0].page_content

    # Perform keyword search on unspecific metadata column
    docs = vectorDB.similarity_search(
        "hello", k=5, filter={"quality": {"$contains": keyword}}
    )
    assert len(docs) == 1
    assert "foo" in docs[0].page_content
    assert "good" in docs[0].metadata["quality"]

    # Perform keyword search with non-existing keyword on unspecific metadata column
    non_existing_keyword = "terrible"
    docs = vectorDB.similarity_search(
        query=non_existing_keyword,
        k=3,
        filter={"quality": {"$contains": non_existing_keyword}},
    )

    # Validate the results
    assert len(docs) == 0, "Expected no results for non-existing keyword"

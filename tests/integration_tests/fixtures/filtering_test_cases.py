"""Module contains test cases for testing filtering of documents in vector stores."""

from langchain_core.documents import Document

metadatas = [
    {
        "name": "adam",
        "date": "2021-01-01",
        "count": 1,
        "is_active": True,
        "tags": ["a", "b"],
        "location": [1.0, 2.0],
        "id": 1,
        "height": 10.0,  # Float column
        "happiness": 0.9,  # Float column
        "sadness": 0.1,  # Float column
    },
    {
        "name": "bob",
        "date": "2021-01-02",
        "count": 2,
        "is_active": False,
        "tags": ["b", "c"],
        "location": [2.0, 3.0],
        "id": 2,
        "height": 5.7,  # Float column
        "happiness": 0.8,  # Float column
        "sadness": 0.1,  # Float column
    },
    {
        "name": "jane",
        "date": "2021-01-01",
        "count": 3,
        "is_active": True,
        "tags": ["b", "d"],
        "location": [3.0, 4.0],
        "id": 3,
        "height": 2.4,  # Float column
        "happiness": None,
        # Sadness missing intentionally
    },
]
texts = ["id {id}".format(id=metadata["id"]) for metadata in metadatas]

DOCUMENTS = [
    Document(page_content=text, metadata=metadata)
    for text, metadata in zip(texts, metadatas)
]


TYPE_1_FILTERING_TEST_CASES = [
    # These tests only involve equality checks
    (
        {"id": 1},
        [1],
        "WHERE JSON_VALUE(VEC_META, '$.id') = TO_DOUBLE(?)",
        [1],
    ),
    # String field
    (
        {"name": "adam"},
        [1],
        "WHERE JSON_VALUE(VEC_META, '$.name') = ?",
        ["adam"],
    ),
    # Boolean fields
    (
        {"is_active": True},
        [1, 3],
        "WHERE JSON_VALUE(VEC_META, '$.is_active') = TO_BOOLEAN(?)",
        ["true"],
    ),
    (
        {"is_active": False},
        [2],
        "WHERE JSON_VALUE(VEC_META, '$.is_active') = TO_BOOLEAN(?)",
        ["false"],
    ),
    # And semantics for top level filtering
    (
        {"id": 1, "is_active": True},
        [1],
        "WHERE (JSON_VALUE(VEC_META, '$.id') = TO_DOUBLE(?)) AND (JSON_VALUE(VEC_META, '$.is_active') = TO_BOOLEAN(?))",
        [1, "true"],
    ),
    (
        {"id": 1, "is_active": False},
        [],
        "WHERE (JSON_VALUE(VEC_META, '$.id') = TO_DOUBLE(?)) AND (JSON_VALUE(VEC_META, '$.is_active') = TO_BOOLEAN(?))",
        [1, "false"],
    ),
]

TYPE_2_FILTERING_TEST_CASES = [
    # These involve equality checks and other operators
    # like $ne, $gt, $gte, $lt, $lte, $not
    (
        {"id": 1},
        [1],
        "WHERE JSON_VALUE(VEC_META, '$.id') = TO_DOUBLE(?)",
        [1],
    ),
    (
        {"id": {"$ne": 1}},
        [2, 3],
        "WHERE JSON_VALUE(VEC_META, '$.id') <> TO_DOUBLE(?)",
        [1],
    ),
    (
        {"id": {"$gt": 1}},
        [2, 3],
        "WHERE JSON_VALUE(VEC_META, '$.id') > TO_DOUBLE(?)",
        [1],
    ),
    (
        {"id": {"$gte": 1}},
        [1, 2, 3],
        "WHERE JSON_VALUE(VEC_META, '$.id') >= TO_DOUBLE(?)",
        [1],
    ),
    (
        {"id": {"$lt": 1}},
        [],
        "WHERE JSON_VALUE(VEC_META, '$.id') < TO_DOUBLE(?)",
        [1],
    ),
    (
        {"id": {"$lte": 1}},
        [1],
        "WHERE JSON_VALUE(VEC_META, '$.id') <= TO_DOUBLE(?)",
        [1],
    ),
    # Repeat all the same tests with name (string column)
    (
        {"name": "adam"},
        [1],
        "WHERE JSON_VALUE(VEC_META, '$.name') = ?",
        ["adam"],
    ),
    (
        {"name": "bob"},
        [2],
        "WHERE JSON_VALUE(VEC_META, '$.name') = ?",
        ["bob"],
    ),
    (
        {"name": {"$eq": "adam"}},
        [1],
        "WHERE JSON_VALUE(VEC_META, '$.name') = ?",
        ["adam"],
    ),
    (
        {"name": {"$ne": "adam"}},
        [2, 3],
        "WHERE JSON_VALUE(VEC_META, '$.name') <> ?",
        ["adam"],
    ),
    # And also gt, gte, lt, lte relying on lexicographical ordering
    (
        {"name": {"$gt": "jane"}},
        [],
        "WHERE JSON_VALUE(VEC_META, '$.name') > ?",
        ["jane"],
    ),
    (
        {"name": {"$gte": "jane"}},
        [3],
        "WHERE JSON_VALUE(VEC_META, '$.name') >= ?",
        ["jane"],
    ),
    (
        {"name": {"$lt": "jane"}},
        [1, 2],
        "WHERE JSON_VALUE(VEC_META, '$.name') < ?",
        ["jane"],
    ),
    (
        {"name": {"$lte": "jane"}},
        [1, 2, 3],
        "WHERE JSON_VALUE(VEC_META, '$.name') <= ?",
        ["jane"],
    ),
    (
        {"is_active": {"$eq": True}},
        [1, 3],
        "WHERE JSON_VALUE(VEC_META, '$.is_active') = TO_BOOLEAN(?)",
        ["true"],
    ),
    (
        {"is_active": {"$eq": False}},
        [2],
        "WHERE JSON_VALUE(VEC_META, '$.is_active') = TO_BOOLEAN(?)",
        ["false"],
    ),
    (
        {"is_active": {"$ne": True}},
        [2],
        "WHERE JSON_VALUE(VEC_META, '$.is_active') <> TO_BOOLEAN(?)",
        ["true"],
    ),
    # Test float column.
    (
        {"height": {"$gt": 5.0}},
        [1, 2],
        "WHERE JSON_VALUE(VEC_META, '$.height') > TO_DOUBLE(?)",
        [5.0],
    ),
    (
        {"height": {"$gte": 5.0}},
        [1, 2],
        "WHERE JSON_VALUE(VEC_META, '$.height') >= TO_DOUBLE(?)",
        [5.0],
    ),
    (
        {"height": {"$lt": 5.0}},
        [3],
        "WHERE JSON_VALUE(VEC_META, '$.height') < TO_DOUBLE(?)",
        [5.0],
    ),
    (
        {"height": {"$lte": 5.8}},
        [2, 3],
        "WHERE JSON_VALUE(VEC_META, '$.height') <= TO_DOUBLE(?)",
        [5.8],
    ),
]

TYPE_3_FILTERING_TEST_CASES = [
    # These involve usage of AND and OR operators
    (
        {"$or": [{"id": 1}, {"id": 2}]},
        [1, 2],
        "WHERE (JSON_VALUE(VEC_META, '$.id') = TO_DOUBLE(?)) OR (JSON_VALUE(VEC_META, '$.id') = TO_DOUBLE(?))",
        [1, 2],
    ),
    (
        {"$or": [{"id": 1}, {"name": "bob"}]},
        [1, 2],
        "WHERE (JSON_VALUE(VEC_META, '$.id') = TO_DOUBLE(?)) OR (JSON_VALUE(VEC_META, '$.name') = ?)",
        [1, "bob"],
    ),
    (
        {"$and": [{"id": 1}, {"id": 2}]},
        [],
        "WHERE (JSON_VALUE(VEC_META, '$.id') = TO_DOUBLE(?)) AND (JSON_VALUE(VEC_META, '$.id') = TO_DOUBLE(?))",
        [1, 2],
    ),
    (
        {"$or": [{"id": 1}, {"id": 2}, {"id": 3}]},
        [1, 2, 3],
        "WHERE (JSON_VALUE(VEC_META, '$.id') = TO_DOUBLE(?)) OR (JSON_VALUE(VEC_META, '$.id') = TO_DOUBLE(?)) OR (JSON_VALUE(VEC_META, '$.id') = TO_DOUBLE(?))",
        [1, 2, 3],
    ),
]

TYPE_4_FILTERING_TEST_CASES = [
    # These involve special operators like $in, $nin, $between
    # Test between
    (
        {"id": {"$between": (1, 2)}},
        [1, 2],
        "WHERE JSON_VALUE(VEC_META, '$.id') BETWEEN TO_DOUBLE(?) AND TO_DOUBLE(?)",
        [1, 2],
    ),
    (
        {"id": {"$between": (1, 1)}},
        [1],
        "WHERE JSON_VALUE(VEC_META, '$.id') BETWEEN TO_DOUBLE(?) AND TO_DOUBLE(?)",
        [1, 1],
    ),
    (
        {"name": {"$in": ["adam", "bob"]}},
        [1, 2],
        "WHERE JSON_VALUE(VEC_META, '$.name') IN (?, ?)",
        ["adam", "bob"],
    ),
]

TYPE_4B_FILTERING_TEST_CASES = [
    # Test $nin, which is missing in TYPE_4_FILTERING_TEST_CASES
    (
        {"name": {"$nin": ["adam", "bob"]}},
        [3],
        "WHERE JSON_VALUE(VEC_META, '$.name') NOT IN (?, ?)",
        ["adam", "bob"],
    ),
]

TYPE_5_FILTERING_TEST_CASES = [
    # These involve special operators like $like, $ilike that
    # may be specified to certain databases.
    (
        {"name": {"$like": "a%"}},
        [1],
        "WHERE JSON_VALUE(VEC_META, '$.name') LIKE ?",
        ["a%"],
    ),
    (
        {"name": {"$like": "%a%"}},  # adam and jane
        [1, 3],
        "WHERE JSON_VALUE(VEC_META, '$.name') LIKE ?",
        ["%a%"],
    ),
]

FILTERING_TEST_CASES = [
    *TYPE_1_FILTERING_TEST_CASES,
    *TYPE_2_FILTERING_TEST_CASES,
    *TYPE_3_FILTERING_TEST_CASES,
    *TYPE_4_FILTERING_TEST_CASES,
    *TYPE_4B_FILTERING_TEST_CASES,
    *TYPE_5_FILTERING_TEST_CASES,
]

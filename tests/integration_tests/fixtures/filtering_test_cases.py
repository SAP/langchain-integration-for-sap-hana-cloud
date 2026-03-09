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
    # NULL filtering check
    (
        {"happiness": None},
        [3],
        "WHERE JSON_VALUE(VEC_META, '$.happiness') IS NULL",
        [],
    ),
    # String field
    (
        {"name": "adam"},
        [1],
        "WHERE JSON_VALUE(VEC_META, '$.name') = TO_NVARCHAR(?)",
        ["adam"],
    ),
    # String field (empty string) Issue #66
    (
        {"name": ""},
        [],
        "WHERE JSON_VALUE(VEC_META, '$.name') = TO_NVARCHAR(?)",
        [""],
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
    # like $ne, $gt, $gte, $lt, $lte
    (
        {"id": 1},
        [1],
        "WHERE JSON_VALUE(VEC_META, '$.id') = TO_DOUBLE(?)",
        [1],
    ),
    # $eq with None value
    (
        {"happiness": {"$eq": None}},
        [3],
        "WHERE JSON_VALUE(VEC_META, '$.happiness') IS NULL",
        [],
    ),
    (
        {"id": {"$ne": 1}},
        [2, 3],
        "WHERE JSON_VALUE(VEC_META, '$.id') <> TO_DOUBLE(?)",
        [1],
    ),
    # $ne with None value
    (
        {"sadness": {"$ne": None}},
        [1, 2],
        "WHERE JSON_VALUE(VEC_META, '$.sadness') IS NOT NULL",
        [],
    ),
    (
        {"id": {"$gt": 0}},
        [1, 2, 3],
        "WHERE JSON_VALUE(VEC_META, '$.id') > TO_DOUBLE(?)",
        [0],
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
        "WHERE JSON_VALUE(VEC_META, '$.name') = TO_NVARCHAR(?)",
        ["adam"],
    ),
    (
        {"name": "bob"},
        [2],
        "WHERE JSON_VALUE(VEC_META, '$.name') = TO_NVARCHAR(?)",
        ["bob"],
    ),
    (
        {"name": {"$eq": "adam"}},
        [1],
        "WHERE JSON_VALUE(VEC_META, '$.name') = TO_NVARCHAR(?)",
        ["adam"],
    ),
    (
        {"name": {"$ne": "adam"}},
        [2, 3],
        "WHERE JSON_VALUE(VEC_META, '$.name') <> TO_NVARCHAR(?)",
        ["adam"],
    ),
    # And also gt, gte, lt, lte relying on lexicographical ordering
    (
        {"name": {"$gt": "jane"}},
        [],
        "WHERE JSON_VALUE(VEC_META, '$.name') > TO_NVARCHAR(?)",
        ["jane"],
    ),
    (
        {"name": {"$gte": "jane"}},
        [3],
        "WHERE JSON_VALUE(VEC_META, '$.name') >= TO_NVARCHAR(?)",
        ["jane"],
    ),
    (
        {"name": {"$lt": "jane"}},
        [1, 2],
        "WHERE JSON_VALUE(VEC_META, '$.name') < TO_NVARCHAR(?)",
        ["jane"],
    ),
    (
        {"name": {"$lte": "jane"}},
        [1, 2, 3],
        "WHERE JSON_VALUE(VEC_META, '$.name') <= TO_NVARCHAR(?)",
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
        {"height": 5.7},
        [2],
        "WHERE JSON_VALUE(VEC_META, '$.height') = TO_DOUBLE(?)",
        [5.7],
    ),
    (
        {"height": {"$gt": 0.0}},
        [1, 2, 3],
        "WHERE JSON_VALUE(VEC_META, '$.height') > TO_DOUBLE(?)",
        [0.0],
    ),
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
        "WHERE (JSON_VALUE(VEC_META, '$.id') = TO_DOUBLE(?)) OR (JSON_VALUE(VEC_META, '$.name') = TO_NVARCHAR(?))",
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
        {"id": {"$between": [1, 2]}},
        [1, 2],
        "WHERE JSON_VALUE(VEC_META, '$.id') BETWEEN TO_DOUBLE(?) AND TO_DOUBLE(?)",
        [1, 2],
    ),
    (
        {"id": {"$between": [1, 1]}},
        [1],
        "WHERE JSON_VALUE(VEC_META, '$.id') BETWEEN TO_DOUBLE(?) AND TO_DOUBLE(?)",
        [1, 1],
    ),
    (
        {"name": {"$in": ["adam", "bob"]}},
        [1, 2],
        "WHERE JSON_VALUE(VEC_META, '$.name') IN (TO_NVARCHAR(?), TO_NVARCHAR(?))",
        ["adam", "bob"],
    ),
]

TYPE_4B_FILTERING_TEST_CASES = [
    # Test $nin, which is missing in TYPE_4_FILTERING_TEST_CASES
    (
        {"name": {"$nin": ["adam", "bob"]}},
        [3],
        "WHERE JSON_VALUE(VEC_META, '$.name') NOT IN (TO_NVARCHAR(?), TO_NVARCHAR(?))",
        ["adam", "bob"],
    ),
]

TYPE_5_FILTERING_TEST_CASES = [
    # These involve special operators like $like, $ilike that
    # may be specified to certain databases.
    (
        {"name": {"$like": "a%"}},
        [1],
        "WHERE JSON_VALUE(VEC_META, '$.name') LIKE TO_NVARCHAR(?)",
        ["a%"],
    ),
    (
        {"name": {"$like": "%a%"}},  # adam and jane
        [1, 3],
        "WHERE JSON_VALUE(VEC_META, '$.name') LIKE TO_NVARCHAR(?)",
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

ERROR_FILTERING_TEST_CASES = [
    # These involve invalid filter formats that should raise errors
    # unknown logical operator
    (
        {"$xor": [{"id": 1}, {"id": 2}]},
        "Unsupported logical operation for operator='$xor', operands=[{'id': 1}, {'id': 2}]",
    ),
    # unknown column operator
    (
        {"id": {"$unknown": 1}},
        "Unsupported column operation for operator='$unknown', operands=1",
    ),
    # more than one operator at the same level
    (
      {"name": {"$eq": "adam", "$ne": "bob"}},
      "Expecting a single entry 'operator: operands'"
      f", but got value={{'$eq': 'adam', '$ne': 'bob'}}"
    ),
    # plain value is not supported
    (
        {"name": ["abcd"]},
        "Cannot deduce SQL operand for ['abcd'], type: <class 'list'>"
    ),
    # # logical operators
    (
        {"$or": [{"id": 1}]},
        "Expected a list of at least 2 operands for operator='$or', but got 1"
    ),
    (
        {"$and": "adam"},
        "Expected a list of operands for operator='$and', but got adam"
    ),
    # # contains operator
    (
        {"tags": {"$contains": ""}},
        "Expected a non-empty NVARCHAR operand for IN, but got '' (NVARCHAR)"
    ),
    (
        {"tags": {"$contains": 5}},
        "Expected a non-empty NVARCHAR operand for IN, but got 5.0 (DOUBLE)"
    ),
    # # like operator
    (
        {"name": {"$like": False}},
        "Expected a NVARCHAR operand for LIKE, but got false (BOOLEAN)"
    ),
    # between operator
    (
        {"id": {"$between": [1]}},
        "Expected 2 operands for BETWEEN, but got 1"
    ),
    (
        {"id": {"$between": [1, "2"]}},
        "Expected operands of the same type for BETWEEN, but got [1.0 (DOUBLE), '2' (NVARCHAR)]"
    ),
    (
        {"id": {"$between": [False, True]}},
        "Expected operand types ('DOUBLE', 'NVARCHAR', 'DATE') for BETWEEN, but got false (BOOLEAN)"
    ),
    # in operators
    (
        {"name": {"$in": []}},
        "Expected a non-empty list of operands for IN"
    ),
    (
        {"name": {"$in": ["adam", 1]}},
        "Expected operands of the same type for IN, but got ['adam' (NVARCHAR), 1.0 (DOUBLE)]"
    ),
    (
        {"name": {"$in": {"unexpected": "dict"}}},
        "Expected list of operands, but got {'unexpected': 'dict'}"
    ),
    (
        {"name": {"$nin": []}},
        "Expected a non-empty list of operands for NOT IN"
    ),
    (
        {"name": {"$nin": ["adam", 1]}},
        "Expected operands of the same type for NOT IN, but got ['adam' (NVARCHAR), 1.0 (DOUBLE)]"
    ),
    (
        {"name": {"$nin": {"unexpected": "dict"}}},
        "Expected list of operands, but got {'unexpected': 'dict'}"
    ),
    # eq and ne operators
    (
        {"name": {"$eq": ["unexpected", "list"]}},
        "Cannot deduce SQL operand for ['unexpected', 'list'], type: <class 'list'>"
    ),
    (
        {"name": {"$ne": {"unexpected": "dict"}}},
        "Cannot deduce SQL operand for {'unexpected': 'dict'}, type: <class 'dict'>" 
    ),
    # gt, gte, lt, lte operators
    (
        {"name": {"$gt": ["unexpected", "list"]}},
        "Cannot deduce SQL operand for ['unexpected', 'list'], type: <class 'list'>"
    ),
    (
        {"name": {"$gte": False}},
        "Expected operand types ('DOUBLE', 'NVARCHAR', 'DATE') for >=, but got false (BOOLEAN)"
    ),
    (
        {"name": {"$lt": ["unexpected", "list"]}},
        "Cannot deduce SQL operand for ['unexpected', 'list'], type: <class 'list'>"
    ),
    (
        {"name": {"$lte": True}},
        "Expected operand types ('DOUBLE', 'NVARCHAR', 'DATE') for <=, but got true (BOOLEAN)" 
    ),
]

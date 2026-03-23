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

TYPE_6_DATE_FILTERING_TEST_CASES = [
    # These involve date filtering with TO_DATE
    # Implicit $eq with date
    (
        {"date": {"type": "date", "date": "2021-01-01"}},
        [1, 3],  # adam and jane have date 2021-01-01
        "WHERE JSON_VALUE(VEC_META, '$.date') = TO_DATE(?)",
        ["2021-01-01"],
    ),
    # Explicit $eq with date
    (
        {"date": {"$eq": {"type": "date", "date": "2021-01-02"}}},
        [2],  # bob has date 2021-01-02
        "WHERE JSON_VALUE(VEC_META, '$.date') = TO_DATE(?)",
        ["2021-01-02"],
    ),
    # $ne with date
    (
        {"date": {"$ne": {"type": "date", "date": "2021-01-01"}}},
        [2],  # bob is the only one with a different date
        "WHERE JSON_VALUE(VEC_META, '$.date') <> TO_DATE(?)",
        ["2021-01-01"],
    ),
    # $gt with date
    (
        {"date": {"$gt": {"type": "date", "date": "2021-01-01"}}},
        [2],  # bob has date 2021-01-02
        "WHERE JSON_VALUE(VEC_META, '$.date') > TO_DATE(?)",
        ["2021-01-01"],
    ),
    # $gte with date
    (
        {"date": {"$gte": {"type": "date", "date": "2021-01-01"}}},
        [1, 2, 3],  # all documents
        "WHERE JSON_VALUE(VEC_META, '$.date') >= TO_DATE(?)",
        ["2021-01-01"],
    ),
    # $lt with date
    (
        {"date": {"$lt": {"type": "date", "date": "2021-01-02"}}},
        [1, 3],  # adam and jane
        "WHERE JSON_VALUE(VEC_META, '$.date') < TO_DATE(?)",
        ["2021-01-02"],
    ),
    # $lte with date
    (
        {"date": {"$lte": {"type": "date", "date": "2021-01-01"}}},
        [1, 3],  # adam and jane
        "WHERE JSON_VALUE(VEC_META, '$.date') <= TO_DATE(?)",
        ["2021-01-01"],
    ),
    # $between with dates
    (
        {"date": {"$between": [
            {"type": "date", "date": "2021-01-01"},
            {"type": "date", "date": "2021-01-02"}
        ]}},
        [1, 2, 3],  # all documents
        "WHERE JSON_VALUE(VEC_META, '$.date') BETWEEN TO_DATE(?) AND TO_DATE(?)",
        ["2021-01-01", "2021-01-02"],
    ),
    # $in with dates
    (
        {"date": {"$in": [
            {"type": "date", "date": "2021-01-01"},
            {"type": "date", "date": "2021-01-03"}  # date that doesn't exist
        ]}},
        [1, 3],  # adam and jane
        "WHERE JSON_VALUE(VEC_META, '$.date') IN (TO_DATE(?), TO_DATE(?))",
        ["2021-01-01", "2021-01-03"],
    ),
]

FILTERING_TEST_CASES = [
    *TYPE_1_FILTERING_TEST_CASES,
    *TYPE_2_FILTERING_TEST_CASES,
    *TYPE_3_FILTERING_TEST_CASES,
    *TYPE_4_FILTERING_TEST_CASES,
    *TYPE_4B_FILTERING_TEST_CASES,
    *TYPE_5_FILTERING_TEST_CASES,
    *TYPE_6_DATE_FILTERING_TEST_CASES,
]

ERROR_FILTERING_TEST_CASES = [
    # These involve invalid filter formats that should raise errors
    # unknown logical operator
    (
        {"$xor": [{"id": 1}, {"id": 2}]},
        "Operator $xor is not supported",
    ),
    # unknown column operator
    (
        {"id": {"$unknown": 1}},
        "Operator $unknown is not supported",
    ),
    # more than one operator at the same level
    (
      {"name": {"$eq": "adam", "$ne": "bob"}},
      "Filter expects a single 'operator: operands' entry, but got {'$eq': 'adam', '$ne': 'bob'}"
    ),
    # plain value is not supported (implicit $eq)
    (
        {"name": ["abcd"]},
        "Implicit operator $eq received unsupported operand: ['abcd']"
    ),
    # # logical operators
    (
        {"$or": [{"id": 1}]},
        "Operator $or expects at least 2 operands, but got [{'id': 1}]"
    ),
    (
        {"$and": "adam"},
        "Operator $and expects a list of operands, but got 'adam'"
    ),
    # # contains operator
    (
        {"tags": {"$contains": ""}},
        "Operator $contains expects a non-empty string operand, but got '' (str)"
    ),
    (
        {"tags": {"$contains": 5}},
        "Operator $contains expects a non-empty string operand, but got 5 (int)"
    ),
    # # like operator
    (
        {"name": {"$like": False}},
        "Operator $like expects a string operand, but got False (bool)"
    ),
    # between operator
    (
        {"id": {"$between": [1]}},
        "Operator $between expects 2 operands, but got [1 (int)]"
    ),
    (
        {"id": {"$between": [1, "2"]}},
        "Operator $between expects operands of the same type, but got [1 (int), '2' (str)]"
    ),
    (
        {"id": {"$between": [1, 2.0]}},
        "Operator $between expects operands of the same type, but got [1 (int), 2.0 (float)]"
    ),
    (
        {"id": {"$between": [False, True]}},
        "Operator $between expects operand types (int, float, str, date), but got [False (bool), True (bool)]"
    ),
    # in operators
    (
        {"name": {"$in": []}},
        "Operator $in expects at least 1 operand"
    ),
    (
        {"name": {"$in": ["adam", 1]}},
        "Operator $in expects operands of the same type, but got ['adam', 1]"
    ),
    (
        {"name": {"$in": {"unexpected": "dict"}}},
        "Operator $in expects list/tuple of operands, but got {'unexpected': 'dict'}"
    ),
    (
        {"name": {"$nin": []}},
        "Operator $nin expects at least 1 operand"
    ),
    (
        {"name": {"$nin": ["adam", 1]}},
        "Operator $nin expects operands of the same type, but got ['adam', 1]"
    ),
    (
        {"name": {"$nin": {"unexpected": "dict"}}},
        "Operator $nin expects list/tuple of operands, but got {'unexpected': 'dict'}"
    ),
    # eq and ne operators
    (
        {"name": {"$eq": ["unexpected", "list"]}},
        "Operator $eq expects a single operand, but got list: ['unexpected', 'list']"
    ),
    (
        {"name": {"$ne": {"unexpected": "dict"}}},
        "Operator $ne: Operand cannot be created from {'unexpected': 'dict'}"
    ),
    # gt, gte, lt, lte operators
    (
        {"name": {"$gt": ["unexpected", "list"]}},
        "Operator $gt expects a single operand, but got list: ['unexpected', 'list']"
    ),
    (
        {"name": {"$gte": False}},
        "Operator $gte expects operand of type int/float/str/date, but got False (bool)"
    ),
    (
        {"name": {"$lt": ["unexpected", "list"]}},
        "Operator $lt expects a single operand, but got list: ['unexpected', 'list']"
    ),
    (
        {"name": {"$lte": True}},
        "Operator $lte expects operand of type int/float/str/date, but got True (bool)"
    ),
    # date operand errors
    (
        {"date": {"$eq": {"type": "date"}}},  # missing 'date' key
        "Operator $eq: Date operand missing 'date' key: {'type': 'date'}"
    ),
    (
        {"date": {"$gt": {"type": "date", "date": ""}}},  # empty date string
        "Operator $gt: Date operand with empty value"
    ),
]

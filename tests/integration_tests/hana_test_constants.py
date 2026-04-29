class HanaTestConstants:
    TEXTS = ["foo", "bar", "baz", "bak", "cat"]
    METADATAS = [
        {"start": 0, "end": 100, "quality": "good", "ready": True},
        {"start": 100, "end": 200, "quality": "bad", "ready": False},
        {"start": 200, "end": 300, "quality": "ugly", "ready": True},
        {"start": 200, "quality": "ugly", "ready": True, "Owner": "Steve"},
        {"start": 300, "quality": "ugly", "Owner": "Steve"},
    ]
    TABLE_NAME = "TEST_TABLE"
    TABLE_NAME_CUSTOM_DB = "CUSTOM_TEST_TABLE"

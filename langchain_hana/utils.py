from enum import Enum


class DistanceStrategy(str, Enum):
    """Enumerator of the Distance strategies for calculating distances
    between vectors."""

    EUCLIDEAN_DISTANCE = "EUCLIDEAN_DISTANCE"
    MAX_INNER_PRODUCT = "MAX_INNER_PRODUCT"
    DOT_PRODUCT = "DOT_PRODUCT"
    JACCARD = "JACCARD"
    COSINE = "COSINE"


def _validate_k(k: int):
    if not isinstance(k, int) or k <= 0:
        raise ValueError("Parameter 'k' must be an integer greater than 0")


def _validate_k_and_fetch_k(k: int, fetch_k: int):
    _validate_k(k)
    if not isinstance(fetch_k, int) or fetch_k < k:
        raise ValueError(
            "Parameter 'fetch_k' must be an integer greater than or equal to 'k'"
        )


def _generate_cross_encode_sql_and_params(
        text_column: str, metadata_column: str, query: str, rank_fields: list[str], rerank_model_id: str
    ) -> tuple[str, list]:
        if rank_fields:
            cross_encode_input = f"'{text_column}:' || TO_NVARCHAR({text_column})"
            for field in rank_fields:
                cross_encode_input += f"|| '| {field}:' || TO_NVARCHAR(COALESCE(JSON_VALUE({metadata_column}, '$.{field}'), ''))"
        else:
            cross_encode_input = f"TO_NVARCHAR({text_column})"

        cross_encode_sql = f"CROSS_ENCODE({cross_encode_input}, ?, ?) OVER()"

        cross_encode_params = [query, rerank_model_id]
        return cross_encode_sql, cross_encode_params

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

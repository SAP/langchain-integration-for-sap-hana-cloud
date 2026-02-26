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

class OperatorErrorMessageGenerator:
    """Class to generate error messages for various operator and value type errors
    in CreateWhereClause."""
    
    # Empty Filter
    @staticmethod
    def err_empty_filter():
        return "Empty filter provided"

    # Unknown Operators
    @staticmethod
    def err_unknown_logical_operator(operator, allowed):
        return (
            f"operator='{operator}' not in "
            f"LOGICAL_OPERATORS_TO_SQL.keys()={allowed}"
        )

    @staticmethod
    def err_unknown_comparison_operator(operator, allowed):
        return (
            f"operator='{operator}' not in "
            f"COMPARISON_OPERATORS_TO_SQL.keys()={allowed}"
        )
    
    # Structural Errors
    @staticmethod
    def err_single_operator_expected(value):
        return (
            "Expecting a single entry 'operator: operands', "
            f"but got value={value}"
        )

    @staticmethod
    def err_unsupported_filter_type(the_type, value):
        return f"Unsupported  value type: {the_type=}, value={value}"

    @staticmethod
    def err_logical_operands(operator, operands):
        return (
            f"Expected a list of atleast two operands for operator='{operator}', "
            f"but got operands={operands}"
        )

    # $contains
    @staticmethod
    def err_contains(operator, operands):
        return (
            f"Expected a non-empty string operand for operator='{operator}', "
            f"but got operands={operands}"
        )

    # $like / $ilike
    @staticmethod
    def err_like(operator, operands):
        return (
            f"Expected a string operand for operator='{operator}', "
            f"but got operands={operands}"
        )


    # $between
    @staticmethod
    def err_between_length(operator, operands):
        return (
            f"Expected a list of two operands for operator='{operator}', "
            f"but got operands={operands}"
        )

    @staticmethod
    def err_between_type_match(operator, operands):
        return (
            f"Expected operands of the same type for operator='{operator}', "
            f"but got operands={operands}"
        )

    @staticmethod
    def err_between_allowed_types(operator, operands):
        return (
            f"Expected a list of (int, float, str, date) for operator='{operator}', "
            f"but got operands={operands}"
        )

    # $in / $nin
    @staticmethod
    def err_in_non_empty(operator, operands):
        return (
            f"Expected a non-empty list of operands for operator='{operator}', "
            f"but got operands={operands}"
        )

    @staticmethod
    def err_in_type_match(operator, operands):
        return (
            f"Expected operands of the same type for operator='{operator}', "
            f"but got operands={operands}"
        )

    # $eq / $ne
    @staticmethod
    def err_eq_ne(operator, operands):
        return (
            f"Expected a (int, float, str, bool, date, None) "
            f"for operator='{operator}', but got operands={operands}"
        )

    # $gt / $gte / $lt / $lte
    @staticmethod
    def err_comparison(operator, operands):
        return (
            f"Expected a (int, float, str, date) "
            f"for operator='{operator}', but got operands={operands}"
        )

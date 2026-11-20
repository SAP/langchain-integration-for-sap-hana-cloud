import re
from enum import Enum

from hdbcli import dbapi


class DistanceStrategy(str, Enum):
    """Enumerator of the Distance strategies for calculating distances
    between vectors."""

    EUCLIDEAN_DISTANCE = "EUCLIDEAN_DISTANCE"
    MAX_INNER_PRODUCT = "MAX_INNER_PRODUCT"
    DOT_PRODUCT = "DOT_PRODUCT"
    JACCARD = "JACCARD"
    COSINE = "COSINE"


def _validate_k(k: int) -> None:
    if not isinstance(k, int) or k <= 0:
        raise ValueError("Parameter 'k' must be an integer greater than 0")


def _validate_k_and_fetch_k(k: int, fetch_k: int) -> None:
    _validate_k(k)
    if not isinstance(fetch_k, int) or fetch_k < k:
        raise ValueError(
            "Parameter 'fetch_k' must be an integer greater than or equal to 'k'"
        )


def _table_ref(table_name: str, schema_name: str | None) -> str:
    """Return a quoted, optionally schema-qualified SQL table reference."""
    if schema_name:
        return f'"{schema_name}"."{table_name}"'
    return f'"{table_name}"'


def _sanitize_name(input_str: str) -> str:
    """Strip anything that is not alphanumeric or an underscore."""
    return re.sub(r"[^a-zA-Z0-9_]", "", input_str)


def _json_value(column: str, key: str) -> str:
    """Build a HANA ``JSON_VALUE`` scalar accessor for a metadata key.

    ``column`` is the (already quoted/qualified) JSON column expression and ``key``
    the JSON path key. Callers are responsible for validating/sanitizing ``key``.
    """
    return f"JSON_VALUE({column}, '$.{key}')"


_VALID_IDENTIFIER_RE = re.compile(r"[_a-zA-Z][_a-zA-Z0-9]*")


def _validate_identifier(name: str) -> None:
    """Raise ``ValueError`` if ``name`` is not a safe SQL/JSON identifier.

    Allows letters, digits and underscores only, and forbids a leading digit — the
    rule for an unquoted SQL/JSON member name. Used for metadata keys/columns that
    are interpolated into ``JSON_VALUE`` paths and SQL, so this prevents injection.
    """
    if not _VALID_IDENTIFIER_RE.fullmatch(name):
        raise ValueError(
            f"Invalid identifier {name!r}: only letters, digits, and underscores "
            "are allowed, and it must not start with a digit."
        )


def _table_exists(
    connection: dbapi.Connection,
    table_name: str,
    schema_name: str | None = None,
) -> bool:
    """Return ``True`` if ``table_name`` exists in ``schema_name``.

    Falls back to the connection's current schema when ``schema_name`` is empty.
    """
    schema_str = f"'{schema_name}'" if schema_name else "CURRENT_SCHEMA"
    sql_str = (
        f"SELECT COUNT(*) FROM SYS.TABLES WHERE SCHEMA_NAME = {schema_str} "
        "AND TABLE_NAME = ?"
    )
    cur = connection.cursor()
    try:
        cur.execute(sql_str, (table_name,))
        if cur.has_result_set():
            rows = cur.fetchall()
            return bool(rows) and rows[0][0] >= 1
    finally:
        cur.close()
    return False

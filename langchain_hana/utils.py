import re
import struct
from collections.abc import Sequence
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


# --------------------------------------------------------------------------- #
# Vector column type: serialization, SQL conversion and datatype support
#
# Shared by the HanaDB vector store and the LangGraph memory stores so there is a
# single source of truth for how a configured ``vector_column_type`` (REAL_VECTOR
# or HALF_VECTOR) is serialized, wrapped in SQL, and validated against the HANA
# instance. ``HanaDB`` and the stores pass their own ``vector_column_type`` in.
# --------------------------------------------------------------------------- #

VECTOR_COLUMN_SQL_TYPES = ["REAL_VECTOR", "HALF_VECTOR"]
DEFAULT_VECTOR_COLUMN_TYPE = "REAL_VECTOR"


def _validate_vector_column_type(vector_column_type: str) -> str:
    """Upper-case and validate a vector column type against the supported set.

    Pure (no DB access): checks only that the name is one HANA understands. The
    DB-support check (whether this instance actually offers the type) is
    :func:`_validate_datatype_support`.
    """
    vector_column_type_upper = vector_column_type.upper()
    if vector_column_type_upper not in VECTOR_COLUMN_SQL_TYPES:
        raise ValueError(
            f"Unsupported vector_column_type: {vector_column_type}. "
            f"Must be one of {', '.join(VECTOR_COLUMN_SQL_TYPES)}"
        )
    return vector_column_type_upper


def _serialize_vector(values: Sequence[float], vector_column_type: str) -> bytes:
    """Serialize a float vector to HANA's fvecs binary format for the column type.

    ``REAL_VECTOR`` uses 4-byte floats (``<I{n}f``); ``HALF_VECTOR`` uses 2-byte
    half-precision floats (``<I{n}e``). The leading ``<I`` is the element count.
    """
    if vector_column_type == "HALF_VECTOR":
        return struct.pack(f"<I{len(values)}e", len(values), *values)
    if vector_column_type == "REAL_VECTOR":
        return struct.pack(f"<I{len(values)}f", len(values), *values)
    raise ValueError(f"Unsupported vector column type: {vector_column_type}")


def _deserialize_vector(fvecs: memoryview, vector_column_type: str) -> list[float]:
    """Extract a list of floats from HANA's fvecs binary format for the column type."""
    dim = struct.unpack_from("<I", fvecs, 0)[0]
    if vector_column_type == "HALF_VECTOR":
        return list(struct.unpack_from(f"<{dim}e", fvecs, 4))
    if vector_column_type == "REAL_VECTOR":
        return list(struct.unpack_from(f"<{dim}f", fvecs, 4))
    raise ValueError(f"Unsupported vector column type: {vector_column_type}")


def _to_target_vector_type(expr: str, vector_column_type: str) -> str:
    """Wrap a vector SQL expression in the matching ``TO_<type>`` conversion.

    e.g. ``_to_target_vector_type("?", "HALF_VECTOR")`` -> ``"TO_HALF_VECTOR(?)"``.
    """
    if vector_column_type in VECTOR_COLUMN_SQL_TYPES:
        return f"TO_{vector_column_type}({expr})"
    raise ValueError(f"Unsupported vector type: {vector_column_type}")


def _get_min_supported_version(datatype: str) -> str:
    """Minimum HANA Cloud instance version that offers a given vector datatype."""
    if datatype == "HALF_VECTOR":
        return "2025.15 (QRC 2/2025)"
    if datatype == "REAL_VECTOR":
        return "2024.2 (QRC 1/2024)"
    raise ValueError(f"Unknown datatype: '{datatype}'")


def _get_instance_version(connection: dbapi.Connection) -> str | None:
    """Return the HANA Cloud instance version, or ``None`` if it can't be read."""
    cursor = connection.cursor()
    try:
        cursor.execute("SELECT CLOUD_VERSION FROM SYS.M_DATABASE;")
        result = cursor.fetchone()
        if result is None:
            return None
        return result[0]
    except dbapi.Error:
        return None
    finally:
        cursor.close()


def _get_available_datatypes(connection: dbapi.Connection) -> set[str]:
    """Return the datatype names the HANA instance offers (``SYS.DATA_TYPES``)."""
    cur = connection.cursor()
    try:
        cur.execute("SELECT TYPE_NAME FROM SYS.DATA_TYPES")
        if cur.has_result_set():
            rows = cur.fetchall()
            return {row[0] for row in rows}
        raise ValueError("No data types returned by the database.")
    finally:
        cur.close()


def _datatype_support_error(datatype: str, instance_version: str | None = None) -> str:
    """Build the error message for a vector datatype unavailable on the instance.

    Pure (no DB access): callers that already know the instance version (e.g. the
    async store, which must run its own awaited query) pass it in.
    """
    error_message = f"'{datatype}' is not available on this HANA instance.\n"
    if instance_version:
        error_message += f"Instance version: {instance_version}\n"
    min_instance_version = _get_min_supported_version(datatype)
    error_message += f"Minimum required instance version: {min_instance_version}"
    return error_message


def _validate_datatype_support(connection: dbapi.Connection, datatype: str) -> bool:
    """Return ``True`` if ``datatype`` is available on the instance, else raise.

    Uses a blocking (sync) cursor, so async callers cannot use it; they reuse the
    pure pieces (:func:`_get_min_supported_version`, :func:`_datatype_support_error`)
    and run their own awaited ``SYS.DATA_TYPES`` / ``SYS.M_DATABASE`` queries.
    """
    if datatype in _get_available_datatypes(connection):
        return True
    instance_version = _get_instance_version(connection)
    raise ValueError(_datatype_support_error(datatype, instance_version))

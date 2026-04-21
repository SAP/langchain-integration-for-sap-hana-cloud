"""Shared utilities for vectorstore modules."""

import logging
import re
from typing import Pattern

from hdbcli import dbapi

logger = logging.getLogger(__name__)

# Compiled pattern for validating metadata identifiers
_compiled_pattern: Pattern = re.compile("^[_a-zA-Z][_a-zA-Z0-9]*$")


def _sanitize_metadata_keys(metadata_keys: list[str]):
    """Validate that all metadata keys are valid identifiers."""
    for key in metadata_keys:
        if not _compiled_pattern.match(key):
            raise ValueError(f"Invalid metadata key {key}")


def _validate_rerank_model_id(model_id: str, connection: dbapi.Connection) -> None:
    """Validate that the provided model is supported by SAP HANA for reranking."""
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("model_id must be a non-empty string")
    with connection.cursor() as cur:
        try:
            cur.execute(
                # CROSS_ENCODE IS A WINDOW FUNCTION
                "SELECT CROSS_ENCODE('test', 'test', ?) OVER() FROM SYS.DUMMY",
                [model_id],
            )
        except dbapi.Error as e:
            logger.error(f"Database error while validating rerank model ID: {e}")
            raise


def _generate_cross_encode_sql_and_params(
    text_column: str,
    metadata_column: str,
    query: str,
    rank_fields: list[str],
    rerank_model_id: str,
) -> tuple[str, list]:
    """Generate SQL and parameters for CROSS_ENCODE function."""
    if rank_fields:
        cross_encode_input = f"""'{text_column}:' || TO_NVARCHAR("{text_column}")"""
        for field in rank_fields:
            cross_encode_input += (
                f"|| '| {field}:' || TO_NVARCHAR(COALESCE(JSON_VALUE("
                f""""{metadata_column}", '$.{field}'), ''))"""
            )
    else:
        cross_encode_input = f"""TO_NVARCHAR("{text_column}")"""

    cross_encode_sql = f"CROSS_ENCODE({cross_encode_input}, ?, ?) OVER()"

    cross_encode_params = [query, rerank_model_id]
    return cross_encode_sql, cross_encode_params

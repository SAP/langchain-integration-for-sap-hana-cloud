"""Shared embedding + vector-column-type configs for the store integration suites.

Both ``test_hana_store.py`` (sync) and ``test_async_hana_store.py`` (async)
parametrize their index-dependent tests over :data:`STORE_CONFIG_IDS` so the two
suites stay in parity and exercise the same backends.

:func:`resolve` turns a config id into the ``(IndexConfig, vector_column_type)``
pair a store should be built with, calling :func:`pytest.skip` when the instance
(or environment) cannot support that config -- so the suite stays green on a bare
instance and lights up where the extra capability is provisioned.
"""

from __future__ import annotations

import pytest
from hdbcli import dbapi
from langgraph.store.base import IndexConfig

from tests.integration_tests.fake_embeddings import ConsistentFakeEmbeddings

# External (FakeEmbeddings) dimensionality. Internal configs (added on the
# internal-embeddings branch) read their dimensionality from the environment.
_EXTERNAL_DIMS = 10

# External embeddings over each supported stored vector column type.
EXTERNAL_REAL_VECTOR = "external-REAL_VECTOR"
EXTERNAL_HALF_VECTOR = "external-HALF_VECTOR"

# Maps each config id to the stored vector column type it exercises.
_VECTOR_COLUMN_TYPES: dict[str, str] = {
    EXTERNAL_REAL_VECTOR: "REAL_VECTOR",
    EXTERNAL_HALF_VECTOR: "HALF_VECTOR",
}

#: pytest params: each embedding/column-type config index-dependent tests run over.
STORE_CONFIG_IDS: list[str] = list(_VECTOR_COLUMN_TYPES)


def _half_vector_supported(conn: dbapi.Connection) -> bool:
    cur = conn.cursor()
    try:
        cur.execute("SELECT TYPE_NAME FROM SYS.DATA_TYPES")
        return "HALF_VECTOR" in {row[0] for row in cur.fetchall()}
    finally:
        cur.close()


def _index_config(config_id: str) -> IndexConfig:
    return {
        "dims": _EXTERNAL_DIMS,
        "embed": ConsistentFakeEmbeddings(dimensionality=_EXTERNAL_DIMS),
        "fields": ["text"],
    }


def resolve(config_id: str, admin_conn: dbapi.Connection) -> tuple[IndexConfig, str]:
    """Return ``(index_config, vector_column_type)`` for ``config_id``.

    ``admin_conn`` is a blocking connection used only for capability probes
    (e.g. HALF_VECTOR availability). Skips the test when the config is
    unsupported here.
    """
    vector_column_type = _VECTOR_COLUMN_TYPES[config_id]
    if vector_column_type == "HALF_VECTOR" and not _half_vector_supported(admin_conn):
        pytest.skip("HANA instance does not offer HALF_VECTOR")
    return _index_config(config_id), vector_column_type

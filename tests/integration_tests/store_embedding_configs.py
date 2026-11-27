"""Shared embedding + vector-column-type configs for the store integration suites.

Both ``test_hana_store.py`` (sync) and ``test_async_hana_store.py`` (async)
parametrize their index-dependent tests over :data:`STORE_CONFIG_IDS` so the two
suites stay in parity and exercise the same backends.

:func:`resolve` turns a config id into the ``(IndexConfig, vector_column_type)``
pair a store should be built with, calling :func:`pytest.skip` when the instance
(or environment) cannot support that config -- so the suite stays green on a bare
instance and lights up where the extra capability is provisioned.

The matrix covers:

* external embeddings (``ConsistentFakeEmbeddings``) over ``REAL_VECTOR`` and
  ``HALF_VECTOR`` columns;
* HANA internal embeddings (``VECTOR_EMBEDDING`` computed in-database) over
  ``REAL_VECTOR`` and ``HALF_VECTOR`` columns -- the latter exercises wrapping
  ``VECTOR_EMBEDDING(...)`` in ``TO_HALF_VECTOR(...)``;
* internal embeddings against a remote source, and a remote source with an
  explicit schema.

Internal configs skip unless the relevant environment is provisioned:

* ``HANA_DB_EMBEDDING_MODEL_ID`` (internal, no remote source),
* ``HANA_DB_EMBEDDING_REMOTE_MODEL_ID`` + ``HANA_DB_EMBEDDING_REMOTE_SOURCE``
  (remote-source configs),
* ``HANA_DB_EMBEDDING_REMOTE_SOURCE_SCHEMA`` (remote-source-with-schema).

Internal dimensionality comes from ``HANA_DB_EMBEDDING_DIMS`` when set, otherwise
it is probed live from the model.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass

import pytest
from hdbcli import dbapi
from langgraph.store.base import IndexConfig

from langchain_hana.embeddings import HanaInternalEmbeddings
from langchain_hana.utils import _vector_embedding_sql_and_params
from tests.integration_tests.fake_embeddings import ConsistentFakeEmbeddings

# External (FakeEmbeddings) dimensionality. Internal configs use the model's own
# output dimensionality: HANA_DB_EMBEDDING_DIMS when set, otherwise probed live.
_EXTERNAL_DIMS = 10

# External embeddings over each supported stored vector column type.
EXTERNAL_REAL_VECTOR = "external-REAL_VECTOR"
EXTERNAL_HALF_VECTOR = "external-HALF_VECTOR"
# Internal (in-database VECTOR_EMBEDDING) over each column type.
INTERNAL_REAL_VECTOR = "internal-REAL_VECTOR"
INTERNAL_HALF_VECTOR = "internal-HALF_VECTOR"
# Internal with a remote source, and a remote source plus explicit schema.
INTERNAL_REMOTE_SOURCE_REAL_VECTOR = "internal_remote_source-REAL_VECTOR"
INTERNAL_REMOTE_SOURCE_SCHEMA_REAL_VECTOR = (
    "internal_remote_source_and_schema-REAL_VECTOR"
)


@dataclass(frozen=True)
class _ConfigSpec:
    vector_column_type: str
    internal: bool = False
    remote_source: bool = False
    remote_source_schema: bool = False


_SPECS: dict[str, _ConfigSpec] = {
    EXTERNAL_REAL_VECTOR: _ConfigSpec("REAL_VECTOR"),
    EXTERNAL_HALF_VECTOR: _ConfigSpec("HALF_VECTOR"),
    INTERNAL_REAL_VECTOR: _ConfigSpec("REAL_VECTOR", internal=True),
    INTERNAL_HALF_VECTOR: _ConfigSpec("HALF_VECTOR", internal=True),
    INTERNAL_REMOTE_SOURCE_REAL_VECTOR: _ConfigSpec(
        "REAL_VECTOR", internal=True, remote_source=True
    ),
    INTERNAL_REMOTE_SOURCE_SCHEMA_REAL_VECTOR: _ConfigSpec(
        "REAL_VECTOR", internal=True, remote_source=True, remote_source_schema=True
    ),
}

#: pytest params: each embedding/column-type config index-dependent tests run over.
STORE_CONFIG_IDS: list[str] = list(_SPECS)


def _half_vector_supported(conn: dbapi.Connection) -> bool:
    cur = conn.cursor()
    try:
        cur.execute("SELECT TYPE_NAME FROM SYS.DATA_TYPES")
        return "HALF_VECTOR" in {row[0] for row in cur.fetchall()}
    finally:
        cur.close()


def _external_index_config() -> IndexConfig:
    return {
        "dims": _EXTERNAL_DIMS,
        "embed": ConsistentFakeEmbeddings(dimensionality=_EXTERNAL_DIMS),
        "fields": ["text"],
    }


def _probe_internal_dims(
    admin_conn: dbapi.Connection,
    model_id: str,
    remote_source_schema: str,
    remote_source: str,
) -> int:
    """Return a model's output dimensionality by embedding a probe string.

    ``VECTOR_EMBEDDING`` yields a ``REAL_VECTOR`` whose fvecs blob starts with a
    little-endian ``uint32`` element count -- which is the dimensionality.
    """
    expr, params = _vector_embedding_sql_and_params(
        "dimension probe", "QUERY", model_id, remote_source_schema, remote_source
    )
    cur = admin_conn.cursor()
    try:
        cur.execute(f"SELECT {expr} FROM sys.DUMMY", params)
        row = cur.fetchone()
    finally:
        cur.close()
    assert row is not None
    return int(struct.unpack_from("<I", bytes(row[0]), 0)[0])


def _internal_index_config(
    spec: _ConfigSpec, admin_conn: dbapi.Connection
) -> IndexConfig:
    # Remote-source variants use their own model id (served by the remote source),
    # mirroring tests/integration_tests/test_hana_db_internal_embeddings.py.
    if spec.remote_source:
        model_id = os.environ.get("HANA_DB_EMBEDDING_REMOTE_MODEL_ID")
        if not model_id:
            pytest.skip("HANA_DB_EMBEDDING_REMOTE_MODEL_ID not set")
    else:
        model_id = os.environ.get("HANA_DB_EMBEDDING_MODEL_ID")
        if not model_id:
            pytest.skip("HANA_DB_EMBEDDING_MODEL_ID not set")

    remote_source = ""
    remote_schema = ""
    kwargs: dict[str, str] = {"internal_embedding_model_id": model_id}
    if spec.remote_source:
        remote_source = os.environ.get("HANA_DB_EMBEDDING_REMOTE_SOURCE", "")
        if not remote_source:
            pytest.skip("HANA_DB_EMBEDDING_REMOTE_SOURCE not set")
        kwargs["remote_source"] = remote_source
    if spec.remote_source_schema:
        remote_schema = os.environ.get("HANA_DB_EMBEDDING_REMOTE_SOURCE_SCHEMA", "")
        if not remote_schema:
            pytest.skip("HANA_DB_EMBEDDING_REMOTE_SOURCE_SCHEMA not set")
        kwargs["remote_source_schema"] = remote_schema

    # The stored vector column is declared with a fixed dimensionality, so it must
    # match the model's output. Honor an explicit override, else probe the model.
    env_dims = os.environ.get("HANA_DB_EMBEDDING_DIMS")
    dims = (
        int(env_dims)
        if env_dims
        else _probe_internal_dims(admin_conn, model_id, remote_schema, remote_source)
    )
    return {
        "dims": dims,
        "embed": HanaInternalEmbeddings(**kwargs),
        "fields": ["text"],
    }


def resolve(config_id: str, admin_conn: dbapi.Connection) -> tuple[IndexConfig, str]:
    """Return ``(index_config, vector_column_type)`` for ``config_id``.

    ``admin_conn`` is a blocking connection used for capability probes
    (e.g. HALF_VECTOR availability, internal-model dimensionality). Skips the test
    when the config is unsupported here (column type unavailable, or
    internal-embedding env unset).
    """
    spec = _SPECS[config_id]
    if spec.vector_column_type == "HALF_VECTOR" and not _half_vector_supported(
        admin_conn
    ):
        pytest.skip("HANA instance does not offer HALF_VECTOR")

    if spec.internal:
        return _internal_index_config(spec, admin_conn), spec.vector_column_type
    return _external_index_config(), spec.vector_column_type

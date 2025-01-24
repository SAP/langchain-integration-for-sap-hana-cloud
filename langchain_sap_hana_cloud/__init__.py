from importlib import metadata

from langchain_sap_hana_cloud.vectorstores import HanaDB
from langchain_sap_hana_cloud.embeddings import HanaInternalEmbeddings

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "HanaDB",
    "HanaInternalEmbeddings"
    "__version__",
]

from importlib import metadata

from langchain_hana.embeddings import HanaInternalEmbeddings
from langchain_hana.vectorstores import HanaDB
from langchain_hana.graphs import HanaRdfGraph
from langchain_hana.chains.graph_qa import HanaSparqlQAChain

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "HanaDB",
    "HanaInternalEmbeddings",
    "HanaRdfGraph",
    "HanaSparqlQAChain",
    "__version__",
]

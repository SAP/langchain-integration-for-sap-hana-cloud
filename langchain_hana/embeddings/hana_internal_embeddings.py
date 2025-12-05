from langchain_core.embeddings import Embeddings


class HanaInternalEmbeddings(Embeddings):
    """
    A dummy embeddings class for use with HANA's internal embedding functionality.
    This class prevents the use of standard embedding methods and ensures that
    internal embeddings are handled exclusively via database queries.
    """

    def __init__(self, internal_embedding_model_id: str, remote_source: str = ""):
        """
        Initialize the HanaInternalEmbeddings instance.
        Args:
            internal_embedding_model_id (str): The ID of the internal embedding model
                                               used by the HANA database.
        """
        self.model_id = internal_embedding_model_id
        self.remote_source = remote_source

    def embed_query(self, text: str) -> list[float]:
        """
        Override the embed_query method to raise an error.
        This method is not applicable for HANA internal embeddings.
        Raises:
            NotImplementedError: Always raised to show that this method is unsupported.
        """
        raise NotImplementedError(
            "Internal embeddings cannot be used externally. "
            "Use HANA's internal embedding functionality instead."
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Override the embed_documents method to raise an error.
        This method is not applicable for HANA internal embeddings.
        Raises:
            NotImplementedError: Always raised to show that this method is unsupported.
        """
        raise NotImplementedError(
            "Internal embeddings cannot be used externally. "
            "Use HANA's internal embedding functionality instead."
        )

    def get_model_id(self) -> str:
        """
        Retrieve the internal embedding model ID.
        Returns:
            str: The ID of the internal embedding model.
        """
        return self.model_id
    
    def get_remote_source(self) -> str:
        """
        Retrieve the remote source name.
        Returns:
            str: The remote source associated with the internal embedding.
        """
        return self.remote_source

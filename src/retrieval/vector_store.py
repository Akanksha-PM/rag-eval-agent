"""In-process vector store for embedded document chunks.

Every stored chunk is tagged with its product name as metadata, so
retrieval can be scoped to a single product or to any subset of
registered products (e.g. when comparing several products at once).
"""


class VectorStore:
    """Stores chunk embeddings alongside product-tagged metadata."""

    def add(self, product_name, chunks, embeddings):
        """Add a product's chunks and their embeddings to the store."""
        raise NotImplementedError

    def search(self, query_embedding, product_names=None, top_k=None):
        """Return the top_k most similar chunks, optionally scoped to product_names."""
        raise NotImplementedError

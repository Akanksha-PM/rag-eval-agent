"""Retrieves relevant chunks for a question, scoped to one or more products."""

from config import Config
from src.retrieval.embeddings import get_embedder
from src.retrieval.vector_store import VectorStore


def retrieve(
    query: str, products: list[str] | None = None, top_k: int | None = None
) -> list[dict]:
    """Return the top_k chunks most relevant to query, optionally scoped to products."""
    if top_k is None:
        top_k = Config.TOP_K

    # An empty products list isn't a valid "search nothing" request -- it
    # means the caller didn't select any products, which should behave the
    # same as not filtering at all (search everything), same as None.
    if not products:
        products = None

    store = VectorStore()
    if len(store) == 0:
        # Nothing to search yet, so skip embedding the query entirely: the
        # configured embedder (e.g. TF-IDF) may not have a fitted
        # vocabulary if no product has been ingested, and calling it here
        # would raise for no benefit.
        return []

    embedder = get_embedder()
    query_vector = embedder.embed([query])[0]

    return store.search(query_vector, top_k=top_k, products=products)

"""End-to-end ingestion pipeline: load a source, chunk it, and refit the
embedder across every registered product's chunks.

Every call to ingest_product() refits the embedder on the FULL combined
corpus (all products, not just the one being ingested) and rebuilds the
vector store from scratch. That's not incremental-add for a reason: this
project's default embedder (TF-IDF) needs one shared vocabulary for cross-
product comparison to be meaningful at all -- see
embeddings.TFIDFEmbedder.fit()'s docstring.
"""

from src.ingestion import registry
from src.ingestion.chunker import chunk_text
from src.ingestion.loader import load_from_file, load_from_url
from src.retrieval.embeddings import get_embedder
from src.retrieval.vector_store import VectorStore

_VALID_SOURCE_TYPES = ("file", "url")


def ingest_product(name: str, source_type: str, source_value: str) -> dict:
    """Ingest one source for a product, then refit/rebuild the vector store."""
    if source_type not in _VALID_SOURCE_TYPES:
        raise ValueError(
            f"source_type must be 'file' or 'url', got {source_type!r}."
        )

    if source_type == "file":
        text = load_from_file(source_value)
    else:
        text = load_from_url(source_value)

    registry.add_product(name, source_type, source_value)

    new_chunks = chunk_text(text, product=name, source=source_value)

    store = VectorStore()

    # Re-number the new chunks' ids/chunk_index to continue after this
    # product's existing chunks, so ingesting a second source for the same
    # product never collides with ids the first source already produced
    # (chunk_text always numbers a batch starting from 0 on its own).
    existing_product_chunks = store.get_all(product=name)
    start_index = len(existing_product_chunks)
    product_slug = name.lower().replace(" ", "_")
    for offset, chunk in enumerate(new_chunks):
        chunk_index = start_index + offset
        chunk["chunk_index"] = chunk_index
        chunk["id"] = f"{product_slug}_{chunk_index}"

    combined_chunks = store.get_all() + new_chunks
    combined_texts = [chunk["text"] for chunk in combined_chunks]

    embedder = get_embedder()
    embedder.fit(combined_texts)
    vectors = embedder.embed(combined_texts)

    store.rebuild(combined_chunks, vectors)

    total_chunks = len(store.get_all(product=name))
    registry.update_chunk_count(name, total_chunks)

    return {
        "product": name,
        "chunks_added": len(new_chunks),
        "total_chunks": total_chunks,
        "status": "ready",
    }

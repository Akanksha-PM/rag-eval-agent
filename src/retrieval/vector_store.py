"""In-process vector store for embedded document chunks.

Every stored chunk is tagged with its product name as metadata, so
retrieval can be scoped to a single product or to any subset of
registered products (e.g. when comparing several products at once).
Vectors and metadata persist to disk (vectors.npy / metadata.json under
Config.DATA_DIR) so the store survives app restarts.

Why both add() and rebuild(): this project's default embedder (TF-IDF,
see embeddings.py) fits its vocabulary from a product's whole corpus, so
refitting after new documents are ingested changes the *dimensionality*
and meaning of every existing vector for that product, not just appends
new ones -- old and freshly-embedded vectors are no longer comparable in
the same space. add() is for incrementally extending the store when that
isn't a concern; after a TF-IDF refit, the caller must re-embed the
product's full chunk set and call rebuild() to replace the store's
contents wholesale, rather than mixing incompatible vector spaces.
"""

import json

import numpy as np

from config import Config


def _cosine_similarities(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query_norm = np.linalg.norm(query)
    matrix_norms = np.linalg.norm(matrix, axis=1)
    denom = matrix_norms * query_norm
    dot_products = matrix @ query
    with np.errstate(divide="ignore", invalid="ignore"):
        similarities = np.where(denom > 0, dot_products / denom, 0.0)
    return similarities


class VectorStore:
    """Stores chunk embeddings alongside product-tagged metadata."""

    def __init__(self):
        self._vectors_path = Config.DATA_DIR / "vectors.npy"
        self._metadata_path = Config.DATA_DIR / "metadata.json"
        self.chunks: list[dict] = []
        self.vectors: np.ndarray = np.zeros((0, 0))
        self._load()

    def _load(self) -> None:
        if self._metadata_path.exists():
            self.chunks = json.loads(self._metadata_path.read_text())
        if self._vectors_path.exists():
            self.vectors = np.load(self._vectors_path)

    def _save(self) -> None:
        self._vectors_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self._vectors_path, self.vectors)
        self._metadata_path.write_text(json.dumps(self.chunks))

    def add(self, chunks: list[dict], vectors: list[list[float]]) -> None:
        """Append new chunks and their vectors to the store, then persist."""
        if len(chunks) != len(vectors):
            raise ValueError(
                f"chunks and vectors must have the same length "
                f"(got {len(chunks)} chunks and {len(vectors)} vectors)."
            )
        if not chunks:
            return

        new_vectors = np.array(vectors, dtype=float)
        if len(self.chunks) == 0:
            self.vectors = new_vectors
        else:
            self.vectors = np.vstack([self.vectors, new_vectors])
        self.chunks.extend(chunks)
        self._save()

    def rebuild(self, chunks: list[dict], vectors: list[list[float]]) -> None:
        """Fully replace the store's contents (e.g. after a TF-IDF refit)."""
        if len(chunks) != len(vectors):
            raise ValueError(
                f"chunks and vectors must have the same length "
                f"(got {len(chunks)} chunks and {len(vectors)} vectors)."
            )
        self.chunks = list(chunks)
        self.vectors = np.array(vectors, dtype=float)
        self._save()

    def remove_product(self, product: str) -> int:
        """Remove all chunks belonging to product. Returns the count removed."""
        keep_indices = [i for i, c in enumerate(self.chunks) if c["product"] != product]
        removed_count = len(self.chunks) - len(keep_indices)
        if removed_count == 0:
            return 0

        self.chunks = [self.chunks[i] for i in keep_indices]
        self.vectors = self.vectors[keep_indices]
        self._save()
        return removed_count

    def search(
        self, query_vector: list[float], top_k: int, products: list[str] | None = None
    ) -> list[dict]:
        """Return the top_k chunks most similar to query_vector, each with a score."""
        if len(self.chunks) == 0:
            return []

        if products is not None:
            candidate_indices = [
                i for i, chunk in enumerate(self.chunks) if chunk["product"] in products
            ]
        else:
            candidate_indices = list(range(len(self.chunks)))

        if not candidate_indices:
            return []

        query = np.array(query_vector, dtype=float)
        candidate_vectors = self.vectors[candidate_indices]
        scores = _cosine_similarities(query, candidate_vectors)

        ranked = sorted(zip(candidate_indices, scores), key=lambda pair: pair[1], reverse=True)

        results = []
        for index, score in ranked[:top_k]:
            result = dict(self.chunks[index])
            result["score"] = float(score)
            results.append(result)
        return results

    def __len__(self) -> int:
        return len(self.chunks)

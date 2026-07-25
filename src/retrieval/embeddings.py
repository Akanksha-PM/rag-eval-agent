"""Embedding providers for document chunks and queries.

get_embedder() reads Config.EMBEDDING_PROVIDER and returns the configured
EmbeddingProvider instance.
"""

import json
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from config import Config


class EmbeddingProvider(ABC):
    """Common interface for turning text into vector embeddings."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class TFIDFEmbedder(EmbeddingProvider):
    """Default embedder.

    This is a from-scratch TF-IDF implementation using only numpy, not
    scikit-learn -- deliberately, to keep the dependency footprint minimal
    and to have full ownership of the retrieval math rather than depending
    on a heavier ML library for something this small.

    The vocabulary/idf is GLOBAL, shared across every registered product
    (see fit()'s docstring for why), and persists to a single JSON file
    under Config.DATA_DIR so it survives app restarts. __init__ loads that
    persisted state automatically if it exists, so a freshly started
    process can embed queries without an explicit fit() call.
    """

    def __init__(self):
        self.vocabulary: dict[str, int] = {}
        self.idf: list[float] = []
        self._load()

    def _state_path(self) -> Path:
        return Config.DATA_DIR / "tfidf_vocab.json"

    def _load(self) -> None:
        path = self._state_path()
        if path.exists():
            state = json.loads(path.read_text())
            self.vocabulary = state["vocabulary"]
            self.idf = state["idf"]

    def _save(self) -> None:
        path = self._state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"vocabulary": self.vocabulary, "idf": self.idf}))

    def fit(self, texts: list[str]) -> None:
        """Build the vocabulary and idf scores from a corpus of texts.

        `texts` must be the combined corpus across ALL currently
        registered products, not a single product's chunks -- and this
        must be re-run on the full, updated corpus every time a new
        product is registered. The vocabulary/idf that come out of fit()
        determine the axes of the vector space every embedding lives in,
        so two embedders fit on different text produce vectors that
        aren't comparable to each other. Since this app's whole purpose is
        comparing documentation *across* products, every product's chunks
        need to share one vocabulary, or cross-product cosine similarity
        would be meaningless.
        """
        tokenized_docs = [_tokenize(text) for text in texts]
        vocabulary_tokens = sorted({token for doc in tokenized_docs for token in doc})
        self.vocabulary = {token: index for index, token in enumerate(vocabulary_tokens)}

        n_docs = len(tokenized_docs)
        doc_freq = np.zeros(len(vocabulary_tokens))
        for doc in tokenized_docs:
            for token in set(doc):
                doc_freq[self.vocabulary[token]] += 1

        # Smoothed idf (as in sklearn's TfidfVectorizer): the +1 in the
        # numerator/denominator avoids division by zero and keeps a term
        # that appears in every document from collapsing to zero weight.
        idf = np.log((1 + n_docs) / (1 + doc_freq)) + 1
        self.idf = idf.tolist()

        self._save()

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if not self.vocabulary:
            raise RuntimeError(
                "TFIDFEmbedder has no fitted vocabulary -- call fit(texts) "
                "first, with the combined corpus of every registered "
                "product."
            )

        idf = np.array(self.idf)
        vectors = []
        for text in texts:
            counts = np.zeros(len(self.vocabulary))
            for token in _tokenize(text):
                index = self.vocabulary.get(token)
                if index is not None:
                    counts[index] += 1
            vector = counts * idf
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            vectors.append(vector.tolist())
        return vectors


class VoyageEmbedder(EmbeddingProvider):
    """Stub for Voyage AI embeddings.

    Real API integration comes only if/when we decide to wire this
    provider up -- for now it exists to define the swap-in point and to
    fail loudly (rather than silently) if selected without an API key.
    """

    def __init__(self):
        self.api_key = os.getenv("VOYAGE_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "VoyageEmbedder requires a VOYAGE_API_KEY environment "
                "variable. To use the zero-dependency default instead, set "
                "EMBEDDING_PROVIDER=tfidf (see config.py's "
                "Config.EMBEDDING_PROVIDER)."
            )

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # Placeholder: the real Voyage AI HTTP call goes here. Left
        # unimplemented deliberately -- this class exists to define the
        # swap-in point, not to make network calls yet.
        raise NotImplementedError("VoyageEmbedder API integration is not implemented yet.")


def get_embedder() -> EmbeddingProvider:
    """Return the embedding provider configured via Config.EMBEDDING_PROVIDER."""
    provider = Config.EMBEDDING_PROVIDER.lower()
    if provider == "tfidf":
        return TFIDFEmbedder()
    if provider == "voyage":
        return VoyageEmbedder()
    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {Config.EMBEDDING_PROVIDER!r}")

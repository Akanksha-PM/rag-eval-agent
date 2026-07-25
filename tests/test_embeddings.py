"""Tests for src.retrieval.embeddings."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from config import Config
from src.retrieval import embeddings


def _cosine_similarity(a, b) -> float:
    a = np.array(a)
    b = np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


class _IsolatedDataDirTestCase(unittest.TestCase):
    """Patches Config.DATA_DIR to a temp dir for every test.

    TFIDFEmbedder.fit() always persists to Config.DATA_DIR now (the vocab
    is global, so there's no "no product given, skip persisting" case
    anymore) -- every test that fits an embedder needs this isolation, not
    just the ones that assert on persistence directly, or they'd write
    into this repo's real data/ directory.
    """

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._patcher = patch.object(Config, "DATA_DIR", Path(self._tmpdir.name))
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        self._tmpdir.cleanup()


class TFIDFEmbedderTestCase(_IsolatedDataDirTestCase):
    def test_fit_then_embed_returns_consistent_length_vectors(self):
        embedder = embeddings.TFIDFEmbedder()
        embedder.fit(["apple banana cherry", "banana date", "cherry date apple"])

        vectors = embedder.embed(["apple banana", "date cherry", "apple apple apple"])

        self.assertEqual(len(vectors), 3)
        lengths = {len(v) for v in vectors}
        self.assertEqual(len(lengths), 1)
        self.assertEqual(lengths.pop(), len(embedder.vocabulary))

    def test_embed_empty_list_returns_empty_list(self):
        embedder = embeddings.TFIDFEmbedder()
        embedder.fit(["some text to fit on"])
        self.assertEqual(embedder.embed([]), [])

    def test_same_topic_scores_higher_than_unrelated_topic(self):
        python_text_1 = "How do I install a Python package using pip?"
        python_text_2 = (
            "You can install Python packages with pip, the package installer for Python."
        )
        refund_text = "How do I return a product and get a refund for my purchase?"

        embedder = embeddings.TFIDFEmbedder()
        embedder.fit([python_text_1, python_text_2, refund_text])
        vec1, vec2, vec3 = embedder.embed([python_text_1, python_text_2, refund_text])

        same_topic_similarity = _cosine_similarity(vec1, vec2)
        different_topic_similarity = _cosine_similarity(vec1, vec3)

        self.assertGreater(same_topic_similarity, different_topic_similarity)

    def test_fit_persists_and_reloads_global_state(self):
        embedder = embeddings.TFIDFEmbedder()
        embedder.fit(["apple banana", "banana cherry"])

        state_path = Config.DATA_DIR / "tfidf_vocab.json"
        self.assertTrue(state_path.exists())

        # A fresh instance should load the persisted vocab automatically on
        # __init__, with no explicit fit() call.
        reloaded = embeddings.TFIDFEmbedder()
        self.assertEqual(reloaded.vocabulary, embedder.vocabulary)
        self.assertEqual(reloaded.idf, embedder.idf)

    def test_shared_vocab_makes_vectors_comparable_across_products(self):
        product_a_chunks = [
            "Install the package by running pip install our-sdk.",
            "Configure your API key in the our-sdk settings file.",
        ]
        product_b_chunks = [
            "Submit a refund request through the online returns portal.",
            "Refunds are processed within five business days of approval.",
        ]

        embedder = embeddings.TFIDFEmbedder()
        embedder.fit(product_a_chunks + product_b_chunks)

        query_vector = embedder.embed(["How do I install the package with pip?"])[0]
        product_a_vectors = embedder.embed(product_a_chunks)
        product_b_vectors = embedder.embed(product_b_chunks)

        best_a_score = max(_cosine_similarity(query_vector, v) for v in product_a_vectors)
        best_b_score = max(_cosine_similarity(query_vector, v) for v in product_b_vectors)

        self.assertGreater(best_a_score, best_b_score)


class GetEmbedderTestCase(_IsolatedDataDirTestCase):
    def test_returns_tfidf_by_default(self):
        with patch.object(Config, "EMBEDDING_PROVIDER", "tfidf"):
            embedder = embeddings.get_embedder()
        self.assertIsInstance(embedder, embeddings.TFIDFEmbedder)

    def test_raises_clear_error_for_voyage_without_api_key(self):
        env_without_key = os.environ.copy()
        env_without_key.pop("VOYAGE_API_KEY", None)

        with patch.object(Config, "EMBEDDING_PROVIDER", "voyage"), patch.dict(
            os.environ, env_without_key, clear=True
        ):
            with self.assertRaises(RuntimeError):
                embeddings.get_embedder()


if __name__ == "__main__":
    unittest.main()

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


class TFIDFEmbedderTestCase(unittest.TestCase):
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

    def test_fit_persists_and_reloads_state_for_product(self):
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)

        with patch.object(Config, "DATA_DIR", Path(tmp_dir.name)):
            embedder = embeddings.TFIDFEmbedder(product="Test Product")
            embedder.fit(["apple banana", "banana cherry"])

            state_path = Path(tmp_dir.name) / "embeddings" / "test_product_tfidf.json"
            self.assertTrue(state_path.exists())

            reloaded = embeddings.TFIDFEmbedder(product="Test Product")
            self.assertEqual(reloaded.vocabulary, embedder.vocabulary)
            self.assertEqual(reloaded.idf, embedder.idf)


class GetEmbedderTestCase(unittest.TestCase):
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

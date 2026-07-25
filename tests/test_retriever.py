"""Tests for src.retrieval.retriever."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from config import Config
from src.retrieval import retriever
from src.retrieval.vector_store import VectorStore


class _FakeEmbedder:
    """Deterministic stand-in for a real embedder, so these tests exercise
    retriever's wiring rather than TF-IDF's math (already covered in
    test_embeddings.py).
    """

    def __init__(self, vector):
        self._vector = vector

    def embed(self, texts):
        return [self._vector for _ in texts]


def _make_chunk(chunk_id, product):
    return {
        "id": chunk_id,
        "product": product,
        "source": "source.txt",
        "chunk_index": 0,
        "text": "text",
    }


class RetrieveTestCase(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._patcher = patch.object(Config, "DATA_DIR", Path(self._tmpdir.name))
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        self._tmpdir.cleanup()

    def _seed_store(self):
        store = VectorStore()
        chunks = [
            _make_chunk("a_0", "ProductA"),
            _make_chunk("b_0", "ProductB"),
            _make_chunk("a_1", "ProductA"),
        ]
        vectors = [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]
        store.add(chunks, vectors)

    def test_retrieve_with_no_products_filter_searches_everything(self):
        self._seed_store()
        with patch(
            "src.retrieval.retriever.get_embedder",
            return_value=_FakeEmbedder([1.0, 0.0]),
        ):
            results = retriever.retrieve("anything", products=None, top_k=10)

        self.assertEqual({r["id"] for r in results}, {"a_0", "b_0", "a_1"})

    def test_retrieve_with_products_filter_only_returns_matching_products(self):
        self._seed_store()
        with patch(
            "src.retrieval.retriever.get_embedder",
            return_value=_FakeEmbedder([1.0, 0.0]),
        ):
            results = retriever.retrieve("anything", products=["ProductA"], top_k=10)

        self.assertTrue(results)
        self.assertTrue(all(r["product"] == "ProductA" for r in results))

    def test_retrieve_empty_products_list_treated_as_no_filter(self):
        self._seed_store()
        with patch(
            "src.retrieval.retriever.get_embedder",
            return_value=_FakeEmbedder([1.0, 0.0]),
        ):
            results = retriever.retrieve("anything", products=[], top_k=10)

        self.assertEqual({r["id"] for r in results}, {"a_0", "b_0", "a_1"})

    def test_retrieve_respects_custom_top_k(self):
        self._seed_store()
        with patch(
            "src.retrieval.retriever.get_embedder",
            return_value=_FakeEmbedder([1.0, 0.0]),
        ):
            results = retriever.retrieve("anything", top_k=2)

        self.assertEqual(len(results), 2)

    def test_retrieve_uses_config_top_k_by_default(self):
        self._seed_store()
        with patch.object(Config, "TOP_K", 1), patch(
            "src.retrieval.retriever.get_embedder",
            return_value=_FakeEmbedder([1.0, 0.0]),
        ):
            results = retriever.retrieve("anything")

        self.assertEqual(len(results), 1)

    def test_retrieve_against_empty_store_returns_empty_list(self):
        with patch(
            "src.retrieval.retriever.get_embedder",
            return_value=_FakeEmbedder([1.0, 0.0]),
        ) as mock_get_embedder:
            results = retriever.retrieve("anything")

        self.assertEqual(results, [])
        mock_get_embedder.assert_not_called()


if __name__ == "__main__":
    unittest.main()

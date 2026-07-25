"""Tests for src.retrieval.vector_store."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from config import Config
from src.retrieval.vector_store import VectorStore


def _make_chunk(chunk_id, product, source="source.txt", chunk_index=0, text="text"):
    return {
        "id": chunk_id,
        "product": product,
        "source": source,
        "chunk_index": chunk_index,
        "text": text,
    }


class VectorStoreTestCase(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._patcher = patch.object(Config, "DATA_DIR", Path(self._tmpdir.name))
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        self._tmpdir.cleanup()

    def test_add_then_search_returns_most_similar_first(self):
        store = VectorStore()
        chunks = [
            _make_chunk("a_0", "ProductA", text="about apples"),
            _make_chunk("a_1", "ProductA", text="about oranges"),
        ]
        vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        store.add(chunks, vectors)

        results = store.search([1.0, 0.0, 0.0], top_k=2)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["id"], "a_0")
        self.assertGreater(results[0]["score"], results[1]["score"])

    def test_search_respects_products_filter(self):
        store = VectorStore()
        chunks = [
            _make_chunk("a_0", "ProductA"),
            _make_chunk("b_0", "ProductB"),
        ]
        # ProductB's vector is the closer match to the query, but it should
        # be excluded from results by the products filter.
        vectors = [[0.5, 0.5, 0.0], [1.0, 0.0, 0.0]]
        store.add(chunks, vectors)

        results = store.search([1.0, 0.0, 0.0], top_k=5, products=["ProductA"])

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "a_0")

    def test_remove_product_removes_right_chunks_only(self):
        store = VectorStore()
        chunks = [
            _make_chunk("a_0", "ProductA"),
            _make_chunk("b_0", "ProductB"),
            _make_chunk("a_1", "ProductA"),
        ]
        vectors = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        store.add(chunks, vectors)

        removed = store.remove_product("ProductA")

        self.assertEqual(removed, 2)
        self.assertEqual(len(store), 1)
        self.assertEqual([c["id"] for c in store.chunks], ["b_0"])

    def test_search_on_empty_store_returns_empty_list(self):
        store = VectorStore()
        self.assertEqual(store.search([1.0, 0.0], top_k=5), [])

    def test_rebuild_replaces_prior_contents(self):
        store = VectorStore()
        store.add([_make_chunk("a_0", "ProductA")], [[1.0, 0.0]])

        store.rebuild([_make_chunk("b_0", "ProductB")], [[0.0, 1.0]])

        self.assertEqual(len(store), 1)
        self.assertEqual(store.chunks[0]["id"], "b_0")

    def test_persists_and_reloads_across_instances(self):
        store = VectorStore()
        chunks = [_make_chunk("a_0", "ProductA"), _make_chunk("a_1", "ProductA")]
        vectors = [[1.0, 0.0], [0.0, 1.0]]
        store.add(chunks, vectors)

        reloaded = VectorStore()

        self.assertEqual(len(reloaded), 2)
        self.assertEqual([c["id"] for c in reloaded.chunks], ["a_0", "a_1"])
        results = reloaded.search([1.0, 0.0], top_k=1)
        self.assertEqual(results[0]["id"], "a_0")

    def test_add_raises_on_length_mismatch(self):
        store = VectorStore()
        with self.assertRaises(ValueError):
            store.add([_make_chunk("a_0", "ProductA")], [[1.0, 0.0], [0.0, 1.0]])


if __name__ == "__main__":
    unittest.main()

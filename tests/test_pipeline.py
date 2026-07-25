"""Tests for src.ingestion.pipeline."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from config import Config
from src.ingestion import pipeline, registry
from src.retrieval.vector_store import VectorStore


class IngestProductTestCase(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._tmp_path = Path(self._tmpdir.name)

        self._data_patcher = patch.object(Config, "DATA_DIR", self._tmp_path / "data")
        self._registry_patcher = patch.object(
            Config, "REGISTRY_PATH", self._tmp_path / "data" / "registry.json"
        )
        self._data_patcher.start()
        self._registry_patcher.start()

    def tearDown(self):
        self._data_patcher.stop()
        self._registry_patcher.stop()
        self._tmpdir.cleanup()

    def _write_fixture(self, filename: str, content: str) -> str:
        path = self._tmp_path / filename
        path.write_text(content)
        return str(path)

    def test_ingest_from_txt_file_sets_ready_status(self):
        fixture_path = self._write_fixture(
            "product_a.txt", "Documentation for Product A. " * 20
        )

        result = pipeline.ingest_product("Product A", "file", fixture_path)

        self.assertEqual(result["status"], "ready")
        self.assertGreater(result["total_chunks"], 0)
        self.assertEqual(result["chunks_added"], result["total_chunks"])

        entry = registry.get_product("Product A")
        self.assertEqual(entry["status"], "ready")
        self.assertGreater(entry["chunk_count"], 0)

    def test_ingest_second_product_preserves_first(self):
        fixture_a = self._write_fixture(
            "product_a.txt", "Product A documentation about installing the SDK."
        )
        fixture_b = self._write_fixture(
            "product_b.txt", "Product B documentation about requesting refunds."
        )

        pipeline.ingest_product("Product A", "file", fixture_a)
        result_b = pipeline.ingest_product("Product B", "file", fixture_b)

        store = VectorStore()
        product_a_chunks = store.get_all(product="Product A")
        product_b_chunks = store.get_all(product="Product B")

        self.assertGreater(len(product_a_chunks), 0)
        self.assertEqual(len(product_b_chunks), result_b["total_chunks"])
        self.assertEqual(
            len(store.get_all()), len(product_a_chunks) + len(product_b_chunks)
        )

    def test_ingest_same_product_twice_accumulates_without_id_collisions(self):
        fixture_1 = self._write_fixture(
            "source1.txt", "First source content about installing the package."
        )
        fixture_2 = self._write_fixture(
            "source2.txt", "Second source content about configuring the package."
        )

        result_1 = pipeline.ingest_product("Product A", "file", fixture_1)
        result_2 = pipeline.ingest_product("Product A", "file", fixture_2)

        self.assertEqual(
            result_2["total_chunks"], result_1["total_chunks"] + result_2["chunks_added"]
        )

        store = VectorStore()
        product_chunks = store.get_all(product="Product A")
        ids = [c["id"] for c in product_chunks]

        self.assertEqual(len(product_chunks), result_2["total_chunks"])
        self.assertEqual(len(ids), len(set(ids)))

    def test_invalid_source_type_raises_before_touching_registry(self):
        with self.assertRaises(ValueError):
            pipeline.ingest_product("Product A", "ftp", "irrelevant")

        self.assertIsNone(registry.get_product("Product A"))

    def test_ingest_from_url_uses_load_from_url(self):
        with patch(
            "src.ingestion.pipeline.load_from_url",
            return_value="Mocked page content describing the product's features.",
        ) as mock_load_from_url:
            result = pipeline.ingest_product("Product A", "url", "https://example.com/docs")

        mock_load_from_url.assert_called_once_with("https://example.com/docs")
        self.assertEqual(result["status"], "ready")
        self.assertGreater(result["total_chunks"], 0)


if __name__ == "__main__":
    unittest.main()

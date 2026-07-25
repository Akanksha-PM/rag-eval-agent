"""Tests for src.ingestion.registry."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from config import Config
from src.ingestion import registry


class RegistryTestCase(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.registry_path = Path(self._tmpdir.name) / "registry.json"
        self._patcher = patch.object(Config, "REGISTRY_PATH", self.registry_path)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        self._tmpdir.cleanup()

    def test_add_product_creates_new_pending_entry(self):
        entry = registry.add_product("acme-docs", "url", "https://acme.example.com/docs")
        self.assertEqual(entry["name"], "acme-docs")
        self.assertEqual(entry["status"], "pending")
        self.assertEqual(entry["chunk_count"], 0)
        self.assertEqual(
            entry["sources"], [{"type": "url", "value": "https://acme.example.com/docs"}]
        )

    def test_add_product_appends_additional_source(self):
        registry.add_product("acme-docs", "url", "https://acme.example.com/docs")
        entry = registry.add_product("acme-docs", "file", "/tmp/acme.pdf")
        self.assertEqual(len(entry["sources"]), 2)
        self.assertEqual(entry["sources"][1], {"type": "file", "value": "/tmp/acme.pdf"})

    def test_list_products_returns_all_entries_with_names(self):
        registry.add_product("acme-docs", "url", "https://acme.example.com/docs")
        registry.add_product("other-product", "file", "/tmp/other.txt")
        names = {p["name"] for p in registry.list_products()}
        self.assertEqual(names, {"acme-docs", "other-product"})

    def test_get_product_returns_entry(self):
        registry.add_product("acme-docs", "url", "https://acme.example.com/docs")
        entry = registry.get_product("acme-docs")
        self.assertEqual(entry["name"], "acme-docs")

    def test_get_product_returns_none_for_unknown(self):
        self.assertIsNone(registry.get_product("does-not-exist"))

    def test_remove_product_removes_existing_entry(self):
        registry.add_product("acme-docs", "url", "https://acme.example.com/docs")
        self.assertTrue(registry.remove_product("acme-docs"))
        self.assertIsNone(registry.get_product("acme-docs"))

    def test_remove_product_returns_false_for_unknown(self):
        self.assertFalse(registry.remove_product("does-not-exist"))

    def test_update_chunk_count_flips_pending_to_ready(self):
        registry.add_product("acme-docs", "url", "https://acme.example.com/docs")
        self.assertEqual(registry.get_product("acme-docs")["status"], "pending")

        registry.update_chunk_count("acme-docs", 42)

        entry = registry.get_product("acme-docs")
        self.assertEqual(entry["status"], "ready")
        self.assertEqual(entry["chunk_count"], 42)

    def test_registry_file_created_when_missing(self):
        nested_path = Path(self._tmpdir.name) / "nested" / "registry.json"
        with patch.object(Config, "REGISTRY_PATH", nested_path):
            self.assertEqual(registry.list_products(), [])
            self.assertTrue(nested_path.exists())


if __name__ == "__main__":
    unittest.main()

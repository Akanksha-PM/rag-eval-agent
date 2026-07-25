"""Integration tests for src.api.routes.products, via FastAPI's TestClient."""

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from config import Config
from src.api.app import app
from src.retrieval.retriever import retrieve


class ProductsRouteTestCase(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        tmp_path = Path(self._tmpdir.name)

        self._data_patcher = patch.object(Config, "DATA_DIR", tmp_path / "data")
        self._registry_patcher = patch.object(
            Config, "REGISTRY_PATH", tmp_path / "data" / "registry.json"
        )
        self._data_patcher.start()
        self._registry_patcher.start()

        self.client = TestClient(app)

    def tearDown(self):
        self._data_patcher.stop()
        self._registry_patcher.stop()
        self._tmpdir.cleanup()

    def _upload_file(self, name: str, filename: str, content: str):
        return self.client.post(
            f"/products/{name}/ingest",
            files={"file": (filename, io.BytesIO(content.encode()), "text/plain")},
        )

    def test_ingest_with_file_upload_succeeds_and_appears_in_list(self):
        response = self._upload_file(
            "ProductA", "notes.txt", "Documentation about installing the SDK. " * 10
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["product"], "ProductA")
        self.assertEqual(body["status"], "ready")
        self.assertGreater(body["total_chunks"], 0)

        list_response = self.client.get("/products")
        self.assertEqual(list_response.status_code, 200)
        names = {p["name"] for p in list_response.json()}
        self.assertIn("ProductA", names)

    def test_ingest_with_neither_file_nor_url_returns_400(self):
        response = self.client.post("/products/ProductA/ingest")
        self.assertEqual(response.status_code, 400)

    def test_ingest_with_both_file_and_url_returns_400(self):
        response = self.client.post(
            "/products/ProductA/ingest",
            data={"url": "https://example.com/docs"},
            files={"file": ("notes.txt", io.BytesIO(b"content"), "text/plain")},
        )
        self.assertEqual(response.status_code, 400)

    def test_ingest_with_file_and_empty_string_url_succeeds_as_file_only(self):
        # Mirrors what Swagger UI actually sends for an unfilled optional
        # form field: an empty string, not an omitted field.
        response = self.client.post(
            "/products/ProductA/ingest",
            data={"url": ""},
            files={
                "file": (
                    "notes.txt",
                    io.BytesIO(b"Documentation about installing the SDK. " * 10),
                    "text/plain",
                )
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["product"], "ProductA")
        self.assertEqual(body["status"], "ready")

    def test_delete_removes_product_and_its_chunks_are_no_longer_searchable(self):
        self._upload_file(
            "ProductA", "notes.txt", "Documentation about installing the SDK. " * 10
        )

        delete_response = self.client.delete("/products/ProductA")
        self.assertEqual(delete_response.status_code, 200)
        self.assertGreater(delete_response.json()["chunks_removed"], 0)

        results = retrieve("installing the SDK", products=["ProductA"])
        self.assertEqual(results, [])

    def test_delete_nonexistent_product_returns_404(self):
        response = self.client.delete("/products/does-not-exist")
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()

"""Integration tests for src.api.routes.query, via FastAPI's TestClient."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from config import Config
from src.api.app import app
from src.ingestion import pipeline


class QueryRouteTestCase(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._tmp_path = Path(self._tmpdir.name)

        self._data_patcher = patch.object(Config, "DATA_DIR", self._tmp_path / "data")
        self._registry_patcher = patch.object(
            Config, "REGISTRY_PATH", self._tmp_path / "data" / "registry.json"
        )
        self._llm_patcher = patch.object(Config, "LLM_PROVIDER", "mock")
        self._data_patcher.start()
        self._registry_patcher.start()
        self._llm_patcher.start()

        self.client = TestClient(app)

    def tearDown(self):
        self._llm_patcher.stop()
        self._registry_patcher.stop()
        self._data_patcher.stop()
        self._tmpdir.cleanup()

    def _write_fixture(self, filename: str, content: str) -> str:
        path = self._tmp_path / filename
        path.write_text(content)
        return str(path)

    def test_query_matching_question_returns_answer_and_trace(self):
        fixture = self._write_fixture(
            "product_a.txt",
            "Product A supports single sign-on via SAML and OAuth2. "
            "Configuration is done in the admin settings panel.",
        )
        pipeline.ingest_product("ProductA", "file", fixture)

        response = self.client.post(
            "/query", json={"question": "How does ProductA handle single sign-on?"}
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["answer"])
        self.assertEqual(len(body["trace"]), 4)

    def test_query_with_empty_question_returns_400(self):
        response = self.client.post("/query", json={"question": "   "})
        self.assertEqual(response.status_code, 400)

    def test_query_with_unknown_product_returns_400_listing_valid_products(self):
        fixture = self._write_fixture(
            "product_a.txt", "Product A documentation about installing the SDK."
        )
        pipeline.ingest_product("ProductA", "file", fixture)

        response = self.client.post(
            "/query", json={"question": "anything", "products": ["DoesNotExist"]}
        )

        self.assertEqual(response.status_code, 400)
        detail = response.json()["detail"]
        self.assertIn("DoesNotExist", detail)
        self.assertIn("ProductA", detail)

    def test_query_without_products_filter_searches_everything(self):
        fixture_a = self._write_fixture(
            "product_a.txt",
            "Product A documentation about installing the SDK using pip.",
        )
        fixture_b = self._write_fixture(
            "product_b.txt",
            "Product B documentation about requesting a refund for a purchase.",
        )
        pipeline.ingest_product("ProductA", "file", fixture_a)
        pipeline.ingest_product("ProductB", "file", fixture_b)

        response = self.client.post(
            "/query", json={"question": "How do I install the SDK using pip?"}
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        products_in_chunks = {chunk["product"] for chunk in body["chunks_used"]}
        self.assertIn("ProductA", products_in_chunks)


if __name__ == "__main__":
    unittest.main()

"""Integration tests for src.api.routes.golden, via FastAPI's TestClient."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from config import Config
from src.api.app import app


class GoldenRouteTestCase(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._patcher = patch.object(Config, "GOLDEN_DATASET_DIR", Path(self._tmpdir.name))
        self._patcher.start()

        self.client = TestClient(app)

    def tearDown(self):
        self._patcher.stop()
        self._tmpdir.cleanup()

    def test_create_dataset_then_list_shows_it(self):
        response = self.client.post("/golden-datasets", json={"name": "SupportQA"})
        self.assertEqual(response.status_code, 201)

        list_response = self.client.get("/golden-datasets")
        self.assertEqual(list_response.status_code, 200)
        names = {d["name"] for d in list_response.json()}
        self.assertIn("SupportQA", names)

    def test_post_item_then_get_dataset_includes_it(self):
        self.client.post("/golden-datasets", json={"name": "SupportQA"})

        item_response = self.client.post(
            "/golden-datasets/SupportQA/items",
            json={
                "question": "How do I reset my password?",
                "expected_answer": "Use the reset link on the login page.",
            },
        )
        self.assertEqual(item_response.status_code, 201)

        get_response = self.client.get("/golden-datasets/SupportQA")
        self.assertEqual(get_response.status_code, 200)
        items = get_response.json()["items"]
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["question"], "How do I reset my password?")

    def test_create_duplicate_dataset_returns_400(self):
        self.client.post("/golden-datasets", json={"name": "SupportQA"})

        response = self.client.post("/golden-datasets", json={"name": "SupportQA"})

        self.assertEqual(response.status_code, 400)

    def test_add_item_to_nonexistent_dataset_returns_400_with_clear_message(self):
        response = self.client.post(
            "/golden-datasets/DoesNotExist/items",
            json={"question": "Q?", "expected_answer": "A."},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("DoesNotExist", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()

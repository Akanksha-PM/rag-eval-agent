"""Tests for src.eval.golden_dataset."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from config import Config
from src.eval import golden_dataset


class GoldenDatasetTestCase(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._patcher = patch.object(Config, "GOLDEN_DATASET_DIR", Path(self._tmpdir.name))
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        self._tmpdir.cleanup()

    def test_create_then_list_shows_it_with_zero_items(self):
        golden_dataset.create_dataset("Support QA")

        datasets = golden_dataset.list_datasets()

        self.assertEqual(len(datasets), 1)
        self.assertEqual(datasets[0]["name"], "Support QA")
        self.assertEqual(datasets[0]["item_count"], 0)
        self.assertIn("created_at", datasets[0])

    def test_add_item_then_get_dataset_shows_item_with_correct_id(self):
        golden_dataset.create_dataset("Support QA")

        item = golden_dataset.add_item(
            "Support QA",
            "How do I reset my password?",
            "Use the reset link on the login page.",
        )

        self.assertEqual(item["id"], "support_qa_0")

        dataset = golden_dataset.get_dataset("Support QA")
        self.assertEqual(len(dataset["items"]), 1)
        self.assertEqual(dataset["items"][0]["id"], "support_qa_0")
        self.assertEqual(dataset["items"][0]["question"], "How do I reset my password?")

    def test_add_item_to_nonexistent_dataset_raises_value_error(self):
        with self.assertRaises(ValueError):
            golden_dataset.add_item("Does Not Exist", "Q?", "A.")

    def test_add_item_with_empty_question_raises_value_error(self):
        golden_dataset.create_dataset("Support QA")
        with self.assertRaises(ValueError):
            golden_dataset.add_item("Support QA", "   ", "A valid answer.")

    def test_delete_item_removes_right_item_and_leaves_others(self):
        golden_dataset.create_dataset("Support QA")
        golden_dataset.add_item("Support QA", "Q1?", "A1.")
        item2 = golden_dataset.add_item("Support QA", "Q2?", "A2.")
        golden_dataset.add_item("Support QA", "Q3?", "A3.")

        removed = golden_dataset.delete_item("Support QA", item2["id"])

        self.assertTrue(removed)
        dataset = golden_dataset.get_dataset("Support QA")
        remaining_ids = [item["id"] for item in dataset["items"]]
        self.assertNotIn(item2["id"], remaining_ids)
        self.assertEqual(len(remaining_ids), 2)

    def test_create_dataset_with_existing_name_raises_value_error(self):
        golden_dataset.create_dataset("Support QA")
        with self.assertRaises(ValueError):
            golden_dataset.create_dataset("Support QA")


if __name__ == "__main__":
    unittest.main()

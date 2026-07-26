"""Tests for src.agent.eval_agent."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from config import Config
from src.agent import eval_agent
from src.retrieval.vector_store import VectorStore


class _FakeEmbedder:
    """Deterministic stand-in for a real embedder (same pattern as
    test_retriever.py) so these tests don't depend on TF-IDF fitting.
    """

    def __init__(self, vector):
        self._vector = vector

    def embed(self, texts):
        return [self._vector for _ in texts]


def _make_chunk(chunk_id, product, source, chunk_index, text):
    return {
        "id": chunk_id,
        "product": product,
        "source": source,
        "chunk_index": chunk_index,
        "text": text,
    }


class EvalAgentTestCase(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._data_patcher = patch.object(Config, "DATA_DIR", Path(self._tmpdir.name))
        self._llm_patcher = patch.object(Config, "LLM_PROVIDER", "mock")
        self._data_patcher.start()
        self._llm_patcher.start()

        store = VectorStore()
        store.add(
            [
                _make_chunk(
                    "producta_0",
                    "ProductA",
                    "docs.txt",
                    0,
                    "ProductA supports single sign-on via SAML and OAuth2. "
                    "Configuration is done in the admin settings panel.",
                )
            ],
            [[1.0, 0.0]],
        )

        self._embedder_patcher = patch(
            "src.retrieval.retriever.get_embedder",
            return_value=_FakeEmbedder([1.0, 0.0]),
        )
        self._embedder_patcher.start()

    def tearDown(self):
        self._embedder_patcher.stop()
        self._llm_patcher.stop()
        self._data_patcher.stop()
        self._tmpdir.cleanup()

    def test_run_agent_with_matching_content_returns_answer_with_sources(self):
        result = eval_agent.run_agent("How does ProductA handle single sign-on?")

        self.assertTrue(result["final_answer"])
        self.assertIn("Sources:", result["final_answer"])
        self.assertIn("ProductA", result["final_answer"])

    def test_run_agent_with_non_matching_product_filter_returns_no_info(self):
        result = eval_agent.run_agent("How does SSO work?", products=["ProductB"])

        self.assertIn("No relevant information was found", result["final_answer"])
        self.assertIsNone(result["judge_result"]["faithfulness_score"])

    def test_trace_has_four_entries_in_correct_order(self):
        result = eval_agent.run_agent("How does ProductA handle single sign-on?")

        self.assertEqual(len(result["trace"]), 4)
        self.assertEqual(
            [entry["node"] for entry in result["trace"]],
            ["retrieve", "generate", "judge", "format_citations"],
        )
        for entry in result["trace"]:
            self.assertIn("summary", entry)
            self.assertIn("timestamp", entry)

    def test_judge_node_produces_parseable_result_with_expected_keys(self):
        result = eval_agent.run_agent("How does ProductA handle single sign-on?")

        judge_result = result["judge_result"]
        self.assertIn("faithfulness_score", judge_result)
        self.assertIn("reasoning", judge_result)
        self.assertIn("hallucinated_claims", judge_result)
        self.assertIsInstance(judge_result["hallucinated_claims"], list)


if __name__ == "__main__":
    unittest.main()

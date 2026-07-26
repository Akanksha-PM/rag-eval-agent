"""Tests for src.llm.base, src.llm.mock_provider, and src.llm.anthropic_provider."""

import json
import os
import unittest
from unittest.mock import patch

from config import Config
from src.llm.anthropic_provider import AnthropicProvider
from src.llm.base import get_llm_provider
from src.llm.mock_provider import MockProvider


class MockProviderTestCase(unittest.TestCase):
    def test_generate_with_context_returns_text_from_context(self):
        prompt = (
            "CONTEXT:\n"
            "Python is a general-purpose programming language. It was "
            "created by Guido van Rossum. It emphasizes code readability.\n\n"
            "QUESTION:\nWho created Python?"
        )

        result = MockProvider().generate(prompt)

        self.assertTrue(result.startswith("[Mock response - extractive, not generated]:"))
        self.assertIn("Guido van Rossum", result)

    def test_generate_judge_prompt_returns_valid_json_with_expected_keys(self):
        prompt = (
            "EVALUATE_FAITHFULNESS\n\n"
            "CONTEXT:\n"
            "Python was created by Guido van Rossum in 1991.\n\n"
            "ANSWER:\n"
            "Python was created by Guido van Rossum.\n"
        )

        result = MockProvider().generate(prompt)
        data = json.loads(result)

        self.assertIn("faithfulness_score", data)
        self.assertIn("reasoning", data)
        self.assertIn("hallucinated_claims", data)
        self.assertIsInstance(data["faithfulness_score"], int)
        self.assertTrue(1 <= data["faithfulness_score"] <= 5)
        self.assertEqual(data["hallucinated_claims"], [])

    def test_generate_without_context_returns_fallback_message(self):
        result = MockProvider().generate("QUESTION:\nWho created Python?")
        self.assertEqual(result, "[Mock response]: No context provided to answer from.")


class AnthropicProviderTestCase(unittest.TestCase):
    def test_raises_value_error_when_api_key_missing(self):
        env_without_key = os.environ.copy()
        env_without_key.pop("ANTHROPIC_API_KEY", None)

        with patch.dict(os.environ, env_without_key, clear=True):
            with self.assertRaises(ValueError):
                AnthropicProvider()


class GetLLMProviderTestCase(unittest.TestCase):
    def test_returns_mock_provider_by_default(self):
        with patch.object(Config, "LLM_PROVIDER", "mock"):
            provider = get_llm_provider()
        self.assertIsInstance(provider, MockProvider)

    def test_returns_anthropic_provider_when_configured(self):
        env_with_key = os.environ.copy()
        env_with_key["ANTHROPIC_API_KEY"] = "test-key"

        with patch.object(Config, "LLM_PROVIDER", "anthropic"), patch.dict(
            os.environ, env_with_key
        ):
            provider = get_llm_provider()

        self.assertIsInstance(provider, AnthropicProvider)


if __name__ == "__main__":
    unittest.main()

"""Deterministic mock LLM provider, used when no API key is configured."""

from src.llm.base import LLMProvider


class MockProvider(LLMProvider):
    """Returns canned/deterministic responses without calling any external API."""

    def generate(self, prompt, **kwargs):
        raise NotImplementedError

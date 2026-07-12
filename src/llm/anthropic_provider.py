"""LLM provider backed by the Anthropic API.

Used when ANTHROPIC_API_KEY is set in the environment; otherwise the app
falls back to MockProvider (see mock_provider.py).
"""

from src.llm.base import LLMProvider


class AnthropicProvider(LLMProvider):
    """Generates completions via the Anthropic Claude API."""

    def generate(self, prompt, **kwargs):
        raise NotImplementedError

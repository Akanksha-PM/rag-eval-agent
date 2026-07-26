"""LLM provider backed by the Anthropic API.

Used when ANTHROPIC_API_KEY is set in the environment; otherwise the app
falls back to MockProvider (see mock_provider.py).
"""

import os

import anthropic

from src.llm.base import LLMProvider

DEFAULT_MODEL = "claude-sonnet-5"


class AnthropicProvider(LLMProvider):
    """Generates completions via the Anthropic Claude API."""

    def __init__(self):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "AnthropicProvider requires ANTHROPIC_API_KEY to be set -- "
                "see .env.example. To run without an API key, use the mock "
                'provider instead (Config.LLM_PROVIDER="mock").'
            )
        self.model = os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL)
        self._client = anthropic.Anthropic(api_key=api_key)

    def generate(self, prompt: str, system: str = "") -> str:
        kwargs = {"system": system} if system else {}
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )
        except Exception as exc:
            raise RuntimeError(f"Anthropic API call failed: {exc}") from exc

        return next((block.text for block in response.content if block.type == "text"), "")

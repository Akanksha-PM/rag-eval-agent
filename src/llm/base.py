"""Abstract base class all LLM providers implement, plus the provider factory."""

from abc import ABC, abstractmethod

from config import Config


class LLMProvider(ABC):
    """Common interface for generating text from a prompt."""

    @abstractmethod
    def generate(self, prompt: str, system: str = "") -> str:
        """Return a generated text completion for the given prompt."""
        raise NotImplementedError


def get_llm_provider() -> LLMProvider:
    """Return the LLM provider configured via Config.LLM_PROVIDER."""
    # Imported lazily to avoid a circular import: both provider modules
    # import LLMProvider from this module at load time.
    from src.llm.anthropic_provider import AnthropicProvider
    from src.llm.mock_provider import MockProvider

    provider = Config.LLM_PROVIDER.lower()
    if provider == "mock":
        return MockProvider()
    if provider == "anthropic":
        return AnthropicProvider()
    raise ValueError(f"Unknown LLM_PROVIDER: {Config.LLM_PROVIDER!r}")

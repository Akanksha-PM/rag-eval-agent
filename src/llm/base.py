"""Abstract base class all LLM providers implement."""

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Common interface for generating text from a prompt."""

    @abstractmethod
    def generate(self, prompt, **kwargs):
        """Return a generated text completion for the given prompt."""
        raise NotImplementedError

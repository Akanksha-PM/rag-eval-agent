"""Deterministic mock LLM provider.

Exists so the whole app runs end-to-end with zero API keys -- ingestion,
retrieval, and the eval harness all work against MockProvider without any
network calls. Its answers are extractive excerpts and heuristic scores,
not real generations; swap in AnthropicProvider (set ANTHROPIC_API_KEY and
Config.LLM_PROVIDER="anthropic") for actual answer quality.
"""

import json
import re

from src.llm.base import LLMProvider

_JUDGE_MARKER = "EVALUATE_FAITHFULNESS"


def _extract_section(prompt: str, label: str) -> str:
    """Return the text following `{label}:` up to the next ALL_CAPS: label or EOF."""
    match = re.search(rf"{label}:\s*(.*?)(?=\n[A-Z_]+:|\Z)", prompt, re.DOTALL)
    return match.group(1).strip() if match else ""


class MockProvider(LLMProvider):
    """Returns canned/deterministic responses without calling any external API."""

    def generate(self, prompt: str, system: str = "") -> str:
        if _JUDGE_MARKER in prompt:
            return self._judge_response(prompt)
        return self._generate_response(prompt)

    def _judge_response(self, prompt: str) -> str:
        context = _extract_section(prompt, "CONTEXT")
        answer = _extract_section(prompt, "ANSWER")

        context_words = set(context.lower().split())
        answer_words = answer.lower().split()

        if answer_words:
            overlap_ratio = sum(1 for word in answer_words if word in context_words) / len(
                answer_words
            )
        else:
            overlap_ratio = 0.0

        # Map the 0-1 overlap ratio onto a 1-5 score.
        score = max(1, min(5, round(overlap_ratio * 4) + 1))

        return json.dumps(
            {
                "faithfulness_score": score,
                "reasoning": (
                    f"Mock provider: {round(overlap_ratio * 100)}% of answer "
                    "words also appear in the provided context."
                ),
                "hallucinated_claims": [],
            }
        )

    def _generate_response(self, prompt: str) -> str:
        context = _extract_section(prompt, "CONTEXT")
        if not context:
            return "[Mock response]: No context provided to answer from."

        sentences = re.split(r"(?<=[.!?])\s+", context.strip())
        excerpt = " ".join(sentences[:3]).strip()
        return f"[Mock response - extractive, not generated]: {excerpt}"

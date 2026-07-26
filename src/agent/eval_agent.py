"""LangGraph-based evaluation agent.

Builds a graph with nodes for:
- retrieve: fetch relevant chunks for the question, scoped to the given
  product names (any number of products, not a fixed pair).
- generate: produce a draft answer grounded in the retrieved chunks.
- judge: critique/score the draft answer against the retrieved evidence.
- format_citations: attach citations back to source chunks/products.

The agent accepts a list of product names of any length plus a question,
and returns a grounded, citation-backed answer. run_agent() is the single
entry point everything else (the API, the eval harness) should call.
"""

import json
import re
from datetime import datetime, timezone
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from src.llm.base import get_llm_provider
from src.retrieval.retriever import retrieve

_NO_CONTEXT_MESSAGE = (
    "No relevant information was found in the registered documentation "
    "to answer this question."
)


class AgentState(TypedDict):
    question: str
    products: list[str] | None
    retrieved_chunks: list[dict]
    draft_answer: str
    judge_result: dict
    final_answer: str
    trace: list[dict]


def _trace_entry(node: str, summary: str) -> dict:
    return {
        "node": node,
        "summary": summary,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _format_context(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"[{chunk['product']} chunk {chunk['chunk_index']}]: {chunk['text']}"
        for chunk in chunks
    )


def _parse_judge_response(raw_response: str) -> dict:
    cleaned = raw_response.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        return {
            "faithfulness_score": None,
            "reasoning": f"Could not parse judge response as JSON: {exc}",
            "hallucinated_claims": [],
        }


def retrieve_node(state: AgentState) -> dict:
    chunks = retrieve(state["question"], products=state.get("products"), top_k=None)

    if chunks:
        products_found = sorted({chunk["product"] for chunk in chunks})
        summary = f"Found {len(chunks)} chunk(s) from product(s): {', '.join(products_found)}."
    else:
        summary = "Found 0 chunks."

    return {
        "retrieved_chunks": chunks,
        "trace": state["trace"] + [_trace_entry("retrieve", summary)],
    }


def generate_node(state: AgentState) -> dict:
    chunks = state["retrieved_chunks"]

    if not chunks:
        return {
            "draft_answer": _NO_CONTEXT_MESSAGE,
            "trace": state["trace"]
            + [_trace_entry("generate", "Skipped LLM call: no chunks retrieved.")],
        }

    prompt = (
        "CONTEXT:\n"
        f"{_format_context(chunks)}\n\n"
        "QUESTION:\n"
        f"{state['question']}\n\n"
        "Answer the question using ONLY the information in CONTEXT above. "
        "If the context is insufficient to answer, say so explicitly "
        "rather than guessing."
    )

    draft_answer = get_llm_provider().generate(prompt)

    return {
        "draft_answer": draft_answer,
        "trace": state["trace"]
        + [_trace_entry("generate", f"Generated a draft answer from {len(chunks)} chunk(s).")],
    }


def judge_node(state: AgentState) -> dict:
    draft_answer = state["draft_answer"]

    if draft_answer.strip() == _NO_CONTEXT_MESSAGE:
        judge_result = {
            "faithfulness_score": None,
            "reasoning": "Skipped: no answer was generated to judge.",
            "hallucinated_claims": [],
        }
        return {
            "judge_result": judge_result,
            "trace": state["trace"]
            + [_trace_entry("judge", "Skipped: no draft answer to judge.")],
        }

    chunks = state["retrieved_chunks"]
    prompt = (
        "EVALUATE_FAITHFULNESS\n\n"
        "Judge whether the ANSWER below is faithful to the CONTEXT below. "
        "Return ONLY a JSON object (no markdown, no commentary) with "
        'exactly these keys: "faithfulness_score" (integer 1-5, where 5 '
        'means fully grounded in the context), "reasoning" (string '
        'explaining the score), and "hallucinated_claims" (a list of '
        "strings -- any claims in ANSWER not supported by CONTEXT, empty "
        "if none).\n\n"
        "CONTEXT:\n"
        f"{_format_context(chunks)}\n\n"
        "ANSWER:\n"
        f"{draft_answer}"
    )

    raw_response = get_llm_provider().generate(prompt)
    judge_result = _parse_judge_response(raw_response)

    summary = f"Judged faithfulness_score={judge_result.get('faithfulness_score')}."
    return {
        "judge_result": judge_result,
        "trace": state["trace"] + [_trace_entry("judge", summary)],
    }


def format_citations_node(state: AgentState) -> dict:
    chunks = state["retrieved_chunks"]
    draft_answer = state["draft_answer"]

    if not chunks:
        final_answer = draft_answer
        summary = "No chunks to cite; final_answer is the draft answer unchanged."
    else:
        citation_lines = [
            f"- {chunk['product']}, chunk {chunk['chunk_index']} ({chunk['source']})"
            for chunk in chunks
        ]
        final_answer = draft_answer + "\n\nSources:\n" + "\n".join(citation_lines)
        summary = f"Appended {len(chunks)} source citation(s)."

    return {
        "final_answer": final_answer,
        "trace": state["trace"] + [_trace_entry("format_citations", summary)],
    }


def _build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("judge", judge_node)
    graph.add_node("format_citations", format_citations_node)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "judge")
    graph.add_edge("judge", "format_citations")
    graph.add_edge("format_citations", END)

    return graph.compile()


_compiled_graph = _build_graph()


def run_agent(question: str, products: list[str] | None = None) -> AgentState:
    """Run the eval agent end-to-end for question, optionally scoped to products.

    This is the single entry point everything else (the API, the eval
    harness) should call.
    """
    initial_state: AgentState = {
        "question": question,
        "products": products,
        "retrieved_chunks": [],
        "draft_answer": "",
        "judge_result": {},
        "final_answer": "",
        "trace": [],
    }
    return _compiled_graph.invoke(initial_state)

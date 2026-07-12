"""LangGraph-based evaluation agent.

Builds a graph with nodes for:
- retrieve: fetch relevant chunks for the question, scoped to the given
  product names (any number of products, not a fixed pair).
- generate: produce a draft answer grounded in the retrieved chunks.
- judge: critique/score the draft answer against the retrieved evidence.
- format_citations: attach citations back to source chunks/products.

The agent accepts a list of product names of any length plus a question,
and returns a grounded, citation-backed answer. Not implemented yet.
"""


def run(question, product_names):
    """Run the eval agent graph for question over product_names and return the answer."""
    raise NotImplementedError

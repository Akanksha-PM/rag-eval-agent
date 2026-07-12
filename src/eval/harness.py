"""Golden dataset evaluation harness.

Runs every question in a golden dataset through the eval agent and
returns per-question results: retrieved chunks, generated answer, judge
reasoning, and scores. This is the "chain of thought" record used to
audit and debug the agent's behavior over time.
"""


def run_golden_dataset(name):
    """Run all questions in the named golden dataset and return per-question results."""
    raise NotImplementedError

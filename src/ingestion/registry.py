"""Runtime registry of ingested products, backed by data/registry.json.

Products are registered dynamically at runtime -- no product name is ever
hardcoded. This module is the single source of truth for which products
currently exist and where their ingested documents live.
"""


def add_product(name, source_path):
    """Register a new product in the registry, pointing at its ingested data."""
    raise NotImplementedError


def list_products():
    """Return the list of currently registered product names."""
    raise NotImplementedError


def get_product_path(name):
    """Return the on-disk path for a registered product's ingested data."""
    raise NotImplementedError


def remove_product(name):
    """Remove a product and its data from the registry."""
    raise NotImplementedError

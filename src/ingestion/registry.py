"""Runtime registry of ingested products, backed by a JSON file on disk.

Products are registered dynamically at runtime -- no product name is ever
hardcoded. This module is the single source of truth for which products
currently exist, what sources have been ingested for them, and whether
their chunks are ready for retrieval.
"""

import json

from config import Config


def _load_registry() -> dict:
    path = Config.REGISTRY_PATH
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
        return {}
    return json.loads(path.read_text())


def _save_registry(registry: dict) -> None:
    Config.REGISTRY_PATH.write_text(json.dumps(registry, indent=2))


def add_product(name: str, source_type: str, source_value: str) -> dict:
    """Create or update a product entry, appending a new source.

    source_type is "file" or "url". New products start with
    status="pending" and chunk_count=0.
    """
    registry = _load_registry()
    entry = registry.get(name) or {"status": "pending", "chunk_count": 0, "sources": []}
    entry["sources"].append({"type": source_type, "value": source_value})
    registry[name] = entry
    _save_registry(registry)
    return {"name": name, **entry}


def list_products() -> list[dict]:
    """Return all registry entries, each including its name."""
    registry = _load_registry()
    return [{"name": name, **entry} for name, entry in registry.items()]


def get_product(name: str) -> dict | None:
    """Return a single product entry (including its name), or None if unknown."""
    registry = _load_registry()
    entry = registry.get(name)
    return {"name": name, **entry} if entry is not None else None


def remove_product(name: str) -> bool:
    """Remove a product from the registry. Returns True if it existed."""
    registry = _load_registry()
    if name not in registry:
        return False
    del registry[name]
    _save_registry(registry)
    return True


def update_chunk_count(name: str, count: int) -> None:
    """Set chunk_count for a product and flip its status to "ready"."""
    registry = _load_registry()
    if name not in registry:
        raise KeyError(f"Unknown product: {name}")
    registry[name]["chunk_count"] = count
    registry[name]["status"] = "ready"
    _save_registry(registry)

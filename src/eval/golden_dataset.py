"""JSON-backed storage for golden datasets used by the eval harness.

Each dataset is a standalone JSON file at
GOLDEN_DATASET_DIR/{slugified_name}.json, containing its display name,
creation timestamp, and a list of question/expected-answer items.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from config import Config


def _slugify(name: str) -> str:
    return name.lower().replace(" ", "_")


def _dataset_path(name: str) -> Path:
    return Config.GOLDEN_DATASET_DIR / f"{_slugify(name)}.json"


def create_dataset(name: str) -> dict:
    """Create a new, empty golden dataset. Raises ValueError if it already exists."""
    path = _dataset_path(name)
    if path.exists():
        raise ValueError(f"A golden dataset named {name!r} already exists.")

    dataset = {
        "name": name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "items": [],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dataset, indent=2))
    return dataset


def list_datasets() -> list[dict]:
    """Return a lightweight summary (name, created_at, item_count) for every dataset."""
    directory = Config.GOLDEN_DATASET_DIR
    if not directory.exists():
        return []

    summaries = []
    for path in sorted(directory.glob("*.json")):
        dataset = json.loads(path.read_text())
        summaries.append(
            {
                "name": dataset["name"],
                "created_at": dataset["created_at"],
                "item_count": len(dataset["items"]),
            }
        )
    return summaries


def get_dataset(name: str) -> dict | None:
    """Return the full dataset (including all items), or None if not found."""
    path = _dataset_path(name)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def add_item(
    name: str,
    question: str,
    expected_answer: str,
    expected_products: list[str] | None = None,
    notes: str = "",
) -> dict:
    """Append a new item to the dataset. Returns the created item."""
    path = _dataset_path(name)
    if not path.exists():
        raise ValueError(f"Golden dataset {name!r} does not exist.")

    question = question.strip()
    expected_answer = expected_answer.strip()
    if not question:
        raise ValueError("question cannot be empty.")
    if not expected_answer:
        raise ValueError("expected_answer cannot be empty.")

    dataset = json.loads(path.read_text())
    item = {
        "id": f"{_slugify(name)}_{len(dataset['items'])}",
        "question": question,
        "expected_answer": expected_answer,
        "expected_products": expected_products,
        "notes": notes,
    }
    dataset["items"].append(item)
    path.write_text(json.dumps(dataset, indent=2))
    return item


def delete_item(name: str, item_id: str) -> bool:
    """Remove an item by id. Returns whether it was found and removed."""
    path = _dataset_path(name)
    if not path.exists():
        return False

    dataset = json.loads(path.read_text())
    items = dataset["items"]
    remaining = [item for item in items if item["id"] != item_id]
    if len(remaining) == len(items):
        return False

    dataset["items"] = remaining
    path.write_text(json.dumps(dataset, indent=2))
    return True

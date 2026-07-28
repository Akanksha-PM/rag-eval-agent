"""Routes for managing golden datasets used by the eval harness."""

from typing import Optional

from pydantic import BaseModel

from fastapi import APIRouter, HTTPException

from src.eval import golden_dataset

router = APIRouter(prefix="/golden-datasets", tags=["golden-datasets"])


class GoldenDatasetCreate(BaseModel):
    name: str


class GoldenDatasetItemCreate(BaseModel):
    question: str
    expected_answer: str
    expected_products: Optional[list[str]] = None
    notes: str = ""


@router.get("")
async def list_golden_datasets():
    """List all golden datasets."""
    return golden_dataset.list_datasets()


@router.post("", status_code=201)
async def create_golden_dataset(dataset: GoldenDatasetCreate):
    """Create a new, empty golden dataset."""
    try:
        return golden_dataset.create_dataset(dataset.name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/{name}")
async def get_golden_dataset(name: str):
    """Return a single golden dataset, including all its items."""
    dataset = golden_dataset.get_dataset(name)
    if dataset is None:
        raise HTTPException(status_code=404, detail=f"Golden dataset {name!r} not found.")
    return dataset


@router.post("/{name}/items", status_code=201)
async def add_golden_dataset_item(name: str, item: GoldenDatasetItemCreate):
    """Add a question/answer item to the named golden dataset."""
    try:
        return golden_dataset.add_item(
            name,
            item.question,
            item.expected_answer,
            expected_products=item.expected_products,
            notes=item.notes,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

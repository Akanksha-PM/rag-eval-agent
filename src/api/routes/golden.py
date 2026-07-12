"""Routes for managing golden datasets used by the eval harness."""

from typing import Optional

from pydantic import BaseModel

from fastapi import APIRouter

router = APIRouter(prefix="/golden-datasets", tags=["golden-datasets"])


class GoldenDatasetCreate(BaseModel):
    name: str


class GoldenDatasetItemCreate(BaseModel):
    question: str
    product_names: list[str]
    expected_answer: Optional[str] = None


@router.get("")
async def list_golden_datasets():
    """List all golden datasets."""
    raise NotImplementedError


@router.post("")
async def create_golden_dataset(dataset: GoldenDatasetCreate):
    """Create a new, empty golden dataset."""
    raise NotImplementedError


@router.post("/{name}/items")
async def add_golden_dataset_item(name: str, item: GoldenDatasetItemCreate):
    """Add a question/answer item to the named golden dataset."""
    raise NotImplementedError

"""Route for asking a question against one or more registered products."""

from pydantic import BaseModel

from fastapi import APIRouter

router = APIRouter(prefix="/query", tags=["query"])


class QueryRequest(BaseModel):
    question: str
    product_names: list[str]


@router.post("")
async def query(request: QueryRequest):
    """Run the eval agent for request.question, scoped to request.product_names."""
    raise NotImplementedError

"""Route for asking a question against one or more registered products."""

from pydantic import BaseModel

from fastapi import APIRouter, HTTPException

from src.agent.eval_agent import run_agent
from src.ingestion import registry

router = APIRouter(prefix="/query", tags=["query"])


class QueryRequest(BaseModel):
    question: str
    products: list[str] | None = None


@router.post("")
async def query(request: QueryRequest):
    """Run the eval agent for request.question, scoped to request.products."""
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if request.products:
        registered_names = {product["name"] for product in registry.list_products()}
        unknown = [name for name in request.products if name not in registered_names]
        if unknown:
            valid_names = ", ".join(sorted(registered_names)) if registered_names else "none"
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unknown product(s): {', '.join(unknown)}. "
                    f"Currently registered products: {valid_names}."
                ),
            )

    try:
        state = run_agent(question, request.products)
    except Exception:
        raise HTTPException(
            status_code=500, detail="Internal error while running the query."
        )

    chunks_used = [
        {
            "product": chunk["product"],
            "source": chunk["source"],
            "chunk_index": chunk["chunk_index"],
            "score": chunk["score"],
        }
        for chunk in state["retrieved_chunks"]
    ]

    return {
        "question": question,
        "answer": state["final_answer"],
        "products_searched": request.products if request.products else "all registered products",
        "judge": state["judge_result"],
        "chunks_used": chunks_used,
        "trace": state["trace"],
    }

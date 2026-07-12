"""FastAPI application entrypoint.

Run with: uvicorn src.api.app:app --reload
"""

from fastapi import FastAPI

from src.api.routes import eval as eval_routes
from src.api.routes import golden, products, query

app = FastAPI(title="RAG Eval Agent")

app.include_router(products.router)
app.include_router(golden.router)
app.include_router(query.router)
app.include_router(eval_routes.router)


@app.get("/health")
async def health():
    return {"status": "ok"}

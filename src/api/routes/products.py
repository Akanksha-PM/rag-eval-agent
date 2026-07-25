"""Routes for registering products and synchronously ingesting their docs.

Products are added dynamically at runtime -- there is no fixed set of
product names anywhere in this module.
"""

from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from config import Config
from src.ingestion import registry
from src.ingestion.pipeline import ingest_product as run_ingest
from src.retrieval.vector_store import VectorStore

router = APIRouter(prefix="/products", tags=["products"])


@router.post("/{name}/ingest")
async def ingest_product(
    name: str,
    file: Optional[UploadFile] = File(default=None),
    url: Optional[str] = Form(default=None),
):
    """Synchronously ingest a document for product `name` from an uploaded file or a url.

    The caller waits while the source is fetched (if a url), chunked, and
    embedded. Registers `name` in the registry if it isn't already there.
    """
    # Swagger UI (/docs) sends an empty string for unfilled optional form
    # fields rather than omitting them, so without this normalization the
    # exactly-one-of check below would see url="" as "provided" and reject
    # a valid file-only submission made through /docs as "both".
    if url is not None and url.strip() == "":
        url = None

    if file is None and not url:
        raise HTTPException(
            status_code=400, detail="Provide either a file or a url, not neither."
        )
    if file is not None and url:
        raise HTTPException(
            status_code=400, detail="Provide either a file or a url, not both."
        )

    if file is not None:
        slug = name.lower().replace(" ", "_")
        directory = Config.DATA_DIR / "docs" / slug
        directory.mkdir(parents=True, exist_ok=True)
        saved_path = directory / file.filename
        saved_path.write_bytes(await file.read())

        source_type, source_value = "file", str(saved_path)
    else:
        source_type, source_value = "url", url

    try:
        return run_ingest(name, source_type, source_value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        raise HTTPException(
            status_code=500, detail="Internal error while ingesting the product."
        )


@router.get("")
async def list_products():
    """List all currently registered products."""
    return registry.list_products()


@router.delete("/{name}")
async def delete_product(name: str):
    """Remove a registered product, its vector store chunks, and its ingested data."""
    removed = registry.remove_product(name)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Product {name!r} not found.")

    chunks_removed = VectorStore().remove_product(name)

    return {
        "product": name,
        "message": f"Removed {name!r} and {chunks_removed} chunk(s) from the vector store.",
        "chunks_removed": chunks_removed,
    }

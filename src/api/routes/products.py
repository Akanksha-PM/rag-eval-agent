"""Routes for registering products and synchronously ingesting their docs.

Products are added dynamically at runtime -- there is no fixed set of
product names anywhere in this module.
"""

from typing import Optional

from fastapi import APIRouter, File, Form, UploadFile

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
    raise NotImplementedError


@router.get("")
async def list_products():
    """List all currently registered products."""
    raise NotImplementedError


@router.delete("/{name}")
async def delete_product(name: str):
    """Remove a registered product and its ingested data."""
    raise NotImplementedError

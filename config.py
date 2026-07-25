"""Central configuration for the RAG Eval Agent."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent


class Config:
    """Runtime configuration, overridable via environment variables."""

    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "mock")
    EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "tfidf")

    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
    TOP_K = int(os.getenv("TOP_K", 5))

    DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
    REGISTRY_PATH = Path(os.getenv("REGISTRY_PATH", DATA_DIR / "registry.json"))
    GOLDEN_DATASET_DIR = Path(
        os.getenv("GOLDEN_DATASET_DIR", BASE_DIR / "golden_datasets")
    )

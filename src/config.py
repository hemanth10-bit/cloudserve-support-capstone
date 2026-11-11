"""Settings read from the environment (and .env if present). No secrets live in code."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _path(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (ROOT / p).resolve()


@dataclass(frozen=True)
class Settings:
    openrouter_api_key: str
    openrouter_base_url: str
    model_name: str
    embedding_model: str
    chroma_path: Path
    database_url: str
    cache_path: Path
    log_level: str
    confidence_threshold: float
    retrieval_top_k: int
    retrieval_min_score: float
    llm_timeout_seconds: float
    llm_max_retries: int
    docs_path: Path


def get_settings() -> Settings:
    env = os.environ.get
    return Settings(
        openrouter_api_key=env("OPENROUTER_API_KEY", ""),
        openrouter_base_url=env("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        model_name=env("MODEL_NAME", "meta-llama/llama-3.1-8b-instruct"),
        embedding_model=env("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        chroma_path=_path(env("CHROMA_PATH", "./storage/chroma")),
        database_url=env("DATABASE_URL", "sqlite:///./storage/decisions.db"),
        cache_path=_path(env("CACHE_PATH", "./storage/llm_cache.sqlite")),
        log_level=env("LOG_LEVEL", "INFO"),
        confidence_threshold=float(env("CONFIDENCE_THRESHOLD", "0.80")),
        retrieval_top_k=int(env("RETRIEVAL_TOP_K", "5")),
        retrieval_min_score=float(env("RETRIEVAL_MIN_SCORE", "0.40")),
        llm_timeout_seconds=float(env("LLM_TIMEOUT_SECONDS", "30")),
        llm_max_retries=int(env("LLM_MAX_RETRIES", "3")),
        docs_path=_path(env("DOCS_PATH", "./data/documentation.json")),
    )

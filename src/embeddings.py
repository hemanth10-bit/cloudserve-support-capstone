"""Sentence embeddings (all-MiniLM-L6-v2). One shared model instance, loaded lazily."""

from __future__ import annotations

import threading

import numpy as np

from .config import get_settings

_lock = threading.Lock()
_model = None
_model_name = None


def _get_model():
    global _model, _model_name
    name = get_settings().embedding_model
    with _lock:
        if _model is None or _model_name != name:
            from sentence_transformers import SentenceTransformer  # imported late: it is slow

            _model = SentenceTransformer(name)
            _model_name = name
    return _model


def embed(texts: list[str]) -> np.ndarray:
    """Return L2-normalised float32 embeddings, shape (len(texts), dim).

    Normalised vectors make cosine similarity a plain dot product. Raises on
    model failure; callers that must not fail (retrieval) catch and degrade.
    """
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    vecs = _get_model().encode(texts, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
    return np.asarray(vecs, dtype=np.float32)

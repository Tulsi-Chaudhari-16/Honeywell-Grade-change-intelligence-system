"""
GCIS ML Pipeline — Embedding Inference Wrapper

Called by the Historical Retrieval Agent (partner's code) to embed
the current in-progress transition for Qdrant ANN search.

Usage:
    from backend.app.ml.infer.embed import embed_transition
    vector = embed_transition(feature_vector_sequence)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ..config import ALL_TAGS, INPUT_WINDOW_STEPS
from ..models.embedding_model import TransitionEmbedder
from ..registry import get_model_path

logger = logging.getLogger(__name__)

_embedder: Optional[TransitionEmbedder] = None


def _load_embedder() -> None:
    global _embedder
    try:
        path = get_model_path("embedding_model")
        _embedder = TransitionEmbedder.load(path)
        logger.info(f"[Embed] Embedder loaded from {path}")
    except FileNotFoundError:
        logger.warning("[Embed] Embedding model not in registry — embed calls will return zeros.")


def embed_transition(feature_vector_sequence: list[dict[str, float]]) -> np.ndarray:
    """
    Embed a sequence of feature vectors into a fixed-length vector for Qdrant search.

    Called by Historical Retrieval Agent.

    Args:
        feature_vector_sequence: list of feature dicts ordered by time (oldest first).
            Each dict maps feature_name -> float value.
            The list should contain INPUT_WINDOW_STEPS items; if shorter, it is
            left-padded; if longer, only the last INPUT_WINDOW_STEPS are used.

    Returns:
        np.ndarray of shape (EMBEDDING_DIM,) — float32.
        Returns a zero vector of the correct shape if the model is unavailable.
    """
    from ..config import EMBEDDING_DIM

    if _embedder is None:
        _load_embedder()

    if _embedder is None:
        logger.warning("[Embed] No embedder available, returning zero vector.")
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    # Gather all feature keys from the sequence
    all_keys = sorted(
        set(k for fv in feature_vector_sequence for k in fv.keys())
    )

    # Build matrix: (n_steps, n_features)
    n_steps = len(feature_vector_sequence)
    matrix = np.array(
        [[fv.get(k, 0.0) for k in all_keys] for fv in feature_vector_sequence],
        dtype=np.float32,
    )  # (n_steps, n_features)

    # Pad or truncate to INPUT_WINDOW_STEPS
    if n_steps < INPUT_WINDOW_STEPS:
        pad = np.tile(matrix[0], (INPUT_WINDOW_STEPS - n_steps, 1))
        matrix = np.vstack([pad, matrix])
    else:
        matrix = matrix[-INPUT_WINDOW_STEPS:]

    # (n_features, seq_len) — model expects channels-first
    sequence = matrix.T

    return _embedder.embed(sequence)


def reload_embedder() -> None:
    """Force reload after a new embedding model is registered."""
    global _embedder
    _embedder = None
    _load_embedder()

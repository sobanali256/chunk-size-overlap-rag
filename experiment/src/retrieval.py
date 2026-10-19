"""
retrieval.py - exact cosine search over one paper's chunks.

Embeddings are L2-normalised, so cosine similarity = dot product. Every chunk of the question's
own paper is scored (no approximate index). Ties are broken by chunk position (stable sort), so
rankings are deterministic. This matches the pilot's --scope doc search.
"""
from __future__ import annotations

import numpy as np


def rank_paper(q_vec: np.ndarray, chunk_vecs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (order, scores): chunk indices best first, and every chunk's score."""
    scores = chunk_vecs @ q_vec
    order = np.argsort(-scores, kind="stable")
    return order, scores


def k_for(mode: dict, size: int) -> int:
    if mode["type"] == "token_budget":
        k = mode["budget"] // size
        if k < 1 or k * size != mode["budget"]:
            raise ValueError(f"budget {mode['budget']} is not a multiple of chunk size {size}")
        return k
    if mode["type"] == "fixed_k":
        return int(mode["k"])
    raise ValueError(f"unknown retrieval mode {mode}")

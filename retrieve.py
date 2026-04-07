"""Vector retrieval utilities using cosine similarity."""

from typing import Any, Dict, List

import numpy as np

from embed import embed_text


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Compute cosine similarity for two numeric vectors."""
    arr_a = np.asarray(vec_a, dtype=float)
    arr_b = np.asarray(vec_b, dtype=float)

    if arr_a.shape != arr_b.shape or arr_a.size == 0:
        return 0.0

    norm_a = np.linalg.norm(arr_a)
    norm_b = np.linalg.norm(arr_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return float(np.dot(arr_a, arr_b) / (norm_a * norm_b))


def _rank_chunks_by_query_vector(
    query_vector: List[float], embedded_chunks: List[Dict[str, Any]], top_k: int
) -> List[Dict[str, Any]]:
    """Score and rank embedded chunks against a query vector."""
    scored_chunks: List[Dict[str, Any]] = []
    for chunk in embedded_chunks:
        vector = chunk.get("vector")
        if not vector:
            continue
        score = cosine_similarity(query_vector, vector)
        chunk_with_score = dict(chunk)
        chunk_with_score["score"] = score
        scored_chunks.append(chunk_with_score)

    scored_chunks.sort(key=lambda item: item["score"], reverse=True)
    return scored_chunks[:top_k]


def retrieve(
    query: str, embedded_chunks: List[Dict[str, Any]], top_k: int = 5
) -> List[Dict[str, Any]]:
    """Retrieve the top-k most similar chunks for a text query."""
    query_vector = embed_text(query)
    return _rank_chunks_by_query_vector(query_vector, embedded_chunks, top_k)


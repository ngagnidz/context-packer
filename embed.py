"""Embedding utilities for chunked documents."""

import os
from typing import Any, Dict, List

from openai import OpenAI

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None


EMBEDDING_MODEL = "text-embedding-3-small"


def _build_client() -> OpenAI:
    """Initialize an OpenAI client using environment configuration."""
    if load_dotenv is not None:
        load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is missing. Set it in your .env file "
            "(install `python-dotenv` to auto-load .env)."
        )
    return OpenAI(api_key=api_key)


def embed_text(text: str) -> List[float]:
    """Generate an embedding vector for a single input string."""
    client = _build_client()
    try:
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
        return response.data[0].embedding
    except Exception as exc:
        raise RuntimeError(f"Failed to embed text: {exc}") from exc


def embed_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Embed each chunk and return copies with an attached vector field."""
    embedded_chunks: List[Dict[str, Any]] = []
    for chunk in chunks:
        chunk_with_vector = dict(chunk)
        chunk_text = str(chunk.get("chunk_text", ""))
        try:
            chunk_with_vector["vector"] = embed_text(chunk_text)
        except Exception as exc:
            print(
                f"Warning: could not embed chunk from "
                f"{chunk.get('source_file', 'unknown')}: {exc}"
            )
            chunk_with_vector["vector"] = []
        embedded_chunks.append(chunk_with_vector)
    return embedded_chunks


if __name__ == "__main__":
    test_texts = [
        "Pacific Context Packer ingests local text files.",
        "Embeddings convert text into vectors for semantic search.",
    ]

    for idx, text in enumerate(test_texts, start=1):
        try:
            vector = embed_text(text)
            print(f"Text {idx} first 5 dims: {vector[:5]}")
            print(f"Vector length: {len(vector)}")
        except Exception as exc:
            print(f"Text {idx} embedding failed: {exc}")

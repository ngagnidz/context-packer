"""Document ingestion and chunking utilities."""

from pathlib import Path
from typing import Dict, List

SUPPORTED_EXTENSIONS = {".txt"}
DEFAULT_CHUNK_SIZE_WORDS = 500
DEFAULT_CHUNK_OVERLAP_WORDS = 50


def _read_text_file(file_path: Path) -> str:
    """Read UTF-8 text content from a single file path."""
    return file_path.read_text(encoding="utf-8", errors="ignore").strip()


def _chunk_words(text: str, chunk_size_words: int, overlap_words: int) -> List[str]:
    """Split text into overlapping word-based chunks."""
    if chunk_size_words <= 0:
        raise ValueError("chunk_size_words must be greater than 0")
    if overlap_words < 0:
        raise ValueError("overlap_words must be >= 0")
    if overlap_words >= chunk_size_words:
        raise ValueError("overlap_words must be smaller than chunk_size_words")

    words = text.split()
    if not words:
        return []

    step = chunk_size_words - overlap_words
    chunks: List[str] = []
    for start_idx in range(0, len(words), step):
        end_idx = start_idx + chunk_size_words
        chunk_words = words[start_idx:end_idx]
        if not chunk_words:
            continue
        chunks.append(" ".join(chunk_words))
        if end_idx >= len(words):
            break
    return chunks


def ingest_docs(
    docs_dir: str = "docs",
    chunk_size_words: int = DEFAULT_CHUNK_SIZE_WORDS,
    overlap_words: int = DEFAULT_CHUNK_OVERLAP_WORDS,
) -> List[Dict[str, object]]:
    """Ingest text files recursively and return structured chunk records."""
    root = Path(docs_dir)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Docs directory not found: {docs_dir}")

    chunk_records: List[Dict[str, object]] = []
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    for file_path in files:
        text = _read_text_file(file_path)

        for chunk_index, chunk_text in enumerate(
            _chunk_words(text, chunk_size_words, overlap_words)
        ):
            chunk_records.append(
                {
                    "chunk_text": chunk_text,
                    "source_file": file_path.name,
                    "chunk_index": chunk_index,
                }
            )

    return chunk_records


if __name__ == "__main__":
    generated_chunks = ingest_docs("docs")
    print(f"Created {len(generated_chunks)} chunks from docs.")

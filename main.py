"""Entry point for running smart vs naive retrieval comparison."""

import os
import time
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from openai import OpenAI

from embed import embed_chunks
from ingest import ingest_docs
from retrieve import retrieve

MAX_NAIVE_CONTEXT_TOKENS = 120000
CHAT_MODEL = "gpt-4o-mini"


def _build_chat_client() -> OpenAI:
    """Create a chat client from the configured OpenAI API key."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing. Add it to .env.")
    return OpenAI(api_key=api_key)


def _ask_llm(client: OpenAI, context: str, query: str) -> Tuple[str, int, float]:
    """Submit a context-backed question and return answer, tokens, and latency."""
    start_time = time.time()
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "Answer using the provided context."},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{query}",
            },
        ],
    )
    elapsed = time.time() - start_time
    answer = response.choices[0].message.content or ""
    tokens = response.usage.total_tokens if response.usage is not None else 0
    return answer, tokens, elapsed


def main() -> None:
    """Run and compare smart retrieval against naive full-context prompting."""
    print("Ingesting docs...")
    chunks = ingest_docs("docs")
    print(f"Ingested {len(chunks)} chunks.")

    print("Embedding chunks...")
    embedded_chunks = embed_chunks(chunks)
    print(f"Embedded {len(embedded_chunks)} chunks.")

    query = input("\nType your query: ").strip()
    if not query:
        print("No query provided. Exiting.")
        return

    client = _build_chat_client()

    print("\nRunning SMART retrieval...")
    smart_top_chunks = retrieve(query, embedded_chunks, top_k=5)
    smart_context = "\n\n".join(
        f"[{item.get('source_file', 'unknown')} | score={item.get('score', 0.0):.4f}]\n"
        f"{item.get('chunk_text', '')}"
        for item in smart_top_chunks
    )
    try:
        smart_answer, smart_tokens, smart_time = _ask_llm(client, smart_context, query)
    except Exception as exc:
        print(f"SMART retrieval failed: {exc}")
        return

    print("Running NAIVE retrieval...")
    all_text = "\n\n".join(str(item.get("chunk_text", "")) for item in embedded_chunks)
    words = all_text.split()
    naive_was_truncated = False
    if len(words) > MAX_NAIVE_CONTEXT_TOKENS:
        all_text = " ".join(words[:MAX_NAIVE_CONTEXT_TOKENS])
        naive_was_truncated = True

    try:
        naive_answer, naive_tokens, naive_time = _ask_llm(client, all_text, query)
    except Exception as exc:
        print(f"NAIVE retrieval failed: {exc}")
        return

    saved = naive_tokens - smart_tokens
    saved_pct = (saved / naive_tokens * 100.0) if naive_tokens > 0 else 0.0

    print("\n=== COMPARISON ===")
    print(f"SMART ANSWER: ({smart_tokens} tokens, {smart_time:.2f}s)")
    print(smart_answer)
    print(f"\nNAIVE ANSWER: ({naive_tokens} tokens, {naive_time:.2f}s)")
    print(naive_answer)
    if naive_was_truncated:
        print(
            f"\nNote: Naive context exceeded {MAX_NAIVE_CONTEXT_TOKENS} tokens "
            "approximation and was truncated."
        )
    print(f"\nTokens saved: {saved} ({saved_pct:.2f}%)")


if __name__ == "__main__":
    main()

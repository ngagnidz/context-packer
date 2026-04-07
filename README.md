# Context Packer

Simple retrieval demo that compares:
- **SMART retrieval** (top-k relevant chunks)
- **NAIVE retrieval** (all chunks as context)

It prints both answers, response time, token usage, and tokens saved.

## Requirements

- Python 3.10+
- OpenAI API key

## What's included
- `docs/meeting_notes.txt` — fake team meeting notes
- `docs/anthropic_contextual_retrieval.txt` — Anthropic's contextual retrieval article
- `docs/rag_v_long_context.txt` — RAG vs long context window analysis

## Quick test
Run main.py with this query to see a smart vs naive difference:
"What did we decide about TTFT this year?"

Smart retrieval finds the meeting notes instantly.
Naive processes the RAG articles and uses 50%+ more tokens.

## 1) Clone and enter project

```bash
git clone https://github.com/ngagnidz/context-packer.git
cd context-packer
```

## 2) Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3) Install dependencies

```bash
python3 -m pip install -r requirements.txt
```

## 4) Set environment variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_api_key_here
```

## 5) Add documents

Put your `.txt` files in `docs/` 
```bash
docs/
  meeting_notes.txt
  rag_v_long_context.txt
```
Note: The `docs/` folder already includes sample documents to get you started.

## 6) Run the program

```bash
python3 main.py
```

You will be prompted for a query in the terminal.

## `main.py` 

1. Ingests `.txt` files from `docs/`
2. Chunks text
3. Embeds all chunks
4. Runs SMART retrieval (`top_k=5`) + answer generation
5. Runs NAIVE retrieval (all chunks) + answer generation
6. Prints timing and token comparison
# RAG Quickstart v2 — latest Python, zero heavy dependencies

A working RAG product you can run on your own machine, free. Ask a question,
get an answer with the source chunks underneath. That citation part is the
whole point: an answer without a source is not usable in regulated work.

## Why v2 exists

v1 used ChromaDB + fastembed, which need onnxruntime, and onnxruntime had
no builds for Python 3.15 — so v1 forced a Python 3.12 downgrade. v2 removes
that dependency entirely: embeddings come from a real vendor API (Google
Gemini), vectors live in SQLite, and similarity is pure Python. The only
dependencies are fastapi, uvicorn, httpx and python-multipart — all pure
Python wheels — so v2 runs on Python 3.15 with nothing native to compile
or wait on. This is also the integration lesson: calling a real vendor API
over REST, with a key, retries and fallbacks, instead of a local model.

## How it works

```text
  Upload docs (.txt / .md)
        │
        ▼
     Chunk (~800 words, 100 overlap)
        │
        ▼
  Gemini embed (gemini-embedding-001, REST)
        │
        ▼
  SQLite vectors (stdlib only, vectors as JSON lists)
        │
        ▼
  Ask a question ──▶ Gemini embed the question
        │
        ▼
  Retrieve top 4 by pure-Python cosine similarity
        │
        ▼
  Gemini generate (2.5-flash, fallback 2.0-flash)
        │
        ▼
  Cited answer: answer ONLY from the chunks, [doc chunk n],
  or a plain "the context does not contain the answer"
```

## Run it

```bash
cd ~/workspace/rag-quickstart-v2
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
GEMINI_API_KEY=your-key-here uvicorn app:app --port 8000
```

Then open http://127.0.0.1:8000. On first run (with a key set) the app
ingests the 3 seed documents in `seed_docs/` automatically: a ClaimSorted
product summary, a fictional policy wording excerpt, and a RAG explainer.

Get a free key at https://aistudio.google.com (no card). Never put the key
in chat or in git — pass it as an environment variable only.

## Test it (3 questions)

```bash
curl -s -X POST http://127.0.0.1:8000/ask -H 'Content-Type: application/json' \
  -d '{"question": "What is the storm deductible in the sample policy?"}'
# Expected: 500 (standard deductible is 250; storm is 500), cited from
# 02-sample-policy-wording.md.

curl -s -X POST http://127.0.0.1:8000/ask -H 'Content-Type: application/json' \
  -d '{"question": "What does FNOL mean and what happens first in a claim?"}'
# Expected: First Notice of Loss — the policyholder's first report of a
# loss — followed by triage, cited from 01-claimsorted-product.md.

curl -s -X POST http://127.0.0.1:8000/ask -H 'Content-Type: application/json' \
  -d '{"question": "What is an eval in a RAG system?"}'
# Expected: a test for a RAG system, cited from 03-how-rag-works.md.
```

## Endpoints

- `GET /` — the single-page UI (ask + upload).
- `GET /health` — status, chunk count, embedding model, LLM mode.
- `POST /ask` — `{"question": "..."}` returns `{"answer", "sources", "mode"}`.
- `POST /ingest` — multipart file upload (`.txt`/`.md`) or a `text` form field.

Without `GEMINI_API_KEY` the server still starts, `/health` reports the key
as unset, and `/ask` / `/ingest` return HTTP 503 with a clear message
naming the missing key. Nothing crashes, and nothing fakes an answer.

## Honest scale note

Pure-Python cosine similarity over SQLite is correct and fast enough for
hundreds to low thousands of chunks — exactly this project's scale. It is
a linear scan: every query compares against every chunk, so cost grows
with the collection. Beyond that, move to a real vector index: Supabase
pgvector is the next integration lesson (managed Postgres + the `vector`
extension + an index, same retrieve-then-generate shape, no application
code change to the answering logic). Do not reach for a vector database
before you need one; do reach for it the moment query latency or chunk
count makes the scan noticeable.

## Files

- `app.py` — the whole backend: chunking, Gemini REST calls, SQLite
  storage, pure-Python retrieval, cited generation.
- `index.html` — the UI, served by FastAPI, no build step.
- `seed_docs/` — 3 seed documents (ClaimSorted summary, fictional policy
  wording, RAG explainer). All invented/fictional where marked; no real
  company data.
- `rag.db` — the SQLite store, created on first run (not committed).

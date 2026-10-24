"""RAG Quickstart v2: FastAPI + Gemini embeddings + SQLite vectors.

Why v2 exists: v1 used ChromaDB + fastembed, which need onnxruntime, and
onnxruntime had no builds for Python 3.15 — so v1 forced a Python 3.12
downgrade. v2 removes that dependency entirely: embeddings come from a
real vendor API (Google Gemini text-embedding-004 over REST), vectors are
stored in SQLite via the Python standard library, and similarity is pure
Python cosine similarity. Dependencies are ONLY fastapi, uvicorn, httpx
and python-multipart — all pure Python, so this runs on Python 3.15
(and any other modern Python) with no native wheels at all.

Run:  python3 -m venv .venv && source .venv/bin/activate
      pip install -r requirements.txt
      GEMINI_API_KEY=your-key-here uvicorn app:app --port 8000

Behaviour: the app answers ONLY from retrieved chunks, cites
[doc chunk n], and says plainly when the context lacks the answer.
Without GEMINI_API_KEY it does not crash and never fakes an answer: the
endpoints return a clear error naming the missing key.
"""
import json
import math
import os
import sqlite3
from pathlib import Path

import httpx
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

BASE = Path(__file__).parent
SEED_DIR = BASE / "seed_docs"
DB_PATH = BASE / "rag.db"
EMBED_MODEL = "text-embedding-004"
GEN_MODELS = ("gemini-2.5-flash", "gemini-2.0-flash")
TOP_K = 4

app = FastAPI(title="RAG Quickstart v2")


def require_key() -> str:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Get a free key at "
            "https://aistudio.google.com, then run with "
            "GEMINI_API_KEY=your-key-here uvicorn app:app --port 8000"
        )
    return key


def db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute(
        "CREATE TABLE IF NOT EXISTS chunks "
        "(id TEXT PRIMARY KEY, doc TEXT, chunk INTEGER, text TEXT, vector TEXT)"
    )
    return conn


def chunk_text(text: str, size: int = 800, overlap: int = 100):
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + size]))
        i += size - overlap
    return [c for c in chunks if c.strip()]


def embed_texts(texts):
    """Gemini embeddings over REST. Returns a list of float lists."""
    key = require_key()
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{EMBED_MODEL}:batchEmbedContents?key={key}"
    )
    payload = {
        "requests": [
            {"model": f"models/{EMBED_MODEL}", "content": {"parts": [{"text": t}]}}
            for t in texts
        ]
    }
    resp = httpx.post(url, json=payload, timeout=120)
    try:
        resp.raise_for_status()
    except httpx.HTTPStatusError:
        # Print Google's own reason (the body never contains the key) so a
        # 400/404 can be diagnosed instead of guessed at.
        print(f"[v2] Gemini embedding error {resp.status_code}: {resp.text[:500]}")
        raise
    out = []
    for emb in resp.json()["embeddings"]:
        out.append(emb["values"])
    return out


def ingest_text(doc_name: str, text: str) -> int:
    chunks = chunk_text(text)
    if not chunks:
        return 0
    vectors = embed_texts(chunks)
    conn = db()
    for n, (chunk, vec) in enumerate(zip(chunks, vectors)):
        conn.execute(
            "INSERT OR REPLACE INTO chunks (id, doc, chunk, text, vector) "
            "VALUES (?, ?, ?, ?, ?)",
            (f"{doc_name}#{n}", doc_name, n, chunk, json.dumps(vec)),
        )
    conn.commit()
    conn.close()
    return len(chunks)


def seed_if_empty() -> int:
    conn = db()
    count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    conn.close()
    if count > 0:
        return count
    total = 0
    for path in sorted(SEED_DIR.glob("*.md")):
        total += ingest_text(path.name, path.read_text())
    return total


def cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def retrieve(question: str, k: int = TOP_K):
    qvec = embed_texts([question])[0]
    conn = db()
    rows = conn.execute("SELECT doc, chunk, text, vector FROM chunks").fetchall()
    conn.close()
    scored = []
    for doc, chunk, text, vec_json in rows:
        scored.append(
            {
                "doc": doc,
                "chunk": chunk,
                "text": text,
                "score": round(cosine(qvec, json.loads(vec_json)), 4),
            }
        )
    scored.sort(key=lambda s: s["score"], reverse=True)
    return scored[:k]


SYSTEM_PROMPT = (
    "Answer ONLY from the context below. Cite sources like [doc chunk n]. "
    "If the context does not contain the answer, say so plainly."
)


def generate_answer(question: str, sources) -> str:
    key = require_key()
    context = "\n\n".join(
        f"[{s['doc']} chunk {s['chunk']}] {s['text']}" for s in sources
    )
    prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {question}"
    last_error = None
    for model in GEN_MODELS:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?key={key}"
        )
        resp = httpx.post(
            url,
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.2},
            },
            timeout=120,
        )
        if resp.status_code == 200:
            parts = resp.json()["candidates"][0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts)
        last_error = f"Gemini {model} returned HTTP {resp.status_code}"
        if resp.status_code not in (400, 404):
            resp.raise_for_status()
    raise RuntimeError(last_error or "Gemini generation failed")


@app.on_event("startup")
def startup():
    # Seeding needs a key; without one the app still starts and the
    # endpoints explain what is missing instead of crashing.
    try:
        seed_if_empty()
    except Exception as e:
        print(f"[v2] startup seeding skipped: {e}")


@app.get("/health")
def health():
    conn = db()
    count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    conn.close()
    key_set = bool(os.getenv("GEMINI_API_KEY"))
    return {
        "status": "ok",
        "chunks_indexed": count,
        "embedding_model": EMBED_MODEL,
        "llm": "gemini-2.5-flash (fallback gemini-2.0-flash)"
        if key_set
        else "not configured — GEMINI_API_KEY is not set",
        "gemini_key_set": key_set,
    }


@app.get("/")
def ui():
    return FileResponse(BASE / "index.html")


class AskIn(BaseModel):
    question: str


@app.post("/ingest")
async def ingest(file: UploadFile = File(None), text: str = Form(None)):
    try:
        if file is not None:
            raw = await file.read()
            doc_name = file.filename or "upload.txt"
            try:
                content = raw.decode("utf-8")
            except UnicodeDecodeError:
                return JSONResponse(
                    {"error": "Only .txt/.md (UTF-8) files are supported."},
                    status_code=400,
                )
        elif text:
            doc_name = "pasted-text.txt"
            content = text
        else:
            return JSONResponse(
                {"error": "Send a file or raw text."}, status_code=400
            )
        n = ingest_text(doc_name, content)
    except RuntimeError as e:
        return JSONResponse({"error": str(e)}, status_code=503)
    conn = db()
    total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    conn.close()
    return {"doc": doc_name, "chunks_added": n, "total_chunks": total}


@app.post("/ask")
def ask(body: AskIn):
    try:
        sources = retrieve(body.question)
    except RuntimeError as e:
        return JSONResponse({"error": str(e)}, status_code=503)
    if not sources:
        return {
            "answer": "No documents are indexed yet. Upload a file first.",
            "sources": [],
            "mode": "empty",
        }
    try:
        answer = generate_answer(body.question, sources)
    except RuntimeError as e:
        return JSONResponse({"error": str(e)}, status_code=503)
    return {
        "answer": answer,
        "sources": [
            {"doc": s["doc"], "chunk": s["chunk"], "text": s["text"]}
            for s in sources
        ],
        "mode": "gemini",
    }

"""RAG quickstart: FastAPI + ChromaDB + fastembed embeddings, optional LLM.

Run:  source .venv/bin/activate && uvicorn app:app --port 8000
LLM provider priority: GROQ_API_KEY, then GEMINI_API_KEY, then Ollama
(local, OLLAMA_MODEL or auto-detected Hermes model), then extractive mode.
With no provider at all the app still works in extractive mode: it returns
the top retrieved chunks and says clearly which key would enable generation.
"""
import os
import re
from pathlib import Path

import httpx

import chromadb
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastembed import TextEmbedding
from pydantic import BaseModel

BASE = Path(__file__).parent
SEED_DIR = BASE / "seed_docs"
CHROMA_DIR = BASE / "chroma_db"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 4

app = FastAPI(title="RAG Quickstart")
_embedder = None
_client = None
_collection = None


def embedder():
    """fastembed running the same all-MiniLM-L6-v2 model via ONNX.

    Real vector embeddings (384-dim), not keyword search. fastembed was
    chosen over sentence-transformers here because the full torch install
    (550MB+ CUDA wheel) was too heavy/slow for this VM.
    """
    global _embedder
    if _embedder is None:
        _embedder = TextEmbedding(model_name=MODEL_NAME)
    return _embedder


def encode(texts):
    return [e.tolist() for e in embedder().embed(texts)]


def collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = _client.get_or_create_collection("docs")
    return _collection


def chunk_text(text: str, size: int = 800, overlap: int = 100):
    """Split text into overlapping chunks on sentence/word boundaries."""
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + size]))
        i += size - overlap
    return [c for c in chunks if c.strip()]


def ingest_text(doc_name: str, text: str):
    col = collection()
    chunks = chunk_text(text)
    if not chunks:
        return 0
    embs = encode(chunks)
    ids = [f"{doc_name}#{n}" for n in range(len(chunks))]
    col.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embs,
        metadatas=[{"doc": doc_name, "chunk": n} for n in range(len(chunks))],
    )
    return len(chunks)


def seed_if_empty():
    col = collection()
    if col.count() > 0:
        return col.count()
    total = 0
    for path in sorted(SEED_DIR.glob("*.md")):
        total += ingest_text(path.name, path.read_text())
    return total


@app.on_event("startup")
def startup():
    seed_if_empty()


def active_provider():
    """Return (name, detail) for the first available LLM provider, else None.

    Priority: Groq, then Gemini, then local Ollama. No keys are required
    to run the app — with none of these, ask() falls back to extractive.
    """
    if os.getenv("GROQ_API_KEY"):
        return ("groq", "openai/gpt-oss-20b")
    if os.getenv("GEMINI_API_KEY"):
        return ("gemini", "gemini-2.5-flash")
    model = detect_ollama_model()
    if model:
        return ("ollama", model)
    return None


def detect_ollama_model():
    """Pick OLLAMA_MODEL if set, else a Hermes model if installed, else any
    installed model. Returns None when Ollama is not reachable."""
    forced = os.getenv("OLLAMA_MODEL")
    if forced:
        return forced
    try:
        resp = httpx.get("http://localhost:11434/api/tags", timeout=3)
        models = [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        return None
    for name in models:
        if "hermes" in name.lower():
            return name
    return models[0] if models else None


@app.get("/health")
def health():
    provider = active_provider()
    return {
        "status": "ok",
        "chunks_indexed": collection().count(),
        "embedding_model": MODEL_NAME,
        "llm": f"{provider[0]} ({provider[1]})" if provider else "extractive (no GROQ_API_KEY, GEMINI_API_KEY, or Ollama model found)",
    }


@app.get("/")
def ui():
    return FileResponse(BASE / "index.html")


class AskIn(BaseModel):
    question: str


@app.post("/ingest")
async def ingest(file: UploadFile = File(None), text: str = Form(None)):
    if file is not None:
        raw = await file.read()
        doc_name = file.filename or "upload.txt"
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError:
            return JSONResponse({"error": "Only .txt/.md (UTF-8) files are supported."}, status_code=400)
    elif text:
        doc_name = "pasted-text.txt"
        content = text
    else:
        return JSONResponse({"error": "Send a file or raw text."}, status_code=400)
    n = ingest_text(doc_name, content)
    return {"doc": doc_name, "chunks_added": n, "total_chunks": collection().count()}


def retrieve(question: str, k: int = TOP_K):
    col = collection()
    if col.count() == 0:
        return []
    emb = encode([question])
    res = col.query(query_embeddings=emb, n_results=min(k, col.count()))
    sources = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        sources.append({"doc": meta["doc"], "chunk": meta["chunk"], "distance": round(float(dist), 4), "text": doc})
    return sources


def highlight(text: str, question: str) -> str:
    """Bold the question words found in a chunk (simple, for extractive mode)."""
    words = {w.strip("?,.") for w in question.lower().split() if len(w) > 3}
    out = text
    for w in sorted(words, key=len, reverse=True):
        out = re.sub(rf"(?i)\b({re.escape(w)})\b", r"**\1**", out, count=2)
    return out


SYSTEM_PROMPT = (
    "Answer ONLY from the context below. Cite sources like [doc chunk n]. "
    "If the context does not contain the answer, say so plainly."
)


def _groq_answer(prompt: str) -> str:
    from groq import Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    resp = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content


def _gemini_answer(prompt: str) -> str:
    """Gemini via REST (no extra SDK needed). Tries 2.5-flash, then 2.0-flash."""
    key = os.getenv("GEMINI_API_KEY")
    for model in ("gemini-2.5-flash", "gemini-2.0-flash"):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
        resp = httpx.post(url, json={
            "contents": [{"parts": [{"text": SYSTEM_PROMPT + "\n\n" + prompt}]}],
            "generationConfig": {"temperature": 0.2},
        }, timeout=60)
        if resp.status_code == 200:
            parts = resp.json()["candidates"][0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts)
        if resp.status_code not in (400, 404):
            resp.raise_for_status()
    raise RuntimeError("Gemini: neither gemini-2.5-flash nor gemini-2.0-flash accepted the request")


def _ollama_answer(prompt: str, model: str) -> str:
    resp = httpx.post("http://localhost:11434/api/generate", json={
        "model": model,
        "prompt": SYSTEM_PROMPT + "\n\n" + prompt,
        "stream": False,
        "options": {"temperature": 0.2},
    }, timeout=180)
    resp.raise_for_status()
    return resp.json()["response"]


def generate_answer(question: str, sources):
    """Try providers in priority order. Returns (answer, mode, failed_note).

    mode is one of groq/gemini/ollama/extractive. A provider failure never
    crashes the endpoint: it falls back to extractive mode with a note.
    """
    context = "\n\n".join(f"[{s['doc']} chunk {s['chunk']}] {s['text']}" for s in sources)
    prompt = f"Context:\n{context}\n\nQuestion: {question}"
    provider = active_provider()
    if provider:
        name, detail = provider
        try:
            if name == "groq":
                return _groq_answer(prompt), "groq", None
            if name == "gemini":
                return _gemini_answer(prompt), "gemini", None
            return _ollama_answer(prompt, detail), "ollama", None
        except Exception as e:
            return None, "extractive", f"({name} call failed: {e}. Falling back to extractive mode.)"
    return None, "extractive", None


@app.post("/ask")
def ask(body: AskIn):
    sources = retrieve(body.question)
    if not sources:
        return {"answer": "No documents are indexed yet. Upload a file first.", "sources": [], "mode": "empty"}
    answer, mode, failed_note = generate_answer(body.question, sources)
    if mode == "extractive":
        top = sources[0]
        passage = highlight(top["text"], body.question)
        if failed_note:
            answer = failed_note + "\n\nMost relevant passage:\n\n" + passage
        else:
            answer = (
                "Extractive mode — no LLM provider found. Set GROQ_API_KEY or "
                "GEMINI_API_KEY, or run Ollama locally (see README), for "
                "generated answers.\n\nMost relevant passage:\n\n" + passage
            )
    return {
        "answer": answer,
        "sources": [{"doc": s["doc"], "chunk": s["chunk"], "text": s["text"]} for s in sources],
        "mode": mode,
    }

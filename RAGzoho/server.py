#!/usr/bin/env python3
"""
REST API endpoint for querying client-specific knowledge bases.

Run:
    python server.py

The server uses FastAPI and exposes interactive Swagger docs at:
    http://127.0.0.1:8000/docs
"""

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

load_dotenv()

ARTIFACTS_ROOT = Path("artifacts/clients")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise SystemExit("OPENAI_API_KEY missing in environment.")

try:
    import faiss  # type: ignore

    FAISS_OK = True
except Exception:
    FAISS_OK = False

try:
    from openai import OpenAI
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"Failed to import OpenAI SDK: {exc}") from exc

openai_client = OpenAI(api_key=OPENAI_API_KEY)


class QueryRequest(BaseModel):
    email: str = Field(..., description="Client email to query")
    question: str = Field(..., description="Natural-language question about the client")
    top_k: int = Field(6, ge=1, le=20, description="Maximum number of chunks to retrieve")


class QueryResponse(BaseModel):
    answer: str
    citations: List[Dict[str, str]]
    retrieved: List[Dict[str, str]]


app = FastAPI(
    title="RAGZoho Query API",
    description="Query client knowledge bases extracted from Zoho CRM.",
    version="1.0.0",
)


def require_client(email: str) -> Path:
    client_dir = ARTIFACTS_ROOT / email
    if not client_dir.exists():
        raise HTTPException(status_code=404, detail=f"Client '{email}' not found.")
    chunks_file = client_dir / "chunks.jsonl"
    embeddings_file = client_dir / "embeddings.npy"
    meta_file = client_dir / "chunks_meta.json"
    if not (chunks_file.exists() and embeddings_file.exists() and meta_file.exists()):
        raise HTTPException(
            status_code=400,
            detail=f"Client '{email}' is missing required artifacts. Re-run the pipeline.",
        )
    return client_dir


@lru_cache(maxsize=256)
def load_metadata(email: str):
    client_dir = require_client(email)
    embeddings = np.fromfile(str(client_dir / "embeddings.npy"), dtype=np.float32)
    meta: List[Dict[str, str]] = json.loads((client_dir / "chunks_meta.json").read_text(encoding="utf-8"))
    if len(meta) == 0:
        raise HTTPException(status_code=400, detail=f"No chunks indexed for {email}.")
    dim = int(embeddings.size / len(meta))
    matrix = embeddings.reshape(-1, dim)
    index = None
    if FAISS_OK:
        faiss_path = client_dir / "faiss.index"
        if faiss_path.exists():
            index = faiss.read_index(str(faiss_path))
    chunk_map: Dict[int, Dict[str, str]] = {}
    with (client_dir / "chunks.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            chunk_map[int(record["id"])] = record
    return matrix, meta, index, chunk_map


def embed_query(question: str) -> np.ndarray:
    response = openai_client.embeddings.create(model="text-embedding-3-small", input=[question])
    vec = np.array(response.data[0].embedding, dtype="float32")
    vec = vec / (np.linalg.norm(vec) + 1e-12)
    return vec.astype("float32")


def retrieve_chunks(email: str, question: str, top_k: int):
    matrix, meta, index, chunk_map = load_metadata(email)
    query_vec = embed_query(question)
    if index is not None:
        scores, ids = index.search(query_vec.reshape(1, -1), top_k)
        hits = []
        for score, idx in zip(scores[0], ids[0]):
            if 0 <= idx < len(meta):
                info = meta[int(idx)].copy()
                info["text"] = chunk_map.get(int(info["id"]), {}).get("text", "")[:1200]
                hits.append({"score": float(score), "meta": info})
        return hits
    similarities = matrix @ query_vec
    top_indices = similarities.argsort()[-top_k:][::-1]
    hits = []
    for idx in top_indices:
        info = meta[int(idx)].copy()
        info["text"] = chunk_map.get(int(info["id"]), {}).get("text", "")[:1200]
        hits.append({"score": float(similarities[int(idx)]), "meta": info})
    return hits


def build_prompt(email: str, question: str, hits: List[Dict[str, Dict[str, str]]]) -> str:
    if not hits:
        return (
            f"You are a helpful assistant. No context is available for {email}. "
            "If unsure, answer: 'I don't know.'\n\nQUESTION:\n" + question
        )
    parts = []
    for hit in hits:
        meta = hit["meta"]
        source = Path(meta["source"]).name
        label = f"[{source}#chunk{meta['chunk_id']}]"
        text = meta.get("text", "")
        parts.append(f"{label}\n{text}")
    context = "\n\n---\n\n".join(parts)
    return (
        f"You are a precise assistant. Answer ONLY using the provided CONTEXT for client {email}. "
        'If the answer is missing, reply: "I don\'t know." Include citations like [file#chunkN].\n\n'
        f"CONTEXT:\n{context}\n\nQUESTION:\n{question}\n"
    )


def call_llm(prompt: str) -> str:
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a precise financial research assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=500,
    )
    content = response.choices[0].message.content or ""
    return content.encode("ascii", "ignore").decode("ascii")


def query_service(email: str, question: str, top_k: int):
    hits = retrieve_chunks(email, question, top_k)
    prompt = build_prompt(email, question, hits)
    answer = call_llm(prompt)
    citations = [
        {"source": Path(hit["meta"]["source"]).name, "chunk": str(hit["meta"]["chunk_id"])}
        for hit in hits
    ]
    retrieved = [
        {
            "score": f"{hit['score']:.4f}",
            "source": Path(hit["meta"]["source"]).name,
            "chunk_id": str(hit["meta"]["chunk_id"]),
            "preview": hit["meta"].get("text", ""),
        }
        for hit in hits
    ]
    return answer, citations, retrieved


@app.post("/query", response_model=QueryResponse, summary="Ask a question about a specific client")
def query_endpoint(payload: QueryRequest = Depends()):
    answer, citations, retrieved = query_service(payload.email, payload.question, payload.top_k)
    return QueryResponse(answer=answer, citations=citations, retrieved=retrieved)


@app.get("/clients", summary="List registered client emails")
def list_clients():
    if not ARTIFACTS_ROOT.exists():
        return {"clients": []}
    clients = [p.name for p in ARTIFACTS_ROOT.iterdir() if p.is_dir()]
    return {"clients": sorted(clients)}


if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)





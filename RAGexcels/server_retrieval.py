#!/usr/bin/env python3
"""
Retrieval-only API over local artifacts

Usage:
  1) pip install -r requirements.txt
  2) uvicorn server_retrieval:app --host 0.0.0.0 --port 8000

Endpoints:
  - GET  /health                 -> {status: ok}
  - POST /retrieve               -> returns contexts (text previews) and scores
  - POST /retrieve_prompt        -> returns a minimal prompt string your model can send to its chat API

Environment:
  - ARTIFACTS_DIR (optional) default: artifacts
"""
import os
from pathlib import Path
from typing import List

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_query import load_index, embed_query, search, build_prompt


ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))

app = FastAPI(title="RAG Retrieval API", version="1.0.0")

# Allow calling from your app / model runtime
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RetrieveIn(BaseModel):
    question: str
    top_n: int = 10
    final_k: int = 1  # return only the single best context by default


@app.on_event("startup")
def _startup() -> None:
    # Load FAISS index + metadata once at server startup
    app.state.loader = load_index(ARTIFACTS_DIR)


def _clean_context(text: str) -> str:
    """Extract only the answer portion from a row like
    "Question: ... | Answer: ... | ...". If 'Answer:' is present,
    return the substring after it; otherwise return the original text.
    Also trims whitespace and squashes repeated spaces.
    """
    if not text:
        return ""
    lower = text.lower()
    key = "answer:"
    if key in lower:
        # Find index in original string using lower-cased position
        idx = lower.find(key)
        # Skip the label length to start after 'Answer:'
        ans = text[idx + len("answer:") :].strip()
        return ans
    return text.strip()


def _trim_to_sentence(text: str, max_chars: int = 450) -> str:
    """Keep only the first sentence (or up to max_chars)."""
    if not text:
        return ""
    out = text.split(".", 1)[0].strip()
    if not out:
        out = text.strip()
    if len(out) > max_chars:
        out = out[:max_chars].rstrip() + "..."
    return out


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/retrieve")
def retrieve(body: RetrieveIn, request: Request):
    loader = request.app.state.loader
    qv = embed_query(body.question)
    hits = search(loader, qv, top_n=body.top_n)
    # Return only the top final_k cleaned, brief contexts to avoid overwhelming the model
    contexts: List[str] = [
        _trim_to_sentence(_clean_context(h["meta"].get("text_preview", "")))
        for h in hits[: max(1, body.final_k)]
    ]
    scores: List[float] = [h["score"] for h in hits]
    return {
        "question": body.question,
        "contexts": contexts,
        "scores": scores[: len(contexts)],
    }


@app.post("/retrieve_prompt")
def retrieve_prompt(body: RetrieveIn, request: Request):
    loader = request.app.state.loader
    qv = embed_query(body.question)
    hits = search(loader, qv, top_n=body.top_n)
    # Build a concise prompt your model can send to https://api.wealthai1.in/api/chat
    # Uses existing build_prompt but strips citations from the instruction.
    minimal_prompt = (
        "Answer the question directly based only on the context. "
        "Respond with a single concise sentence, nothing more.\n\n"
        "CONTEXT:\n"
    )
    # Join top-k brief contexts as plain text blocks
    contexts = [
        _trim_to_sentence(_clean_context(h["meta"].get("text_preview", "")))
        for h in hits[: max(1, body.final_k)]
    ]
    minimal_prompt += ("\n\n---\n\n").join(contexts)
    minimal_prompt += f"\n\nQUESTION:\n{body.question}\n\nANSWER:"

    return {
        "question": body.question,
        "prompt": minimal_prompt,
        "used_contexts": contexts,
    }



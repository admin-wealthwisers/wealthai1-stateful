#!/usr/bin/env python3
"""
Unified FastAPI server that exposes all RAG workflows (Excels, NGEN, Zoho)
under a single process. Run with:

    uvicorn server:app --host 0.0.0.0 --port 8005
"""
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env")


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


excels_module = _load_module("ragexcels_rag_query", BASE_DIR / "RAGexcels" / "rag_query.py")
ngen_module = _load_module("ragngen_rag_query", BASE_DIR / "RAGngen" / "rag_query.py")
zoho_server_module = _load_module("ragzoho_server", BASE_DIR / "RAGzoho" / "server.py")


EXCELS_ARTIFACTS = Path(os.getenv("EXCELS_ARTIFACTS_DIR", BASE_DIR / "RAGexcels" / "artifacts"))
NGEN_ARTIFACTS = Path(os.getenv("NGEN_ARTIFACTS_DIR", BASE_DIR / "RAGngen" / "artifacts"))


app = FastAPI(title="Unified RAG Backend", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ExcelsRetrieveIn(BaseModel):
    question: str
    top_n: int = Field(10, ge=1, le=200)
    final_k: int = Field(1, ge=1, le=50)


class ExcelsRetrieveOut(BaseModel):
    question: str
    contexts: List[str]
    scores: List[float]


class ExcelsPromptOut(BaseModel):
    question: str
    prompt: str
    used_contexts: List[str]


class NgenQAIn(BaseModel):
    question: str
    top_n: int = Field(50, ge=1, le=200)
    final_k: int = Field(10, ge=1, le=100)


class NgenQAOut(BaseModel):
    question: str
    answer: str
    prompt_used: str


class ZohoQueryIn(BaseModel):
    email: str
    question: str
    top_k: int = Field(6, ge=1, le=20)


class ZohoQueryOut(BaseModel):
    question: str
    answer: str
    citations: List[Dict[str, str]]
    retrieved: List[Dict[str, str]]


_excels_loader: Optional[Dict[str, Any]] = None
_ngen_loader: Optional[Dict[str, Any]] = None


def _clean_context(text: str) -> str:
    if not text:
        return ""
    lower = text.lower()
    key = "answer:"
    if key in lower:
        idx = lower.find(key)
        return text[idx + len("answer:") :].strip()
    return text.strip()


def _trim_sentence(text: str, max_chars: int = 450) -> str:
    if not text:
        return ""
    sentence = text.split(".", 1)[0].strip() or text.strip()
    if len(sentence) > max_chars:
        sentence = sentence[:max_chars].rstrip() + "..."
    return sentence


@app.on_event("startup")
def _startup_loaders():
    global _excels_loader, _ngen_loader
    if _excels_loader is None:
        _excels_loader = excels_module.load_index(EXCELS_ARTIFACTS)
    if _ngen_loader is None:
        _ngen_loader = ngen_module.load_index(NGEN_ARTIFACTS)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/excels/retrieve", response_model=ExcelsRetrieveOut)
def excels_retrieve(body: ExcelsRetrieveIn):
    if _excels_loader is None:
        raise HTTPException(status_code=500, detail="Excels loader not initialized")
    qv = excels_module.embed_query(body.question)
    hits = excels_module.search(_excels_loader, qv, top_n=body.top_n)
    contexts = [
        _trim_sentence(_clean_context(h["meta"].get("text_preview", "")))
        for h in hits[: max(1, body.final_k)]
    ]
    scores = [h["score"] for h in hits[: len(contexts)]]
    return ExcelsRetrieveOut(question=body.question, contexts=contexts, scores=scores)


@app.post("/excels/retrieve_prompt", response_model=ExcelsPromptOut)
def excels_retrieve_prompt(body: ExcelsRetrieveIn):
    if _excels_loader is None:
        raise HTTPException(status_code=500, detail="Excels loader not initialized")
    qv = excels_module.embed_query(body.question)
    hits = excels_module.search(_excels_loader, qv, top_n=body.top_n)
    contexts = [
        _trim_sentence(_clean_context(h["meta"].get("text_preview", "")))
        for h in hits[: max(1, body.final_k)]
    ]
    prompt = (
        "Answer the question directly based only on the context. "
        "Respond with a single concise sentence, nothing more.\n\nCONTEXT:\n"
        + ("\n\n---\n\n".join(contexts))
        + f"\n\nQUESTION:\n{body.question}\n\nANSWER:"
    )
    return ExcelsPromptOut(question=body.question, prompt=prompt, used_contexts=contexts)


@app.post("/ngen/qa", response_model=NgenQAOut)
def ngen_qa(body: NgenQAIn):
    if _ngen_loader is None:
        raise HTTPException(status_code=500, detail="NGEN loader not initialized")
    qv = ngen_module.embed_query(body.question)
    hits = ngen_module.search(_ngen_loader, qv, top_n=body.top_n, question=body.question)
    retrieved = hits[: max(1, body.final_k)]
    prompt = ngen_module.build_prompt(body.question, retrieved)
    answer = ngen_module.ask_llm(prompt, temperature=0.0, max_tokens=300)
    return NgenQAOut(question=body.question, answer=answer, prompt_used=prompt)


@app.post("/zoho/query", response_model=ZohoQueryOut)
def zoho_query(body: ZohoQueryIn):
    answer, citations, retrieved = zoho_server_module.query_service(
        body.email, body.question, body.top_k
    )
    return ZohoQueryOut(
        question=body.question,
        answer=answer,
        citations=citations,
        retrieved=retrieved,
    )


@app.get("/zoho/clients")
def zoho_clients():
    return zoho_server_module.list_clients()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8005, reload=False)


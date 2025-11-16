#!/usr/bin/env python3
"""
Retrieval-only API over local artifacts

Usage:
  1) pip install -r requirements.txt
  2) uvicorn server_retrieval:app --host 0.0.0.0 --port 8000

Endpoints:
  - GET  /health                 -> {status: ok}
  - POST /qa                     -> returns only question and answer

Environment:
  - ARTIFACTS_DIR (optional) default: artifacts
"""
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_query import load_index, embed_query, search, build_prompt, ask_llm, extract_aum_from_chunk, extract_combined_market_value


ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load FAISS index + metadata once at server startup
    app.state.loader = load_index(ARTIFACTS_DIR)
    yield
    # Shutdown: cleanup if needed (currently nothing to clean up)


app = FastAPI(title="RAG Retrieval API", version="1.0.0", lifespan=lifespan)

# Allow calling from your app / model runtime
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QAIn(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}




@app.post("/qa")
def qa(body: QAIn, request: Request):
    """Question Answering - returns only question and answer using artifacts folder."""
    loader = request.app.state.loader
    qv = embed_query(body.question)
    
    question_lower = body.question.lower()
    
    # For different question types, retrieve appropriate number of chunks
    if "nfo" in question_lower or "new fund offer" in question_lower:
        top_n = 100  # Need to find all NFOs
        final_k = 63  # Use all NFO chunks (there are 63 NFOs)
    elif ("aaa" in question_lower or "aa" in question_lower or "a rated" in question_lower) and ("bond" in question_lower or "debt" in question_lower):
        top_n = 100  # Need to find multiple debt funds for comparison
        final_k = 30  # Use more chunks to get comprehensive bond rating data
    elif "top 25" in question_lower or "25 stocks" in question_lower:
        top_n = 50  # Need to find all 25 stocks
        final_k = 30  # Use more chunks to get all stocks
    elif "amc" in question_lower:
        top_n = 100  # Get more candidates for AMC aggregation
        final_k = 30  # Use more chunks for context
    else:
        top_n = 50
        final_k = 10
    
    # Search many candidates to find the right data
    hits = search(loader, qv, top_n=top_n, question=body.question)
    
    # Use top chunks to ensure we get the relevant data
    retrieved = [{"meta": h["meta"], "chunk_text": h.get("chunk_text", "")} for h in hits[:final_k]]
    
    # Try direct extraction first for combined market value questions
    if ("top 25" in question_lower or "25 stocks" in question_lower) and ("combined" in question_lower or "total" in question_lower or "market value" in question_lower):
        combined_value = extract_combined_market_value(retrieved)
        if combined_value:
            return {
                "question": body.question,
                "answer": combined_value,
            }
    
    # Try direct extraction first for AUM questions
    if "aum" in question_lower:
        for r in retrieved:
            chunk_text = r.get("chunk_text", "")
            source = r.get("meta", {}).get("source", "").lower()
            # Check if it's from category averages and matches the category
            if "category averages" in source:
                if "large cap" in question_lower and "mid" not in question_lower:
                    if "large cap fund" in chunk_text.lower() and "mid" not in chunk_text.lower():
                        aum_value = extract_aum_from_chunk(chunk_text)
                        if aum_value:
                            return {
                                "question": body.question,
                                "answer": aum_value,
                            }
    
    # Build prompt with improved instructions
    prompt = build_prompt(body.question, retrieved)
    answer = ask_llm(prompt, temperature=0.0, max_tokens=200)
    
    return {
        "question": body.question,
        "answer": answer,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


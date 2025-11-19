#!/usr/bin/env python3
# rag_indexer.py
import os, json, argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv

# Try to get from environment first (set by server.py), if not found, load from .env
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    # Load .env from project root (one level up from this file)
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(dotenv_path=env_path, override=True)
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

import google.generativeai as genai

# Set up Gemini API
# IMPORTANT: Set GEMINI_API_KEY in environment variable or .env file
if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY not set. Please set it in your .env file or environment variable.\n"
        "Get your API key from: https://makersuite.google.com/app/apikey"
    )
genai.configure(api_key=GEMINI_API_KEY)

EMBED_MODEL = "models/text-embedding-004"  # Google Gemini embedding model

import faiss  # make sure faiss-cpu is installed

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def embed_batches(texts: List[str], batch_size: int = 100) -> np.ndarray:
    # Note: Gemini embedding API processes texts individually
    vectors = []
    total = len(texts)
    for i, text in enumerate(texts):
        if (i + 1) % 50 == 0:
            print(f"[index] embedded {i + 1}/{total} chunks...")
        try:
            result = genai.embed_content(
                model=EMBED_MODEL,
                content=text,
                task_type="retrieval_document"
            )
            embedding = result['embedding']
            vectors.append(embedding)
        except Exception as e:
            error_msg = str(e)
            if "leaked" in error_msg.lower() or "403" in error_msg or "PermissionDenied" in error_msg:
                print("\n" + "="*80)
                print("[ERROR] API Key Blocked for Embeddings!")
                print("="*80)
                print("Your Google Gemini API key has been flagged as leaked.")
                print("Even though it might work for chat, the embedding API blocks it.")
                print("\nSOLUTION: Get a NEW API key:")
                print("1. Go to: https://makersuite.google.com/app/apikey")
                print("2. Create a NEW API key")
                print("3. Update your .env file with the new key")
                print("4. Restart the indexing process")
                print("="*80 + "\n")
                raise RuntimeError("API key blocked for embeddings. Get a new key.")
            print(f"Failed to embed text {i}: {e}")
            # Get embedding dimension from first successful embedding or use default
            if vectors:
                dim = len(vectors[0])
            else:
                dim = 768  # Default embedding dimension for text-embedding-004
            # Add zero vector as fallback
            vectors.append([0.0] * dim)
    
    arr = np.array(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True); norms[norms==0] = 1.0
    return arr / norms

def build_index(artifacts_dir: str = "artifacts"):
    artifacts = Path(artifacts_dir)
    chunks_path = artifacts / "chunks.jsonl"
    meta_path   = artifacts / "chunks_meta.json"
    emb_path    = artifacts / "embeddings.npy"
    faiss_path  = artifacts / "faiss.index"

    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set (.env or environment)")

    chunks = read_jsonl(chunks_path)
    texts = [c["text"] for c in chunks]
    print(f"[index] embedding {len(texts)} chunks with {EMBED_MODEL} ...")
    emb = embed_batches(texts)

    np.save(str(emb_path), emb)
    print(f"[index] saved embeddings -> {emb_path}")
    print(f"[index] meta -> {meta_path}")

    dim = emb.shape[1]
    faiss.normalize_L2(emb)
    idx = faiss.IndexHNSWFlat(dim, 32)
    idx.hnsw.efConstruction = 200
    idx.add(emb)
    faiss.write_index(idx, str(faiss_path))
    print(f"[index] FAISS index -> {faiss_path}")

def _cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts_dir", default="artifacts")
    args = ap.parse_args()
    build_index(args.artifacts_dir)

if __name__ == "__main__":
    _cli()

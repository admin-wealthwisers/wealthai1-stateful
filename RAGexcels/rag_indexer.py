#!/usr/bin/env python3
# rag_indexer.py
import os, json, argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBED_MODEL = "text-embedding-3-small"  # or "text-embedding-3-large" for higher accuracy (costlier)

import faiss  # make sure faiss-cpu is installed

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def embed_batches(texts: List[str], batch_size: int = 256) -> np.ndarray:
    vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        vectors.extend([d.embedding for d in resp.data])
    arr = np.array(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True); norms[norms==0] = 1.0
    return arr / norms

def build_index(artifacts_dir: str = "artifacts"):
    artifacts = Path(artifacts_dir)
    chunks_path = artifacts / "chunks.jsonl"
    meta_path   = artifacts / "chunks_meta.json"
    emb_path    = artifacts / "embeddings.npy"
    faiss_path  = artifacts / "faiss.index"

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set (.env or environment)")

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

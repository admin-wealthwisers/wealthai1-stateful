#!/usr/bin/env python3
# rag_query.py
import os, json, argparse
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL  = "gpt-4o-mini"

import faiss

def load_index(artifacts_dir: Path):
    meta = json.loads((artifacts_dir / "chunks_meta.json").read_text(encoding="utf-8"))
    emb_path = artifacts_dir / "embeddings.npy"
    faiss_path = artifacts_dir / "faiss.index"
    if not emb_path.exists() or not faiss_path.exists():
        raise FileNotFoundError("Missing embeddings or faiss.index; run chunker + indexer first.")
    idx = faiss.read_index(str(faiss_path))
    return {"index": idx, "meta": meta}

def embed_query(q: str) -> np.ndarray:
    resp = client.embeddings.create(model=EMBED_MODEL, input=[q])
    v = np.array(resp.data[0].embedding, dtype=np.float32)
    v = v / (np.linalg.norm(v) + 1e-12)
    return v

def search(loader, qvec: np.ndarray, top_n: int):
    meta = loader["meta"]
    q = np.array([qvec], dtype=np.float32)
    faiss.normalize_L2(q)
    D, I = loader["index"].search(q, top_n)
    out = []
    for score, idx in zip(D[0], I[0]):
        if 0 <= idx < len(meta):
            out.append({"score": float(score), "meta": meta[idx]})
    return out

def build_prompt(question: str, retrieved: List[Dict[str,Any]], max_chars: int = 3800):
    parts, used = [], 0
    for r in retrieved:
        m = r["meta"]
        # citation label differs for excel vs pdf
        if m.get("kind") == "pdf":
            tag = f"[{m['source']}#page{m['page']}#chunk{m['chunk_index']}]"
        else:
            tag = f"[{m['source']}#row{m['row_index']}#chunk{m['chunk_index']}]"
        s = m["text_preview"]
        if used + len(s) > max_chars: break
        parts.append(f"{tag}\n{s}")
        used += len(s)
    ctx = "\n\n---\n\n".join(parts)
    return f"""Answer the question directly and concisely based on the context provided. Give only what is asked, nothing more.

CONTEXT:
{ctx}

QUESTION:
{question}

ANSWER:"""

def ask_llm(prompt: str, temperature: float = 0.0, max_tokens: int = 600) -> str:
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role":"system","content":"You are a helpful assistant. Answer questions directly and concisely. Give only what is asked, nothing more."},
            {"role":"user","content":prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )
    try:
        return resp.choices[0].message.content
    except Exception:
        return str(resp)

def _cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts_dir", default="artifacts")
    ap.add_argument("--question", required=True)
    ap.add_argument("--top_n", type=int, default=50)
    ap.add_argument("--final_k", type=int, default=6)
    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set. Put it in .env")

    artifacts = Path(args.artifacts_dir)
    loader = load_index(artifacts)

    qv = embed_query(args.question)
    retrieved = search(loader, qv, top_n=args.top_n)

    prompt = build_prompt(args.question, retrieved[:args.final_k])
    answer = ask_llm(prompt)
    print(answer)

if __name__ == "__main__":
    _cli()

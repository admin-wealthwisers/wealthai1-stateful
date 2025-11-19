#!/usr/bin/env python3
# rag_query.py
import os, json, argparse
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

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

EMBED_MODEL = "models/text-embedding-004"
CHAT_MODEL  = "gemini-1.5-flash-latest"  # Using Gemini 1.5 Flash (latest version)

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
    # Use Gemini embedding model
    try:
        result = genai.embed_content(
            model=EMBED_MODEL,
            content=q,
            task_type="retrieval_query"
        )
        v = np.array(result['embedding'], dtype=np.float32)
        v = v / (np.linalg.norm(v) + 1e-12)
        return v
    except Exception as e:
        error_msg = str(e)
        if "leaked" in error_msg.lower() or "403" in error_msg or "PermissionDenied" in error_msg:
            raise RuntimeError(
                f"API Key Error: Your Google Gemini API key has been flagged as leaked and is blocked for embedding operations.\n"
                f"Even though it might work for chat, the embedding API has stricter security checks.\n\n"
                f"SOLUTION: You MUST get a NEW API key:\n"
                f"1. Go to: https://makersuite.google.com/app/apikey\n"
                f"2. Create a NEW API key\n"
                f"3. Update your .env file with the new key\n"
                f"4. Restart the server\n\n"
                f"Original error: {error_msg}"
            ) from e
        raise

def search(loader, qvec: np.ndarray, top_n: int):
    meta = loader["meta"]
    index = loader["index"]
    
    # Check dimension mismatch
    query_dim = qvec.shape[0]
    index_dim = index.d
    
    if query_dim != index_dim:
        raise RuntimeError(
            f"Embedding dimension mismatch!\n"
            f"  Query vector dimension: {query_dim}\n"
            f"  FAISS index dimension: {index_dim}\n\n"
            f"This happens when the index was built with a different embedding model.\n"
            f"SOLUTION: Rebuild the index with Gemini embeddings:\n"
            f"  1. Go to RAGexcel directory\n"
            f"  2. Run: python rag_indexer.py --artifacts_dir artifacts\n"
            f"  3. This will regenerate embeddings.npy and faiss.index with Gemini embeddings\n\n"
            f"Note: The index was likely built with OpenAI embeddings ({index_dim} dims) "
            f"but you're now using Gemini embeddings ({query_dim} dims)."
        )
    
    q = np.array([qvec], dtype=np.float32)
    faiss.normalize_L2(q)
    D, I = index.search(q, top_n)
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
    # Use Gemini chat model - try multiple model names for compatibility
    model_names = [
        "gemini-1.5-flash-latest",
        "gemini-1.5-flash",
        "gemini-2.0-flash-exp",
        "gemini-pro"
    ]
    
    # Combine system and user message for Gemini
    full_prompt = "You are a helpful assistant. Answer questions directly and concisely. Give only what is asked, nothing more.\n\n" + prompt
    
    generation_config = genai.types.GenerationConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
    )
    
    last_error = None
    for model_name in model_names:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(
                full_prompt,
                generation_config=generation_config
            )
            if response and response.text:
                return response.text
        except Exception as e:
            last_error = e
            continue  # Try next model
    
    # If all models failed, return error with details
    error_msg = str(last_error) if last_error else "Unknown error"
    return f"Error: Failed to generate response with all available models. Last error: {error_msg}"

def _cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts_dir", default="artifacts")
    ap.add_argument("--question", required=True)
    ap.add_argument("--top_n", type=int, default=50)
    ap.add_argument("--final_k", type=int, default=6)
    args = ap.parse_args()

    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set. Put it in .env or set it in the code")

    artifacts = Path(args.artifacts_dir)
    loader = load_index(artifacts)

    qv = embed_query(args.question)
    retrieved = search(loader, qv, top_n=args.top_n)

    prompt = build_prompt(args.question, retrieved[:args.final_k])
    answer = ask_llm(prompt)
    print(answer)

if __name__ == "__main__":
    _cli()

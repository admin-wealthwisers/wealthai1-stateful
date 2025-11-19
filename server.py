#!/usr/bin/env python3
"""
Unified RAG Server - Combines RAGzoho, RAGngen, and RAGexcel on a single port

Usage:
  1) pip install -r requirements.txt  (installs all dependencies for all three RAGs)
  2) python server.py
     or
     uvicorn server:app --host 0.0.0.0 --port 8000

Note: All dependencies are installed from the root requirements.txt file.
      No need to install separately for each RAG directory.

Endpoints:
  Zoho RAG:
    - POST /zoho/query      -> Query client knowledge bases from Zoho CRM
    - GET  /zoho/clients    -> List registered client emails
  
  NGEN RAG:
    - GET  /ngen/health     -> Health check
    - POST /ngen/qa         -> Question answering for NGEN Markets data
    - POST /ngen/retrieve   -> Retrieve raw chunks for NGEN Markets query
  
  Excel RAG:
    - GET  /excel/health    -> Health check
    - POST /excel/qa        -> Question answering for Excel Q&A data
    - POST /excel/retrieve  -> Retrieve raw chunks for Excel Q&A query
  
  Root:
    - GET  /                -> API information
    - GET  /health          -> Overall health check
"""
import json
import os
import re
import sys
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Load .env file from the current directory
load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

# ============================================================================
# Configuration
# ============================================================================

# Zoho RAG Configuration
ZOHO_ARTIFACTS_ROOT = Path("RAGzoho/artifacts/clients")
ZOHO_ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)

# NGEN RAG Configuration
NGEN_ARTIFACTS_DIR = Path("RAGngen/artifacts")
NGEN_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Excel RAG Configuration
EXCEL_ARTIFACTS_DIR = Path("RAGexcel/artifacts")
EXCEL_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


# Gemini API Configuration for all RAGs
# IMPORTANT: Set GEMINI_API_KEY in environment variable or .env file
# Never hardcode API keys in source code!
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("[WARN] GEMINI_API_KEY not set in environment.")
    print("[INFO] Please set it in your .env file or environment variable.")
    print("[INFO] Get your API key from: https://makersuite.google.com/app/apikey")
    # Don't exit - let individual RAG modules handle the error
else:
    # Set it in environment so all imported modules can access it
    # This MUST be done before importing RAG modules
    os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
    print(f"[INFO] GEMINI_API_KEY loaded and set in environment (length: {len(GEMINI_API_KEY)})")

try:
    import faiss  # type: ignore
    FAISS_OK = True
except Exception:
    FAISS_OK = False

try:
    import google.generativeai as genai
    # Configure Gemini only if API key is available
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        print("[INFO] Google Gemini API configured successfully.")
    else:
        print("[WARN] GEMINI_API_KEY not set - Gemini features will not work.")
except Exception as exc:
    print(f"[WARN] Failed to import Google Generative AI SDK: {exc}")
    print("[INFO] Some RAG endpoints may not work without Gemini API.")


# ============================================================================
# Import NGEN RAG modules
# ============================================================================

# Add NGEN RAG directory to path
ngen_rag_path = Path("RAGngen")
if ngen_rag_path.exists():
    try:
        # Import using importlib to ensure we get the correct module (consistent with Excel RAG)
        import importlib.util
        ngen_rag_query_path = ngen_rag_path / "rag_query.py"
        if ngen_rag_query_path.exists():
            spec = importlib.util.spec_from_file_location("ngen_rag_query", ngen_rag_query_path)
            ngen_rag_query = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(ngen_rag_query)
            
            ngen_load_index = ngen_rag_query.load_index
            ngen_embed_query = ngen_rag_query.embed_query
            ngen_search = ngen_rag_query.search
            ngen_build_prompt = ngen_rag_query.build_prompt
            ngen_ask_llm = ngen_rag_query.ask_llm
            extract_aum_from_chunk = ngen_rag_query.extract_aum_from_chunk
            extract_combined_market_value = ngen_rag_query.extract_combined_market_value
            NGEN_RAG_AVAILABLE = True
        else:
            raise ImportError(f"rag_query.py not found in {ngen_rag_path}")
    except Exception as e:
        print(f"[WARN] NGEN RAG modules not available: {e}")
        print(f"[INFO] Import error details: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        print("[INFO] NGEN RAG endpoints will be disabled.")
        NGEN_RAG_AVAILABLE = False
else:
    print("[WARN] NGEN RAG directory not found. NGEN RAG endpoints will be disabled.")
    NGEN_RAG_AVAILABLE = False

# ============================================================================
# Import Excel RAG modules
# ============================================================================

# Add Excel RAG directory to path
excel_rag_path = Path("RAGexcel")
if excel_rag_path.exists():
    # Add the RAGexcel directory to path so we can import modules directly
    # Use absolute path and ensure it's at the front of sys.path
    excel_abs_path = str(excel_rag_path.absolute())
    if excel_abs_path not in sys.path:
        sys.path.insert(0, excel_abs_path)
    try:
        # Import using importlib to ensure we get the correct module
        import importlib.util
        excel_rag_query_path = excel_rag_path / "rag_query.py"
        if excel_rag_query_path.exists():
            spec = importlib.util.spec_from_file_location("excel_rag_query", excel_rag_query_path)
            excel_rag_query = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(excel_rag_query)
            
            excel_load_index = excel_rag_query.load_index
            excel_embed_query = excel_rag_query.embed_query
            excel_search = excel_rag_query.search
            excel_build_prompt = excel_rag_query.build_prompt
            excel_ask_llm = excel_rag_query.ask_llm
            EXCEL_RAG_AVAILABLE = True
        else:
            raise ImportError(f"rag_query.py not found in {excel_rag_path}")
    except Exception as e:
        print(f"[WARN] Excel RAG modules not available: {e}")
        print(f"[INFO] Import error details: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        print("[INFO] Excel RAG endpoints will be disabled.")
        EXCEL_RAG_AVAILABLE = False
else:
    print("[WARN] Excel RAG directory not found. Excel RAG endpoints will be disabled.")
    EXCEL_RAG_AVAILABLE = False

# ============================================================================
# Pydantic Models
# ============================================================================

class ZohoQueryRequest(BaseModel):
    query: str = Field(..., description="Query string that must contain client email. Example: 'give me the portfolio details about this client example@gmail.com'")
    top_k: int = Field(6, ge=1, le=20, description="Maximum number of chunks to retrieve")
 

class ZohoQueryResponse(BaseModel):
    answer: str
    citations: List[Dict[str, str]]
    retrieved: List[Dict[str, str]]


class ZohoRetrieveResponse(BaseModel):
    email: str
    question: str
    retrieved_chunks: List[Dict[str, Any]]
    total_chunks: int


class NgenQAIn(BaseModel):
    question: str


class NgenQAResponse(BaseModel):
    question: str
    answer: str


class NgenRetrieveResponse(BaseModel):
    question: str
    retrieved_chunks: List[Dict[str, Any]]
    total_chunks: int


class ExcelQAIn(BaseModel):
    question: str


class ExcelQAResponse(BaseModel):
    question: str
    answer: str


class ExcelRetrieveResponse(BaseModel):
    question: str
    retrieved_chunks: List[Dict[str, Any]]
    total_chunks: int

# ============================================================================
# Zoho RAG Functions
# ============================================================================

def zoho_require_client(email: str) -> Path:
    client_dir = ZOHO_ARTIFACTS_ROOT / email
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
def zoho_load_metadata(email: str):
    client_dir = zoho_require_client(email)
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


def zoho_embed_query(question: str) -> np.ndarray:
    try:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=question,
            task_type="retrieval_query"
        )
        vec = np.array(result['embedding'], dtype="float32")
        vec = vec / (np.linalg.norm(vec) + 1e-12)
        return vec.astype("float32")
    except Exception as e:
        error_msg = str(e)
        if "leaked" in error_msg.lower() or "403" in error_msg or "PermissionDenied" in error_msg:
            raise HTTPException(
                status_code=403,
                detail=(
                    "API Key Error: Your Google Gemini API key has been flagged as leaked and is blocked for embedding operations. "
                    "Even though it might work for chat, the embedding API has stricter security checks. "
                    "SOLUTION: Get a NEW API key from https://makersuite.google.com/app/apikey and update your .env file."
                )
            )
        raise HTTPException(status_code=500, detail=f"Embedding error: {error_msg}")


def zoho_retrieve_chunks(email: str, question: str, top_k: int):
    matrix, meta, index, chunk_map = zoho_load_metadata(email)
    query_vec = zoho_embed_query(question)
    if index is not None:
        # Check dimension mismatch
        query_dim = query_vec.shape[0]
        index_dim = index.d
        
        if query_dim != index_dim:
            # Dimension mismatch - fall back to numpy search instead of failing
            # This allows the system to work even with old indexes
            print(f"[WARN] FAISS dimension mismatch ({query_dim} vs {index_dim}). Using numpy search fallback.")
            print(f"[INFO] To fix: Rebuild indexes by running RAGzoho indexing process.")
            index = None  # Force numpy fallback
    
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


def zoho_build_prompt(email: str, question: str, hits: List[Dict[str, Dict[str, str]]]) -> str:
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


def zoho_call_llm(prompt: str) -> str:
    # Try multiple model names for compatibility
    model_names = [
        "gemini-1.5-flash-latest",
        "gemini-1.5-flash",
        "gemini-2.0-flash-exp",
        "gemini-pro"
    ]
    
    # Combine system and user message for Gemini
    full_prompt = "You are a precise financial research assistant.\n\n" + prompt
    
    generation_config = genai.types.GenerationConfig(
        temperature=0.0,
        max_output_tokens=500,
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
                content = response.text
                return content.encode("ascii", "ignore").decode("ascii")
        except Exception as e:
            last_error = e
            continue  # Try next model
    
    # If all models failed, return error with details
    error_msg = str(last_error) if last_error else "Unknown error"
    return f"Error: Failed to generate response with all available models. Last error: {error_msg}"


def extract_email_from_query(query: str) -> str:
    """Extract email address from query string."""
    # Email regex pattern
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    matches = re.findall(email_pattern, query)
    
    if not matches:
        raise HTTPException(
            status_code=400,
            detail="Email address not found in query. Please include a valid email address in your query. Example: 'give me the portfolio details about this client example@gmail.com'"
        )
    
    # Return the first email found
    email = matches[0].strip().lower()
    
    # Verify email exists in the system
    client_dir = ZOHO_ARTIFACTS_ROOT / email
    if not client_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Client '{email}' not found. Please check the email address or ensure the client data has been indexed."
        )
    
    return email


def zoho_query_service(email: str, question: str, top_k: int):
    hits = zoho_retrieve_chunks(email, question, top_k)
    prompt = zoho_build_prompt(email, question, hits)
    answer = zoho_call_llm(prompt)
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

# ============================================================================
# FastAPI App Setup
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load NGEN RAG index if available
    if NGEN_RAG_AVAILABLE:
        try:
            print("[INFO] Loading NGEN RAG index...")
            app.state.ngen_loader = ngen_load_index(NGEN_ARTIFACTS_DIR)
            print("[OK] NGEN RAG index loaded successfully.")
        except Exception as e:
            print(f"[WARN] Failed to load NGEN RAG index: {e}")
            print("[INFO] NGEN RAG endpoints will be disabled.")
            app.state.ngen_loader = None
    else:
        app.state.ngen_loader = None
    
    # Startup: Load Excel RAG index if available
    if EXCEL_RAG_AVAILABLE:
        try:
            print("[INFO] Loading Excel RAG index...")
            app.state.excel_loader = excel_load_index(EXCEL_ARTIFACTS_DIR)
            # Load chunks.jsonl for full text retrieval
            chunks_path = EXCEL_ARTIFACTS_DIR / "chunks.jsonl"
            app.state.excel_chunk_map = {}
            if chunks_path.exists():
                with chunks_path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        if line.strip():
                            record = json.loads(line)
                            app.state.excel_chunk_map[int(record["id"])] = record
            print("[OK] Excel RAG index loaded successfully.")
        except Exception as e:
            print(f"[WARN] Failed to load Excel RAG index: {e}")
            print("[INFO] Excel RAG endpoints will be disabled.")
            app.state.excel_loader = None
            app.state.excel_chunk_map = {}
    else:
        app.state.excel_loader = None
        app.state.excel_chunk_map = {}
    yield
    # Shutdown: cleanup if needed
    pass


app = FastAPI(
    title="Unified RAG API",
    description="Combined RAG API for Zoho CRM client data, NGEN Markets financial data, and Excel Q&A data.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Root Endpoints
# ============================================================================

@app.get("/")
def root():
    """API information and available endpoints."""
    endpoints = {
        "zoho": {
            "description": "Zoho CRM client data RAG",
            "endpoints": {
                "POST /zoho/query": "Query client knowledge bases (with LLM answer)",
                "POST /zoho/retrieve": "Retrieve raw chunks (no LLM generation)",
                "GET /zoho/clients": "List registered client emails",
            },
        },
        "ngen": {
            "description": "NGEN Markets financial data RAG",
            "available": NGEN_RAG_AVAILABLE and hasattr(app.state, "ngen_loader") and app.state.ngen_loader is not None,
            "endpoints": {
                "GET /ngen/health": "Health check",
                "POST /ngen/qa": "Question answering (with LLM answer)",
                "POST /ngen/retrieve": "Retrieve raw chunks (no LLM generation)",
            },
        },
        "excel": {
            "description": "Excel Q&A RAG",
            "available": EXCEL_RAG_AVAILABLE and hasattr(app.state, "excel_loader") and app.state.excel_loader is not None,
            "endpoints": {
                "GET /excel/health": "Health check",
                "POST /excel/qa": "Question answering (with LLM answer)",
                "POST /excel/retrieve": "Retrieve raw chunks (no LLM generation)",
            },
        },
    }
    return {
        "name": "Unified RAG API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": endpoints,
    }


@app.get("/health")
def health():
    """Overall health check."""
    status = {
        "status": "ok",
        "zoho_rag": "available",
        "ngen_rag": "available" if (NGEN_RAG_AVAILABLE and hasattr(app.state, "ngen_loader") and app.state.ngen_loader is not None) else "unavailable",
        "excel_rag": "available" if (EXCEL_RAG_AVAILABLE and hasattr(app.state, "excel_loader") and app.state.excel_loader is not None) else "unavailable",
    }
    return status

# ============================================================================
# Zoho RAG Endpoints
# ============================================================================

@app.post("/zoho/query", response_model=ZohoQueryResponse, summary="Ask a question about a specific client")
def zoho_query_endpoint(payload: ZohoQueryRequest):
    """Query Zoho CRM client knowledge base. Email must be included in the query string."""
    # Extract email from query
    email = extract_email_from_query(payload.query)
    
    # Use the full query as the question
    question = payload.query
    
    # Query the service
    answer, citations, retrieved = zoho_query_service(email, question, payload.top_k)
    return ZohoQueryResponse(answer=answer, citations=citations, retrieved=retrieved)


@app.get("/zoho/clients", summary="List registered client emails")
def zoho_list_clients():
    """List all registered client emails from Zoho CRM."""
    if not ZOHO_ARTIFACTS_ROOT.exists():
        return {"clients": []}
    clients = [p.name for p in ZOHO_ARTIFACTS_ROOT.iterdir() if p.is_dir()]
    return {"clients": sorted(clients)}


@app.post("/zoho/retrieve", response_model=ZohoRetrieveResponse, summary="Retrieve raw chunks for a client query (no LLM generation)")
def zoho_retrieve_endpoint(payload: ZohoQueryRequest):
    """Retrieve raw chunks from Zoho CRM knowledge base without generating an answer. Email must be included in the query string."""
    # Extract email from query
    email = extract_email_from_query(payload.query)
    
    # Use the full query as the question
    question = payload.query
    
    # Load metadata to get full chunk text
    matrix, meta, index, chunk_map = zoho_load_metadata(email)
    query_vec = zoho_embed_query(question)
    
    # Perform search
    if index is not None:
        # Check dimension mismatch
        query_dim = query_vec.shape[0]
        index_dim = index.d
        
        if query_dim != index_dim:
            # Dimension mismatch - fall back to numpy search
            print(f"[WARN] FAISS dimension mismatch ({query_dim} vs {index_dim}). Using numpy search fallback.")
            index = None  # Force numpy fallback
    
    if index is not None:
        scores, ids = index.search(query_vec.reshape(1, -1), payload.top_k)
        hits = []
        for score, idx in zip(scores[0], ids[0]):
            if 0 <= idx < len(meta):
                info = meta[int(idx)].copy()
                # Get full chunk text from chunk_map
                full_text = chunk_map.get(int(info["id"]), {}).get("text", "")
                info["text"] = full_text
                hits.append({"score": float(score), "meta": info})
    else:
        similarities = matrix @ query_vec
        top_indices = similarities.argsort()[-payload.top_k:][::-1]
        hits = []
        for idx in top_indices:
            info = meta[int(idx)].copy()
            # Get full chunk text from chunk_map
            full_text = chunk_map.get(int(info["id"]), {}).get("text", "")
            info["text"] = full_text
            hits.append({"score": float(similarities[int(idx)]), "meta": info})
    
    # Format response
    retrieved_chunks = []
    for hit in hits:
        meta = hit["meta"]
        chunk_text = meta.get("text", "")
        retrieved_chunks.append({
            "score": hit["score"],
            "source": Path(meta["source"]).name,
            "chunk_id": meta.get("chunk_id", ""),
            "text": chunk_text,  # Full text, not truncated
            "metadata": {
                "email": meta.get("email", email),
                "source_path": meta.get("source", ""),
                "id": meta.get("id", ""),
            }
        })
    
    return ZohoRetrieveResponse(
        email=email,
        question=question,
        retrieved_chunks=retrieved_chunks,
        total_chunks=len(retrieved_chunks)
    )

# ============================================================================
# NGEN RAG Endpoints
# ============================================================================

@app.get("/ngen/health")
def ngen_health():
    """NGEN RAG health check."""
    if not NGEN_RAG_AVAILABLE:
        raise HTTPException(status_code=503, detail="NGEN RAG is not available. Check if artifacts are properly set up.")
    if not hasattr(app.state, "ngen_loader") or app.state.ngen_loader is None:
        raise HTTPException(status_code=503, detail="NGEN RAG index not loaded.")
    return {"status": "ok", "service": "ngen_rag"}


@app.post("/ngen/qa", response_model=NgenQAResponse, summary="Question answering for NGEN Markets data")
def ngen_qa(body: NgenQAIn, request: Request):
    """Question Answering - returns question and answer using NGEN Markets artifacts."""
    if not NGEN_RAG_AVAILABLE:
        raise HTTPException(status_code=503, detail="NGEN RAG is not available. Check if artifacts are properly set up.")
    
    if not hasattr(request.app.state, "ngen_loader") or request.app.state.ngen_loader is None:
        raise HTTPException(status_code=503, detail="NGEN RAG index not loaded. Check server logs.")
    
    loader = request.app.state.ngen_loader
    qv = ngen_embed_query(body.question)
    
    question_lower = body.question.lower()
    
    # For different question types, retrieve appropriate number of chunks
    if "nfo" in question_lower or "new fund offer" in question_lower:
        top_n = 100
        final_k = 63
    elif ("aaa" in question_lower or "aa" in question_lower or "a rated" in question_lower) and ("bond" in question_lower or "debt" in question_lower):
        top_n = 100
        final_k = 30
    elif "top 25" in question_lower or "25 stocks" in question_lower:
        top_n = 50
        final_k = 30
    elif "amc" in question_lower:
        top_n = 100
        final_k = 30
    else:
        top_n = 50
        final_k = 10
    
    # Search many candidates to find the right data
    hits = ngen_search(loader, qv, top_n=top_n, question=body.question)
    
    # Use top chunks to ensure we get the relevant data
    retrieved = [{"meta": h["meta"], "chunk_text": h.get("chunk_text", "")} for h in hits[:final_k]]
    
    # Try direct extraction first for combined market value questions
    if ("top 25" in question_lower or "25 stocks" in question_lower) and ("combined" in question_lower or "total" in question_lower or "market value" in question_lower):
        combined_value = extract_combined_market_value(retrieved)
        if combined_value:
            return NgenQAResponse(
                question=body.question,
                answer=combined_value,
            )
    
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
                            return NgenQAResponse(
                                question=body.question,
                                answer=aum_value,
                            )
    
    # Build prompt with improved instructions
    prompt = ngen_build_prompt(body.question, retrieved)
    answer = ngen_ask_llm(prompt, temperature=0.0, max_tokens=200)
    
    return NgenQAResponse(
        question=body.question,
        answer=answer,
    )


@app.post("/ngen/retrieve", response_model=NgenRetrieveResponse, summary="Retrieve raw chunks for NGEN Markets query (no LLM generation)")
def ngen_retrieve(body: NgenQAIn, request: Request):
    """Retrieve raw chunks from NGEN Markets knowledge base without generating an answer."""
    if not NGEN_RAG_AVAILABLE:
        raise HTTPException(status_code=503, detail="NGEN RAG is not available. Check if artifacts are properly set up.")
    
    if not hasattr(request.app.state, "ngen_loader") or request.app.state.ngen_loader is None:
        raise HTTPException(status_code=503, detail="NGEN RAG index not loaded. Check server logs.")
    
    loader = request.app.state.ngen_loader
    qv = ngen_embed_query(body.question)
    
    question_lower = body.question.lower()
    
    # For different question types, retrieve appropriate number of chunks
    if "nfo" in question_lower or "new fund offer" in question_lower:
        top_n = 100
        final_k = 63
    elif ("aaa" in question_lower or "aa" in question_lower or "a rated" in question_lower) and ("bond" in question_lower or "debt" in question_lower):
        top_n = 100
        final_k = 30
    elif "top 25" in question_lower or "25 stocks" in question_lower:
        top_n = 50
        final_k = 30
    elif "amc" in question_lower:
        top_n = 100
        final_k = 30
    else:
        top_n = 50
        final_k = 10
    
    # Search many candidates to find the right data
    hits = ngen_search(loader, qv, top_n=top_n, question=body.question)
    
    # Format retrieved chunks with full details
    retrieved_chunks = []
    for hit in hits[:final_k]:
        meta = hit.get("meta", {})
        chunk_text = hit.get("chunk_text", "")
        retrieved_chunks.append({
            "score": hit.get("score", 0.0),
            "combined_score": hit.get("combined_score", 0.0) if "combined_score" in hit else None,
            "source": meta.get("source", ""),
            "chunk_id": meta.get("chunk_id", ""),
            "text": chunk_text,
            "text_preview": meta.get("text_preview", chunk_text[:200]) if meta else chunk_text[:200],
            "metadata": {
                "id": meta.get("id", ""),
                "source_path": meta.get("source", ""),
            }
        })
    
    return NgenRetrieveResponse(
        question=body.question,
        retrieved_chunks=retrieved_chunks,
        total_chunks=len(retrieved_chunks)
    )

# ============================================================================
# Excel RAG Endpoints
# ============================================================================

@app.get("/excel/health")
def excel_health():
    """Excel RAG health check."""
    if not EXCEL_RAG_AVAILABLE:
        raise HTTPException(status_code=503, detail="Excel RAG is not available. Check if artifacts are properly set up.")
    if not hasattr(app.state, "excel_loader") or app.state.excel_loader is None:
        raise HTTPException(status_code=503, detail="Excel RAG index not loaded.")
    return {"status": "ok", "service": "excel_rag"}


@app.post("/excel/qa", response_model=ExcelQAResponse, summary="Question answering for Excel Q&A data")
def excel_qa(body: ExcelQAIn, request: Request):
    """Question Answering - returns question and answer using Excel Q&A artifacts."""
    if not EXCEL_RAG_AVAILABLE:
        raise HTTPException(status_code=503, detail="Excel RAG is not available. Check if artifacts are properly set up.")
    
    if not hasattr(request.app.state, "excel_loader") or request.app.state.excel_loader is None:
        raise HTTPException(status_code=503, detail="Excel RAG index not loaded. Check server logs.")
    
    loader = request.app.state.excel_loader
    try:
        qv = excel_embed_query(body.question)
    except RuntimeError as e:
        error_msg = str(e)
        if "dimension mismatch" in error_msg.lower() or "embedding dimension" in error_msg.lower():
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Embedding dimension mismatch! The FAISS index was built with a different embedding model. "
                    f"Rebuild the index: cd RAGexcel && python rag_indexer.py --artifacts_dir artifacts"
                )
            )
        raise HTTPException(status_code=500, detail=error_msg)
    
    # Default retrieval parameters
    top_n = 50
    final_k = 6
    
    # Search for relevant chunks
    try:
        hits = excel_search(loader, qv, top_n=top_n)
    except RuntimeError as e:
        error_msg = str(e)
        if "dimension mismatch" in error_msg.lower() or "embedding dimension" in error_msg.lower():
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Embedding dimension mismatch! The FAISS index was built with a different embedding model. "
                    f"Rebuild the index: cd RAGexcel && python rag_indexer.py --artifacts_dir artifacts"
                )
            )
        raise HTTPException(status_code=500, detail=error_msg)
    
    # Use top chunks
    retrieved = [{"meta": h["meta"]} for h in hits[:final_k]]
    
    # Build prompt and get answer
    prompt = excel_build_prompt(body.question, retrieved)
    answer = excel_ask_llm(prompt, temperature=0.0, max_tokens=600)
    
    return ExcelQAResponse(
        question=body.question,
        answer=answer,
    )


@app.post("/excel/retrieve", response_model=ExcelRetrieveResponse, summary="Retrieve raw chunks for Excel Q&A query (no LLM generation)")
def excel_retrieve(body: ExcelQAIn, request: Request):
    """Retrieve raw chunks from Excel Q&A knowledge base without generating an answer."""
    if not EXCEL_RAG_AVAILABLE:
        raise HTTPException(status_code=503, detail="Excel RAG is not available. Check if artifacts are properly set up.")
    
    if not hasattr(request.app.state, "excel_loader") or request.app.state.excel_loader is None:
        raise HTTPException(status_code=503, detail="Excel RAG index not loaded. Check server logs.")
    
    loader = request.app.state.excel_loader
    try:
        qv = excel_embed_query(body.question)
    except RuntimeError as e:
        error_msg = str(e)
        if "dimension mismatch" in error_msg.lower() or "embedding dimension" in error_msg.lower():
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Embedding dimension mismatch! The FAISS index was built with a different embedding model. "
                    f"Rebuild the index: cd RAGexcel && python rag_indexer.py --artifacts_dir artifacts"
                )
            )
        raise HTTPException(status_code=500, detail=error_msg)
    
    # Default retrieval parameters
    top_n = 50
    final_k = 6
    
    # Search for relevant chunks
    try:
        hits = excel_search(loader, qv, top_n=top_n)
    except RuntimeError as e:
        error_msg = str(e)
        if "dimension mismatch" in error_msg.lower() or "embedding dimension" in error_msg.lower():
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Embedding dimension mismatch! The FAISS index was built with a different embedding model. "
                    f"Rebuild the index: cd RAGexcel && python rag_indexer.py --artifacts_dir artifacts"
                )
            )
        raise HTTPException(status_code=500, detail=error_msg)
    
    # Format retrieved chunks with full details and raw content
    retrieved_chunks = []
    chunk_map = getattr(request.app.state, "excel_chunk_map", {})
    
    for hit in hits[:final_k]:
        meta = hit.get("meta", {})
        chunk_id = meta.get("id")
        
        # Get full text from chunk_map if available, otherwise use text_preview
        if chunk_id is not None:
            chunk_id_int = int(chunk_id) if not isinstance(chunk_id, int) else chunk_id
            if chunk_id_int in chunk_map:
                chunk_text = chunk_map[chunk_id_int].get("text", "")
            else:
                chunk_text = meta.get("text_preview", "")
        else:
            chunk_text = meta.get("text_preview", "")
        
        # Build citation tag
        if meta.get("kind") == "pdf":
            citation = f"[{meta.get('source', '')}#page{meta.get('page', '')}#chunk{meta.get('chunk_index', '')}]"
        else:
            citation = f"[{meta.get('source', '')}#row{meta.get('row_index', '')}#chunk{meta.get('chunk_index', '')}]"
        
        retrieved_chunks.append({
            "score": hit.get("score", 0.0),
            "source": meta.get("source", ""),
            "chunk_id": str(chunk_id) if chunk_id is not None else "",
            "text": chunk_text,  # Full raw content
            "text_preview": meta.get("text_preview", chunk_text[:200]),
            "citation": citation,
            "metadata": {
                "id": chunk_id,
                "kind": meta.get("kind", ""),
                "source_path": meta.get("source", ""),
                "row_index": meta.get("row_index"),
                "page": meta.get("page"),
                "chunk_index": meta.get("chunk_index"),
            }
        })
    
    return ExcelRetrieveResponse(
        question=body.question,
        retrieved_chunks=retrieved_chunks,
        total_chunks=len(retrieved_chunks)
    )

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8005, reload=False)


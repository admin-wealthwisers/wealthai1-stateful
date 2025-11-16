## Unified RAG Backend – Detailed Walkthrough

### Overview
- **Components**: `RAGexcels`, `RAGngen`, `RAGzoho` exposed via a unified FastAPI server (`server.py`).
- **Shared stack**: Tokenizer-aware semantic chunking with overlap and low‑content merging; embeddings with `text-embedding-3-small`; FAISS indices; prompt-building for concise, context-grounded answers.
- **Chat model**: `gpt-4o-mini` (Excels via CLI; NGEN and Zoho in API).


## RAGexcels

### Purpose
- Index tabular (Excel/CSV) and PDF content; retrieve best matches; build compact prompts for LLM answers.

### Files and roles
- `RAGexcels/rag_chunker.py`: Reads Excel/CSV/PDF; normalizes text; tokenizer-aware semantic chunking with sentence/pipe/table awareness; 100-token overlap; merges low-content slices; writes `chunks.jsonl` + `chunks_meta.json`.
- `RAGexcels/rag_indexer.py`: Embeds chunks with `text-embedding-3-small`, L2-normalizes, builds FAISS HNSW, saves `embeddings.npy` and `faiss.index`.
- `RAGexcels/rag_query.py`: Loads artifacts, embeds queries, retrieves from FAISS, builds prompts, and (CLI) can call the chat model.
- `RAGexcels/server_retrieval.py`, `RAGexcels/rag_main.py`: Thin helpers for serving/running the pipeline.
- Unified endpoints are in root `server.py` under `/excels`.

### Chunking strategy (semantic + overlap + low-content merge)
- Token-aware via `tiktoken`; paragraph/sentence/table-cell units to avoid mid-sentence/cell cuts.
- Targets ~600 tokens (max 900, min 180) with ~100-token overlap.
- Merges sub-minimum slices into the previous chunk to keep embeddings dense and context-preserving.

Code reference:

```120:206:RAGexcels/rag_chunker.py
def _split_units(text: str) -> List[str]:
    text = _normalize(text)
    ...
    sentences = _SENT_SPLIT.split(para)
    ...
    if "|" in sentence:
        cells = [cell.strip() for cell in _PIPE_SPLIT.split(sentence) if cell.strip()]
        ...
    return units
```

```146:206:RAGexcels/rag_chunker.py
while start_idx < total:
    ...
    if token_sum < MIN_TOKENS and idx < total:
        extra = units[idx]
        ...
    chunk_text = " ".join(u["text"] for u in chunk_units).strip()
    if chunk_text:
        chunk_record = {"text": chunk_text, "tokens": token_sum}
        if chunk_records and chunk_record["tokens"] < MIN_TOKENS:
            chunk_records[-1]["text"] = f"{chunk_records[-1]['text']} {chunk_record['text']}".strip()
            chunk_records[-1]["tokens"] += chunk_record["tokens"]
        else:
            chunk_records.append(chunk_record)
...
return [c["text"] for c in chunk_records]
```

### Indexing and search
- Embedding in batches; L2 normalization; FAISS HNSW with efConstruction=200.

```36:61:RAGexcels/rag_indexer.py
emb = embed_batches(texts)
...
faiss.normalize_L2(emb)
idx = faiss.IndexHNSWFlat(dim, 32)
idx.hnsw.efConstruction = 200
idx.add(emb)
faiss.write_index(idx, str(faiss_path))
```

### Query path
- Load index/meta → embed query → FAISS search → build concise, context-only prompt or return contexts directly via API.

```19:43:RAGexcels/rag_query.py
def load_index(artifacts_dir: Path):
    meta = json.loads((artifacts_dir / "chunks_meta.json").read_text(...))
    idx = faiss.read_index(str(faiss_path))
    return {"index": idx, "meta": meta}

def search(loader, qvec: np.ndarray, top_n: int):
    q = np.array([qvec], dtype=np.float32)
    faiss.normalize_L2(q)
    D, I = loader["index"].search(q, top_n)
    ...
```

```45:67:RAGexcels/rag_query.py
def build_prompt(question: str, retrieved: List[Dict[str,Any]], max_chars: int = 3800):
    ...
    return f"""Answer the question directly and concisely based on the context provided...
```

### Server endpoints (Excels)
```133:163:server.py
@app.post("/excels/retrieve"...)
@app.post("/excels/retrieve_prompt"...)
...
prompt = (
   "Answer the question directly based only on the context. ..."
)
```

### End-to-end flow
- Ingest → chunk (`rag_chunker.py`) → index (`rag_indexer.py`) → query (`rag_query.py`) → build prompt → answer (CLI) or return contexts/prompt (API).


## RAGngen

### Purpose
- Financial-domain RAG optimized for questions on categories, NFOs, AMC performance, bond ratings, top stocks, etc.

### Files and roles
- `RAGngen/rag_chunker.py`: Same improved chunking strategy as Excels.
- `RAGngen/rag_indexer.py`: Same embedding/index pipeline.
- `RAGngen/rag_query.py`: Advanced retrieval: combines FAISS distance with domain-aware boosts; loads full chunk text cache; builds task-specific prompts including “CRITICAL INSTRUCTIONS” and direct extraction paths where applicable.

### Domain-aware re-ranking
- Search retrieves broader candidate set; combines similarity (converted from FAISS distance) with category/domain score (85% weight domain, 15% similarity) for robust, intention-aligned retrieval.

```145:187:RAGngen/rag_query.py
def search(loader, qvec: np.ndarray, top_n: int, question: str = ""):
    ...
    search_k = min(top_n * 5, len(meta))
    D, I = loader["index"].search(q, search_k)
    ...
    category_score = _filter_by_category(question, chunk_text, meta_item)
    similarity = 1.0 / (1.0 + score)
    combined_score = similarity * 0.15 + category_score * 0.85
    ...
    scored_results.sort(key=lambda x: x["combined_score"], reverse=True)
```

### Domain filters
- Rules for category-average vs individual funds; NFO prioritization; debt fund ratings; AMC/equity; top-25 stocks.

```61:143:RAGngen/rag_query.py
def _filter_by_category(question: str, chunk_text: str, meta: Dict = None) -> float:
    ...
    return 0.5
```

### Prompt specialization and direct extraction
- If the question matches special cases (e.g., AUM for Large Cap category), extracts directly from chunks; otherwise assembles a domain-guided prompt.

```238:369:RAGngen/rag_query.py
def build_prompt(question: str, retrieved: List[Dict[str,Any]], max_chars: int = 5000):
    # Try direct AUM extraction for Large Cap Fund
    ...
    # Otherwise build domain-instructed prompt
    ... # CRITICAL INSTRUCTIONS tailored to question intent
```

### Server endpoint (NGEN)
```166:175:server.py
@app.post("/ngen/qa"...)
qv = ngen_module.embed_query(body.question)
hits = ngen_module.search(...)
prompt = ngen_module.build_prompt(...)
answer = ngen_module.ask_llm(prompt, temperature=0.0, max_tokens=300)
```

### End-to-end flow
- Chunk/index as before → embed query → broaden candidates → domain re-ranking → specialized prompt (or direct extraction) → final answer.


## RAGzoho

### Purpose
- Client-specific RAG over Zoho-sourced documents; per-client artifacts; strict citations.

### Files and roles
- `RAGzoho/rag.py`: Complete pipeline — OAuth automation, data fetching, file readers (pdf/excel/csv/txt), tokenizer-aware chunking with overlap and low-content merging (`iter_chunks`), embeddings, FAISS (FlatIP) index, schedulable runner.
- `RAGzoho/server.py`: FastAPI — selects client directory, retrieves top chunks, builds citation-enforced prompt, calls OpenAI, returns answer + citations + retrieved previews.

### Improved chunker in `rag.py`
```1061:1119:RAGzoho/rag.py
def iter_chunks(text: str) -> Iterable[str]:
    ...
    if token_sum < MIN_TOKENS and idx < total:
        ...
    chunk_text = " ".join(u["text"] for u in chunk_units).strip()
    if chunk_text:
        chunk_record = {"text": chunk_text, "tokens": token_sum}
        if chunk_records and chunk_record["tokens"] < MIN_TOKENS:
            chunk_records[-1]["text"] = f"{chunk_records[-1]['text']} {chunk_record['text']}".strip()
            chunk_records[-1]["tokens"] += chunk_record["tokens"]
        else:
            chunk_records.append(chunk_record)
...
for record in chunk_records:
    yield record["text"]
```

### Retrieval and prompt
```109:128:RAGzoho/server.py
def retrieve_chunks(email: str, question: str, top_k: int):
    ...
    info["text"] = chunk_map.get(int(info["id"]), {}).get("text", "")[:1200]

def build_prompt(email: str, question: str, hits: List[...]) -> str:
    return (
        f"You are a precise assistant. Answer ONLY using the provided CONTEXT for client {email}. "
        'If the answer is missing, reply: "I don\'t know." Include citations like [file#chunkN].\n\n'
```

### Endpoints (Zoho)
```186:189:RAGzoho/server.py
@app.post("/query", response_model=QueryResponse, summary="Ask a question about a specific client")
def query_endpoint(...):
    answer, citations, retrieved = query_service(...)
```

### End-to-end flow
- OAuth/fetch → chunk per client (`iter_chunks`) → embed/index → query selects client → retrieve → citation-enforcing prompt → answer + citations + retrieved previews.


## Unified Server

### Root FastAPI (`server.py`)
- Loads `RAGexcels` and `RAGngen` indices at startup; routes Zoho queries to its own module. Provides `/health`, `/excels/retrieve`, `/excels/retrieve_prompt`, `/ngen/qa`, `/zoho/query`, `/zoho/clients`.

```119:126:server.py
@app.on_event("startup")
def _startup_loaders():
    ...
    _excels_loader = excels_module.load_index(EXCELS_ARTIFACTS)
    _ngen_loader = ngen_module.load_index(NGEN_ARTIFACTS)
```


## Models and configuration
- **Embeddings**: `text-embedding-3-small` for all three RAGs.
- **Chat**: `gpt-4o-mini`
  - Excels: Used by CLI `ask_llm`; API endpoints generally return contexts/prompt.
  - NGEN: API builds prompt and calls chat with temperature 0.0.
  - Zoho: API builds citation-enforced prompt and calls chat with temperature 0.0.


## Practical usage (high level)
- Build artifacts per RAG:
  - Chunk: run the chunker (`rag_chunker.py` or `rag.py` for Zoho).
  - Index: run the indexer (`rag_indexer.py` or `rag.py` indexing step for Zoho).
- Serve:
  - Unified server: `uvicorn server:app --host 0.0.0.0 --port 8005`.
  - Zoho server (standalone): `python RAGzoho/server.py` (optional).
- Query:
  - Excels: `/excels/retrieve`, `/excels/retrieve_prompt`
  - NGEN: `/ngen/qa`
  - Zoho: `/zoho/query` and `/zoho/clients`


## Notes on strategy improvements
- Tokenizer-aware chunking across all RAGs ensures uniform embedding lengths and reduces mid-sentence cuts.
- Overlap (~100 tokens) preserves boundary context for better hit recall.
- Low-content merge raises semantic density per chunk, improving similarity and ranking quality.
- NGEN’s domain-aware re-ranking (85% domain signal, 15% distance-derived similarity) aligns retrieval with financial question intent.



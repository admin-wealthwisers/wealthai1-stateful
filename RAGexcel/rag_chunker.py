#!/usr/bin/env python3
# rag_chunker.py
import argparse, json
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd

# Token-aware chunking (preferred)
try:
    import tiktoken
except Exception:
    tiktoken = None

# PDF reading
try:
    import pdfplumber
except Exception:
    pdfplumber = None
from PyPDF2 import PdfReader as _PdfReader

# Chunk sizes
CHUNK_TOKENS = 800
OVERLAP_TOKENS = 200
CHUNK_WORDS_FALLBACK = 500
OVERLAP_WORDS_FALLBACK = 120

# ------------ Excel/CSV helpers ------------
def smart_read_table(path: Path, skiprows: int = 0) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf in (".xls", ".xlsx"):
        return pd.read_excel(path)
    for enc in ("utf-8", "utf-8-sig", "latin1", "cp1252"):
        try:
            df = pd.read_csv(path, encoding=enc, engine="python", on_bad_lines="warn", skiprows=skiprows)
            if df.shape[1] > 1:
                return df
        except Exception:
            pass
    # fallback: single-column
    text = path.read_text(encoding="latin1", errors="replace")
    return pd.DataFrame({0: text.splitlines()})

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().replace("\n"," ").replace("\r"," ") for c in df.columns]
    return df

def row_to_text(row: pd.Series) -> str:
    parts = []
    for k, v in row.items():
        if pd.isna(v): 
            continue
        parts.append(f"{k}: {v}")
    return " | ".join(parts)

# ------------ PDF helpers ------------
def pdf_extract_text_plumber(path: Path) -> List[str]:
    # returns list of per-page strings
    pages = []
    with pdfplumber.open(path) as pdf:
        for p in pdf.pages:
            txt = p.extract_text() or ""
            pages.append(txt)
    return pages

def pdf_extract_text_pypdf(path: Path) -> List[str]:
    pages = []
    reader = _PdfReader(str(path))
    for p in reader.pages:
        txt = p.extract_text() or ""
        pages.append(txt)
    return pages

def read_pdf_pages(path: Path) -> List[str]:
    if pdfplumber is not None:
        try:
            return pdf_extract_text_plumber(path)
        except Exception:
            pass
    return pdf_extract_text_pypdf(path)

# ------------ chunking ------------
def get_encoder():
    if not tiktoken: return None
    try:
        return tiktoken.encoding_for_model("gpt-4o-mini")
    except Exception:
        try: return tiktoken.get_encoding("cl100k_base")
        except Exception: return None

def chunk_tokenwise(text: str, enc) -> List[str]:
    toks = enc.encode(text)
    if not toks: return []
    out, start, L = [], 0, len(toks)
    while start < L:
        end = min(L, start + CHUNK_TOKENS)
        out.append(enc.decode(toks[start:end]))
        start += CHUNK_TOKENS - OVERLAP_TOKENS
    return [c for c in out if len(c.strip()) > 20]

def chunk_wordwise(text: str) -> List[str]:
    words = text.split()
    if not words: return []
    out, start, L = [], 0, len(words)
    while start < L:
        end = min(L, start + CHUNK_WORDS_FALLBACK)
        out.append(" ".join(words[start:end]))
        start += CHUNK_WORDS_FALLBACK - OVERLAP_WORDS_FALLBACK
    return [c for c in out if len(c.strip()) > 20]

# ------------ public API ------------
def ingest_and_chunk(
    excel_files: List[str],
    pdf_files: List[str],
    artifacts_dir: str = "artifacts",
    skiprows: int = 0
) -> Dict[str, Any]:
    artifacts = Path(artifacts_dir); artifacts.mkdir(parents=True, exist_ok=True)
    chunks_path = artifacts / "chunks.jsonl"
    meta_path   = artifacts / "chunks_meta.json"

    enc = get_encoder()
    gid = 0
    meta = []

    with chunks_path.open("w", encoding="utf-8") as out:
        # Excel/CSV rows -> chunks
        for f in excel_files:
            p = Path(f)
            df = smart_read_table(p, skiprows=skiprows)
            df = normalize_columns(df)
            print(f"[chunk] EXCEL {p.name} shape={df.shape}")
            for rid, row in df.iterrows():
                text = row_to_text(row)
                if not text.strip(): 
                    continue
                parts = chunk_tokenwise(text, enc) if enc else chunk_wordwise(text)
                if not parts:
                    parts = [text]
                for cid, c in enumerate(parts):
                    rec = {
                        "id": gid,
                        "kind": "excel",
                        "source": p.name,
                        "row_index": int(rid),
                        "chunk_index": cid,
                        "text": c
                    }
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    meta.append({
                        "id": gid,
                        "kind": "excel",
                        "source": p.name,
                        "row_index": int(rid),
                        "chunk_index": cid,
                        "text_preview": c[:1000]
                    })
                    gid += 1

        # PDF pages -> chunks
        for f in pdf_files:
            p = Path(f)
            pages = read_pdf_pages(p)
            print(f"[chunk] PDF   {p.name} pages={len(pages)}")
            for pid, page_text in enumerate(pages, start=1):
                if not (page_text or "").strip():
                    continue
                parts = chunk_tokenwise(page_text, enc) if enc else chunk_wordwise(page_text)
                if not parts:
                    parts = [page_text]
                for cid, c in enumerate(parts):
                    rec = {
                        "id": gid,
                        "kind": "pdf",
                        "source": p.name,
                        "page": pid,
                        "chunk_index": cid,
                        "text": c
                    }
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    meta.append({
                        "id": gid,
                        "kind": "pdf",
                        "source": p.name,
                        "page": pid,
                        "chunk_index": cid,
                        "text_preview": c[:1000]
                    })
                    gid += 1

    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)

    print(f"[chunk] wrote {gid} chunks -> {chunks_path}")
    print(f"[chunk] meta -> {meta_path}")
    return {"chunks": gid, "chunks_path": str(chunks_path), "meta_path": str(meta_path)}

# CLI (optional standalone)
def _cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excels", nargs="*", default=[])
    ap.add_argument("--pdfs", nargs="*", default=[])
    ap.add_argument("--artifacts_dir", default="artifacts")
    ap.add_argument("--skiprows", type=int, default=0)
    args = ap.parse_args()
    ingest_and_chunk(args.excels, args.pdfs, args.artifacts_dir, args.skiprows)

if __name__ == "__main__":
    _cli()

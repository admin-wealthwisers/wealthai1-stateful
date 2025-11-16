#!/usr/bin/env python3
# rag_chunker.py
import argparse, json, re
from pathlib import Path
from typing import List, Dict, Any

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

# Chunking hyper-parameters
TARGET_TOKENS = 600
MAX_TOKENS = 900
MIN_TOKENS = 180
OVERLAP_TOKENS = 100

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_PIPE_SPLIT = re.compile(r"\s*\|\s*")
_MULTISPACE = re.compile(r"\s+")
_PARA_SPLIT = re.compile(r"\n{2,}")

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
    if not tiktoken:
        return None
    try:
        return tiktoken.encoding_for_model("gpt-4o-mini")
    except Exception:
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception:
            return None


def _token_len(enc, text: str) -> int:
    if not text:
        return 0
    if enc:
        try:
            return len(enc.encode(text))
        except Exception:
            pass
    # lightweight fallback: whitespace tokens
    return max(1, len(text.split()))


def _normalize(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r", " ").strip()
    return _MULTISPACE.sub(" ", text)


def _split_units(text: str) -> List[str]:
    text = _normalize(text)
    if not text:
        return []
    units: List[str] = []
    paragraphs = _PARA_SPLIT.split(text)
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        sentences = _SENT_SPLIT.split(para)
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if "|" in sentence:
                cells = [cell.strip() for cell in _PIPE_SPLIT.split(sentence) if cell.strip()]
                if cells:
                    units.extend(cells)
                    continue
            units.append(sentence)
    if not units:
        return [text]
    return units


def chunk_semantic(text: str, enc) -> List[str]:
    units = [{"text": u, "tokens": _token_len(enc, u)} for u in _split_units(text)]
    units = [u for u in units if u["tokens"] > 0]
    if not units:
        return []

    chunk_records: List[Dict[str, Any]] = []
    total = len(units)
    start_idx = 0

    while start_idx < total:
        idx = start_idx
        chunk_units: List[Dict[str, Any]] = []
        token_sum = 0

        while idx < total:
            unit = units[idx]
            if chunk_units and token_sum + unit["tokens"] > MAX_TOKENS:
                break
            chunk_units.append(unit)
            token_sum += unit["tokens"]
            idx += 1
            if token_sum >= TARGET_TOKENS:
                break

        if not chunk_units:
            unit = units[idx]
            chunk_units.append(unit)
            token_sum = unit["tokens"]
            idx += 1

        if token_sum < MIN_TOKENS and idx < total:
            extra = units[idx]
            chunk_units.append(extra)
            token_sum += extra["tokens"]
            idx += 1

        chunk_text = " ".join(u["text"] for u in chunk_units).strip()
        if chunk_text:
            chunk_record = {"text": chunk_text, "tokens": token_sum}
            if chunk_records and chunk_record["tokens"] < MIN_TOKENS:
                chunk_records[-1]["text"] = f"{chunk_records[-1]['text']} {chunk_record['text']}".strip()
                chunk_records[-1]["tokens"] += chunk_record["tokens"]
            else:
                chunk_records.append(chunk_record)

        if idx >= total:
            break

        overlap_tokens = 0
        overlap_units = 0
        for unit in reversed(chunk_units):
            overlap_units += 1
            overlap_tokens += unit["tokens"]
            if overlap_tokens >= OVERLAP_TOKENS:
                break

        if overlap_units >= len(chunk_units):
            overlap_units = max(0, len(chunk_units) - 1)

        next_start = idx - overlap_units if overlap_units else idx
        if next_start <= start_idx:
            next_start = idx
        start_idx = next_start

    return [c["text"] for c in chunk_records]

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
                parts = chunk_semantic(text, enc) or [text.strip()]
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
                parts = chunk_semantic(page_text, enc) or [page_text.strip()]
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

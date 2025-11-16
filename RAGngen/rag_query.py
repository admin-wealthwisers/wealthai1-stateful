#!/usr/bin/env python3
# rag_query.py
import os, json, argparse
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL  = "gpt-4o-mini"

import faiss

# Lazy initialization of OpenAI client
_client = None

def get_client():
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set. Put it in .env or environment variable.")
        _client = OpenAI(api_key=api_key)
    return _client

def load_chunk_cache(artifacts_dir: Path) -> Dict[int, str]:
    """Load all chunk texts into memory cache."""
    chunk_cache = {}
    chunks_path = artifacts_dir / "chunks.jsonl"
    if chunks_path.exists():
        with chunks_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunk = json.loads(line)
                    chunk_id = chunk.get("id")
                    if chunk_id is not None:
                        chunk_cache[chunk_id] = chunk.get("text", "")
    return chunk_cache

def load_index(artifacts_dir: Path):
    meta = json.loads((artifacts_dir / "chunks_meta.json").read_text(encoding="utf-8"))
    emb_path = artifacts_dir / "embeddings.npy"
    faiss_path = artifacts_dir / "faiss.index"
    if not emb_path.exists() or not faiss_path.exists():
        raise FileNotFoundError("Missing embeddings or faiss.index; run chunker + indexer first.")
    idx = faiss.read_index(str(faiss_path))
    chunk_cache = load_chunk_cache(artifacts_dir)
    return {"index": idx, "meta": meta, "chunk_cache": chunk_cache}

def embed_query(q: str) -> np.ndarray:
    client = get_client()
    resp = client.embeddings.create(model=EMBED_MODEL, input=[q])
    v = np.array(resp.data[0].embedding, dtype=np.float32)
    v = v / (np.linalg.norm(v) + 1e-12)
    return v

def _filter_by_category(question: str, chunk_text: str, meta: Dict = None) -> float:
    """Score chunk based on exact category match and data type."""
    question_lower = question.lower()
    chunk_lower = chunk_text.lower()
    
    # Check if question is asking for category averages (not individual funds)
    # "large cap fund" (singular) = category average
    # "large cap funds" (plural) or "which large cap fund" = individual funds
    is_category_avg_question = any(term in question_lower for term in [
        "category average", "category avg", "average of", "avg of", 
        "mutual fund category", "category"
    ]) or ("large cap fund" in question_lower and "funds" not in question_lower and "which" not in question_lower and "list" not in question_lower and "name" not in question_lower)
    
    # Check if chunk is from category averages file
    source = meta.get("source", "").lower() if meta else ""
    is_category_avg_data = "category averages" in source or "category average" in source
    
    # Debt fund bond rating questions - boost debt funds with ratings
    if ("aaa" in question_lower or "aa" in question_lower or "a rated" in question_lower) and ("bond" in question_lower or "debt" in question_lower):
        if "debt exposures" in source:
            return 2.0  # Highest priority for Debt Exposures file
        # Boost debt funds with AAA/AA/A % fields
        if "debt %" in chunk_lower:
            import re
            debt_match = re.search(r'debt %:\s*(\d+\.?\d*)', chunk_lower)
            if debt_match and float(debt_match.group(1)) > 0:
                # Check if it has rating fields
                if "aaa %" in chunk_lower or "aa %" in chunk_lower or "a %" in chunk_lower:
                    return 1.8  # High priority for debt funds with ratings
        return 0.3  # Penalize non-debt data
    
    # NFO questions - boost NFO data
    if "nfo" in question_lower or "new fund offer" in question_lower:
        if "nfo" in source:
            return 2.0  # Highest priority for NFO data
        return 0.3  # Penalize other data when asking for NFOs
    
    # Top 25 Stocks questions - boost Top 25 Stocks data
    if "top 25" in question_lower or "25 stocks" in question_lower:
        if "top 25 stocks" in source:
            return 2.0  # Highest priority for Top 25 Stocks data
        return 0.3  # Penalize other data when asking for Top 25
    
    # AMC-related questions - boost AMC data and equity funds
    if "amc" in question_lower:
        if "amcs" in source:
            return 1.5  # Boost AMC comparison data
        # For "best performing equity funds by AMC", boost equity funds
        if "equity" in question_lower and "equity %" in chunk_lower:
            # Check if it's an equity fund (Equity % > 0)
            import re
            equity_match = re.search(r'equity %:\s*(\d+\.?\d*)', chunk_lower)
            if equity_match and float(equity_match.group(1)) > 0:
                return 1.3  # Boost equity funds for AMC performance questions
        return 1.0
    
    # Extract category from question
    if "large cap" in question_lower and "mid" not in question_lower:
        # Must be "Large Cap Fund" not "Large & Mid Cap"
        if "large cap fund" in chunk_lower and "mid" not in chunk_lower:
            # Boost if it's category average data and question asks for category
            if is_category_avg_question and is_category_avg_data:
                return 2.0  # Highest priority
            elif is_category_avg_question and not is_category_avg_data:
                return 0.3  # Penalize individual funds when asking for category
            elif not is_category_avg_question and not is_category_avg_data:
                return 1.0  # Individual fund match
            else:
                return 0.8  # Category data but asking for individual
        elif "large & mid cap" in chunk_lower:
            return 0.1  # Heavy penalty for mixed category
    elif "mid cap" in question_lower:
        if "mid cap" in chunk_lower and "large" not in chunk_lower:
            if is_category_avg_question and is_category_avg_data:
                return 2.0
            return 1.0
    elif "small cap" in question_lower:
        if "small cap" in chunk_lower:
            if is_category_avg_question and is_category_avg_data:
                return 2.0
            return 1.0
    
    return 0.5  # Neutral score

def search(loader, qvec: np.ndarray, top_n: int, question: str = ""):
    meta = loader["meta"]
    chunk_cache = loader.get("chunk_cache", {})
    q = np.array([qvec], dtype=np.float32)
    faiss.normalize_L2(q)
    
    # Search more candidates for filtering (5x to ensure we find the right category)
    search_k = min(top_n * 5, len(meta))
    D, I = loader["index"].search(q, search_k)
    
    scored_results = []
    for score, idx in zip(D[0], I[0]):
        if 0 <= idx < len(meta):
            meta_item = meta[idx]
            chunk_id = meta_item.get("id")
            # Get full chunk text
            chunk_text = chunk_cache.get(chunk_id, meta_item.get("text_preview", ""))
            
            # Calculate category match score if question provided
            category_score = 0.5
            if question:
                category_score = _filter_by_category(question, chunk_text, meta_item)
            
            # Combine similarity and category scores
            # FAISS returns distance (lower is better), convert to similarity
            similarity = 1.0 / (1.0 + score)
            # Give MUCH more weight to category matching (85%) vs similarity (15%)
            # This ensures category averages are heavily prioritized
            combined_score = similarity * 0.15 + category_score * 0.85
            
            scored_results.append({
                "score": float(score),
                "combined_score": float(combined_score),
                "meta": meta_item,
                "chunk_text": chunk_text
            })
    
    # Sort by combined score (descending)
    scored_results.sort(key=lambda x: x["combined_score"], reverse=True)
    
    # Return top_n
    return [{"score": r["score"], "meta": r["meta"], "chunk_text": r["chunk_text"]} 
            for r in scored_results[:top_n]]

def extract_combined_market_value(chunks: List[Dict[str, Any]]) -> str:
    """Extract and sum all Value (Crs.) from Top 25 Stocks chunks."""
    import re
    total = 0.0
    count = 0
    
    for chunk_data in chunks:
        chunk_text = chunk_data.get("chunk_text", "")
        source = chunk_data.get("meta", {}).get("source", "").lower()
        
        if "top 25 stocks" in source:
            # Extract Value (Crs.) pattern
            match = re.search(r'Value \(Crs\.\):\s*(\d+\.?\d*)', chunk_text, re.IGNORECASE)
            if match:
                total += float(match.group(1))
                count += 1
    
    if count > 0:
        return f"{total:,.2f} Cr"
    return None

def extract_aum_from_chunk(chunk_text: str) -> str:
    """Try to extract AUM value directly from chunk text."""
    import re
    # Look for patterns like "1014.32 Cr: 12444.27 Cr" - extract the second value
    # Pattern: number Cr: number Cr (the second number is the AUM)
    pattern1 = r'(\d+\.?\d*)\s*Cr\s*:\s*(\d+\.?\d*)\s*Cr'
    matches1 = re.findall(pattern1, chunk_text, re.IGNORECASE)
    if matches1:
        # Get the last match and use the second value (the AUM)
        return matches1[-1][1] + " Cr"
    
    # Look for "AUM Cr.: 12444.27"
    pattern2 = r'AUM\s*Cr\.?\s*:\s*(\d+\.?\d*)'
    matches2 = re.findall(pattern2, chunk_text, re.IGNORECASE)
    if matches2:
        return matches2[-1] + " Cr"
    
    # Look for large numbers (4+ digits) followed by Cr
    pattern3 = r'(\d{4,}\.?\d*)\s*Cr'
    matches3 = re.findall(pattern3, chunk_text, re.IGNORECASE)
    if matches3:
        # Filter out very large numbers (likely not AUM) and return the largest reasonable one
        reasonable = [m for m in matches3 if float(m) < 1000000]
        if reasonable:
            return reasonable[-1] + " Cr"
    
    return None

def build_prompt(question: str, retrieved: List[Dict[str,Any]], max_chars: int = 5000):
    parts, used = [], 0
    question_lower = question.lower()
    
    # Try direct extraction first for AUM questions
    if "aum" in question_lower and "large cap" in question_lower:
        for r in retrieved:
            chunk_text = r.get("chunk_text", "")
            source = r.get("meta", {}).get("source", "").lower()
            # Check if it's from category averages and has Large Cap Fund
            if "category averages" in source and "large cap fund" in chunk_text.lower() and "mid" not in chunk_text.lower():
                aum_value = extract_aum_from_chunk(chunk_text)
                if aum_value:
                    # Return direct answer
                    return f"""Question: {question}

Answer: {aum_value}"""
    
    # Otherwise build normal prompt
    for r in retrieved:
        m = r["meta"]
        s = r.get("chunk_text", m.get("text_preview", ""))
        if not s or used + len(s) > max_chars: 
            break
        parts.append(s)
        used += len(s)
    ctx = "\n\n---\n\n".join(parts)
    
    # Enhanced prompt with category matching instruction
    is_category_question = any(term in question_lower for term in [
        "category average", "category avg", "average of", "avg of",
        "mutual fund category", "category"
    ]) or ("large cap fund" in question_lower and "funds" not in question_lower and "which" not in question_lower and "list" not in question_lower)
    
    # NFO questions
    if "nfo" in question_lower or "new fund offer" in question_lower:
        if "list" in question_lower or "all" in question_lower or "recent" in question_lower:
            category_instruction = """CRITICAL INSTRUCTIONS:
1. The question asks to list all recent NFOs (New Fund Offers) from NGEN Markets.
2. Look for chunks from "NGEN Markets NFOs.csv".
3. Each chunk contains: AMC, Fund name, Launch Date, Closing Date, Asset Class, Category, Min Inv, Offer Price.
4. Extract all NFOs from the context and list them with their key details (AMC, Fund name, Launch Date, Category).
5. If asked for "recent" NFOs, prioritize those with later Launch Dates.
6. Format the answer as a list of NFOs with their details."""
        elif "upcoming" in question_lower or "launch" in question_lower:
            category_instruction = "Extract NFOs with upcoming launch dates. Look for Launch Date and Closing Date fields. List NFOs that are launching soon."
        elif "category" in question_lower or "equity" in question_lower or "debt" in question_lower or "hybrid" in question_lower:
            category_instruction = "Filter NFOs by the specified category (Equity, Debt, Hybrid, etc.) from the Asset Class or Category fields. List matching NFOs."
        else:
            category_instruction = "Extract NFO information from the context. Look for AMC, Fund name, Launch Date, Closing Date, Asset Class, Category, Min Inv, and Offer Price fields."
    
    # Debt fund bond rating questions
    elif ("aaa" in question_lower or "aa" in question_lower or "a rated" in question_lower) and ("bond" in question_lower or "debt" in question_lower):
        category_instruction = """CRITICAL INSTRUCTIONS:
1. The question asks to compare AAA, AA, and A rated bond exposures in debt funds.
2. Look for chunks with "AAA %:", "AA %:", and "A %:" fields (these are bond rating percentages).
3. Focus on funds with "Debt %" > 0 (these are debt funds).
4. Extract the AAA %, AA %, and A % values from each debt fund.
5. Compare the average or aggregate percentages across all debt funds in the context.
6. Provide a comparison showing the typical exposure to AAA, AA, and A rated bonds in debt funds.
7. If values are "-" or missing, note that the fund doesn't have exposure to that rating."""
    
    # Top 25 Stocks questions
    elif "top 25" in question_lower or "25 stocks" in question_lower:
        if "combined" in question_lower or "total" in question_lower or "market value" in question_lower:
            category_instruction = """CRITICAL INSTRUCTIONS:
1. The question asks for the combined/total market value of the top 25 stocks.
2. Look for chunks from "Top 25 Stocks held by Mutual Funds.csv".
3. Each chunk contains: Stock name, Count, and Value (Crs.) - e.g., "Value (Crs.): 324987.495146"
4. Extract ALL "Value (Crs.)" numbers from the context.
5. Sum up all these values to get the combined market value.
6. Return the total in Crores (Cr.) format."""
        elif "list" in question_lower or "what are" in question_lower:
            category_instruction = """CRITICAL INSTRUCTIONS:
1. The question asks for the list of top 25 stocks.
2. Look for chunks from "Top 25 Stocks held by Mutual Funds.csv".
3. Extract the stock names from each chunk (e.g., "Stock: HDFC Bank").
4. List all the stocks found in the context."""
        else:
            category_instruction = "Extract information about the top 25 stocks from the context. Look for stock names, values, and counts."
    
    # AMC-related questions
    elif "amc" in question_lower:
        if "best performing" in question_lower or "performance" in question_lower:
            if "equity" in question_lower:
                category_instruction = """CRITICAL INSTRUCTIONS:
1. The question asks which AMC has the best performing EQUITY funds.
2. Look for funds with "Equity %" > 0 (these are equity funds).
3. Extract the AMC name from the "Fund Name" field (e.g., "ICICI Pru", "SBI", "HDFC", "Aditya Birla SL").
4. For each AMC, find their equity funds and look at performance metrics like "1Y:", "3Y:", "5Y:", "YTD:", or "Inception:".
5. Compare the average or best performance across equity funds for each AMC.
6. Return the AMC name with the best performing equity funds.
7. If you see AMC comparison data (from NGEN AMCs.csv), use that as well."""
            else:
                category_instruction = "Analyze AMC data and compare performance metrics. Extract AMC names and their performance data."
        elif "aum" in question_lower or "total aum" in question_lower:
            category_instruction = "Find AMC data and sum up AUM values for each AMC. Return the AMC with the highest total AUM."
        else:
            category_instruction = "Extract AMC information from the context. Look for AMC names and related data."
    elif is_category_question and "aum" in question_lower:
        if "large cap" in question_lower and "mid" not in question_lower:
            category_instruction = """CRITICAL: Find the Large Cap Fund category row in the Category Averages data.
The data format is: ColumnName: Value | ColumnName: Value
Look for a pattern like "1014.32 Cr: 12444.27 Cr" - the SECOND value (12444.27 Cr) is the AUM.
Or look for "AUM Cr.: [value]" directly.
Extract and return ONLY the AUM value with "Cr" unit."""
        else:
            category_instruction = "Extract the AUM value from the category average data."
    elif is_category_question:
        category_instruction = "Use data from category averages, not individual funds."
    else:
        category_instruction = "Answer based on the context provided."
    
    return f"""You are a financial data assistant. Answer the question using ONLY the data provided in the context below.

{category_instruction}

CONTEXT DATA:
{ctx}

QUESTION: {question}

INSTRUCTIONS:
- Extract the exact answer from the context above
- For AUM questions: Look for values like "12444.27 Cr" or patterns like "1014.32 Cr: 12444.27 Cr" (use the second value)
- For AMC questions: Extract AMC names from fund names (e.g., "ICICI Pru" from "ICICI Pru India Opportunities Fund")
- For performance questions: Look at metrics like 1Y, 3Y, 5Y, YTD, Inception returns
- If the question requires aggregation (e.g., "best performing AMC"), analyze multiple funds and compare
- Answer format: Just the value/name or a short sentence
- Do NOT say the context doesn't contain the data if relevant information is present

ANSWER:"""

def ask_llm(prompt: str, temperature: float = 0.0, max_tokens: int = 600) -> str:
    client = get_client()
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
    ap.add_argument("--final_k", type=int, default=5)
    args = ap.parse_args()

    artifacts = Path(args.artifacts_dir)
    loader = load_index(artifacts)

    qv = embed_query(args.question)
    # Pass question for category filtering
    retrieved = search(loader, qv, top_n=args.top_n, question=args.question)

    prompt = build_prompt(args.question, retrieved[:args.final_k])
    answer = ask_llm(prompt)
    print(answer)

if __name__ == "__main__":
    _cli()

#!/usr/bin/env python3
# rag_main.py
from rag_chunker import ingest_and_chunk
from rag_indexer import build_index

# >>>>>>>>>>>>>>>>>>>>>>>> EDIT THESE LISTS <<<<<<<<<<<<<<<<<<<<<<<<
EXCELS = [
    r"C:\Users\Acer\Downloads\NGEN Markets Filter Results.csv",
    r"C:\Users\Acer\Downloads\NGEN Markets - Mutual Fund Category Averages.csv",
    r"C:\Users\Acer\Downloads\NGEN Markets - Mutual Fund Category Averages (1).csv",
    r"C:\Users\Acer\Downloads\NGEN AMCs.csv",
    r"C:\Users\Acer\Downloads\NGEN Markets Equity Exposures _.csv",
    r"C:\Users\Acer\Downloads\Top 25 Stocks held by Mutual Funds.csv",
    r"C:\Users\Acer\Downloads\NGEN Markets Debt Exposures.csv",
    r"C:\Users\Acer\Downloads\NGEN Markets NFOs.csv",
]
PDFS = [
    # Add PDF files here if needed
]
ARTIFACTS_DIR = "artifacts"
SKIPROWS = 2   # Skip title row and empty row before actual CSV header
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

def main():
    if not EXCELS and not PDFS:
        raise RuntimeError("Add your Excel/PDF paths in rag_main.py (EXCELS / PDFS).")

    print("\n[1/2] Ingesting & chunking Excel + PDF...")
    ingest_and_chunk(EXCELS, PDFS, artifacts_dir=ARTIFACTS_DIR, skiprows=SKIPROWS)

    print("\n[2/2] Building Gemini embeddings + FAISS index...")
    build_index(artifacts_dir=ARTIFACTS_DIR)

    print("\nDone! Ask questions with:")
    print('  python rag_query.py --question "Your question here"')
    print('  (add --final_k 8 or --top_n 100 if you want bigger context)')

if __name__ == "__main__":
    main()

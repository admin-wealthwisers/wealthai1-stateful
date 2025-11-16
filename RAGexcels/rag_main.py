#!/usr/bin/env python3
# rag_main.py
from rag_chunker import ingest_and_chunk
from rag_indexer import build_index

# >>>>>>>>>>>>>>>>>>>>>>>> EDIT THESE LISTS <<<<<<<<<<<<<<<<<<<<<<<<
EXCELS = [
    r"C:\Users\Acer\Documents\ragfolders\Healthinsurance\QnA--LIFE INSURANCE.xlsx",
    r"C:\Users\Acer\Documents\ragfolders\Healthinsurance\QnA--HEALTH INSURANCE.xlsx",
    r"C:\Users\Acer\Documents\ragfolders\Healthinsurance\QnA-- HEALTH INSURANCE 2.xlsx",
    r"C:\Users\Acer\Documents\ragfolders\MF\amfi_qna_sample.xlsx",
    r"C:\Users\Acer\Documents\ragfolders\MF\mutual_fund_qna.xlsx",
    r"C:\Users\Acer\Documents\ragfolders\MF\mutual_fund_qna_batch2.xlsx",
    r"C:\Users\Acer\Documents\ragfolders\MF\mutual_fund_qna_batch3.xlsx",
    r"C:\Users\Acer\Documents\ragfolders\MF\MutualFunds_QnA.xlsx",
    r"C:\Users\Acer\Documents\ragfolders\MF\MutualFunds_QnA_Batch4.xlsx",
    r"C:\Users\Acer\Documents\ragfolders\MF\MutualFunds_QnA_Batch5.xlsx",
    r"C:\Users\Acer\Documents\ragfolders\MF\MutualFunds_QnA_Batch6.xlsx",
    r"C:\Users\Acer\Documents\ragfolders\MF\MutualFunds_QnA_Batch7.xlsx",
    r"C:\Users\Acer\Documents\ragfolders\MF\MutualFunds_QnA_Batch9.xlsx",
    r"C:\Users\Acer\Documents\ragfolders\MF\MutualFunds_QnA_Batch10.xlsx",
    r"C:\Users\Acer\Documents\ragfolders\MF\MutualFunds_QnA_Batch11.xlsx",
    r"C:\Users\Acer\Documents\ragfolders\MF\MutualFunds_QnA_Batch12.xlsx",
    r"C:\Users\Acer\Documents\ragfolders\MF\MutualFunds_QnA_Batch13.xlsx",
    r"C:\Users\Acer\Documents\ragfolders\MF\MutualFunds_QnA_Batch14.xlsx",
    r"C:\Users\Acer\Documents\ragfolders\MF\MutualFunds_QnA_Batch15.xlsx",
    r"C:\Users\Acer\Documents\ragfolders\MF\MutualFunds_QnA_Batch16.xlsx",
    r"C:\Users\Acer\Documents\ragfolders\MF\MutualFunds_QnA_Latest.xlsx",
    r"C:\Users\Acer\Documents\ragfolders\MF\MutualFunds_QnA_New (1).xlsx",
    r"C:\Users\Acer\Documents\ragfolders\MF\MutualFunds_QnA_New (2).xlsx",
    r"C:\Users\Acer\Documents\ragfolders\MF\MutualFunds_QnA_New (3).xlsx",
    r"C:\Users\Acer\Documents\ragfolders\MF\MutualFunds_QnA_New.xlsx",
    r"C:\Users\Acer\Documents\ragfolders\MF\MutualFunds_QnA_NewBatch.xlsx",
]
PDFS = [
    # Add PDF files here if needed
]
ARTIFACTS_DIR = "artifacts"
SKIPROWS = 0   # if your Excel has a title banner above headers, set to 1 (or more)
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

def main():
    if not EXCELS and not PDFS:
        raise RuntimeError("Add your Excel/PDF paths in rag_main.py (EXCELS / PDFS).")

    print("\n[1/2] Ingesting & chunking Excel + PDF...")
    ingest_and_chunk(EXCELS, PDFS, artifacts_dir=ARTIFACTS_DIR, skiprows=SKIPROWS)

    print("\n[2/2] Building OpenAI embeddings + FAISS index...")
    build_index(artifacts_dir=ARTIFACTS_DIR)

    print("\nDone! Ask questions with:")
    print('  python rag_query.py --question "Your question here"')
    print('  (add --final_k 8 or --top_n 100 if you want bigger context)')

if __name__ == "__main__":
    main()

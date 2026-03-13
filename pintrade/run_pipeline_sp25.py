"""
Run the full unstructured data pipeline for all 25 S&P500 tickers.
Idempotent: existing chunks/scores are skipped (INSERT OR REPLACE / UNIQUE on chunk_id).

Run from project root:
    cd E:/emo/workspace && python pintrade/run_pipeline_sp25.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import sqlite3
import pandas as pd
from pintrade.data.pipeline.pipeline import run_pipeline

SP25 = [
    "AAPL", "MSFT", "GOOG", "AMZN", "NVDA",
    "META", "TSLA", "BRK-B", "JPM", "JNJ",
    "V",    "PG",   "UNH",   "HD",  "MA",
    "DIS",  "BAC",  "XOM",   "CVX", "WMT",
    "NFLX", "ADBE", "CRM",   "AMD", "INTC",
]

BASE   = Path(__file__).parent / "data"
RESULT = run_pipeline(
    tickers     = SP25,
    start_date  = "2019-01-01",
    end_date    = "2024-12-31",
    form_types  = ["10-K", "8-K"],
    include_news= True,
    run_sentiment=True,
    base_dir    = BASE / "filings",
    db_path     = BASE / "financials.db",
    vectordb_dir= BASE / "vectordb",
    sentiment_db= BASE / "sentiment.db",
)

print("\n" + "="*60)
print("PIPELINE COMPLETE")
print("="*60)
for k, v in RESULT.items():
    print(f"  {k:<25} {v}")

# ── Coverage report ────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SENTIMENT COVERAGE REPORT")
print("="*60)
conn = sqlite3.connect(str(BASE / "sentiment.db"))
df = pd.read_sql("""
    SELECT ticker, doc_type, COUNT(*) as chunks
    FROM sentiment
    GROUP BY ticker, doc_type
    ORDER BY ticker, doc_type
""", conn)
conn.close()

pivot = df.pivot_table(index="ticker", columns="doc_type", values="chunks",
                       aggfunc="sum", fill_value=0)
pivot["TOTAL"] = pivot.sum(axis=1)
pivot = pivot.reindex(SP25)   # order by SP25 list

print(pivot.to_string())
print(f"\nTickers with 0 rows: {list(pivot[pivot['TOTAL'] == 0].index)}")
print(f"Tickers with <100 rows: {list(pivot[pivot['TOTAL'] < 100].dropna().index)}")

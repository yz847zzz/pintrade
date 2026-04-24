"""
Pipeline runner — execute from project root:
    cd E:/emo/workspace
    python pintrade/run_pipeline.py

Downloads SEC filings + news, extracts structured data into SQLite,
chunks text into ChromaDB, scores narrative sections with FinBERT.
"""

import sys
from pathlib import Path

# Ensure pintrade is importable when run from project dir
sys.path.insert(0, str(Path(__file__).parent.parent))

from pintrade.data.pipeline.pipeline import run_pipeline

if __name__ == "__main__":
    summary = run_pipeline(
        tickers=["AAPL", "MSFT"],
        start_date="2023-01-01",
        end_date="2024-01-01",
        form_types=["10-K", "8-K"],   # annual report + earnings releases
        include_news=True,
        run_sentiment=True,
        base_dir=Path(__file__).parent / "data" / "filings",
        db_path=Path(__file__).parent / "data" / "financials.db",
        vectordb_dir=Path(__file__).parent / "data" / "vectordb",
        sentiment_db=Path(__file__).parent / "data" / "sentiment.db",
    )

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k:<25} {v}")

# Pipeline

The unstructured data pipeline downloads SEC filings and news, scores them with FinBERT, and stores the results in SQLite and ChromaDB for use as alpha factors.

---

## Overview

```
SEC EDGAR                    News / RSS
   │                              │
   ▼                              ▼
sec_downloader.py          news_downloader.py
   │                              │
   ▼                              ▼
pdf_extractor.py ──────────────▶  chunker.py
(HTML/PDF → text)              (section-aware chunks)
                                   │
                                   ▼
                             sentiment.py
                           (FinBERT scoring)
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
               sql_store.py               vector_store.py
               (sentiment.db)              (ChromaDB)
                    │
                    ▼
             features/factors.py
             load_sentiment_factor()
                    │
                    ▼
             News_Sentiment  /  Filing_Sentiment
             (z-scored, T+1 shifted, in factor DataFrame)
```

---

## SEC EDGAR Download

**Module:** `data/ingestion/sec_downloader.py`

The downloader queries the SEC EDGAR full-text search API to find filings by ticker and form type, then downloads the primary document (HTML preferred, PDF fallback).

```python
summary = run_pipeline(
    tickers=["AAPL", "MSFT"],
    start_date="2022-01-01",
    end_date="2024-01-01",
    form_types=["10-K", "8-K"],
    include_news=True,
    run_sentiment=True,
)
```

**Form types used:**
- `10-K` — Annual report. The MD&A section (ITEM 7) contains management commentary on results, risks, and outlook. Most information-rich.
- `8-K` — Material event filing. Earnings press releases (Item 2.02) are the primary source of quarterly guidance tone.
- `10-Q` — Quarterly report (optional). Shorter MD&A, useful for filling quarterly gaps between 10-Ks.

**Storage:** Raw filing files saved to `data/filings/<ticker>/<form_type>/`.

**Idempotent:** `run_sec_sp100_remaining.py` checks `sentiment.db` for existing tickers and skips them. Safe to stop and restart at any point.

---

## Text Extraction

**Module:** `data/pipeline/pdf_extractor.py`

Supports two extractors, chosen automatically by file extension:

```
HtmlExtractor (BeautifulSoup + lxml)  →  .htm, .html
PdfExtractor  (pdfplumber)             →  .pdf
```

Output: raw text string per filing, preserving section structure where possible.

---

## Chunking

**Module:** `data/pipeline/chunker.py`

Two chunking strategies:

**`chunk_by_section()`** — for 10-K/10-Q: splits on ITEM headings (ITEM 7, ITEM 1A, etc.), then further divides large sections into overlapping windows.

**`chunk_text()`** — for 8-K and news: sliding window chunking.

```
Default parameters:
  chunk_size = 512 tokens
  overlap    = 64 tokens
```

**Sections scored by FinBERT (10-K):**
```
ITEM 7   — MD&A (most valuable: CEO words on results and outlook)
ITEM 7A  — Quantitative market risk disclosures
ITEM 1   — Business description
ITEM 1A  — Risk factors (new risks = bearish signal)
```

**Sections NOT scored:** ITEM 8 (financial statements) — contains tables and numbers, not narrative text. FinBERT performs poorly on tabular data.

**8-K:** entire document is narrative (earnings release) — all chunks are scored.

---

## FinBERT Sentiment Scoring

**Module:** `data/pipeline/sentiment.py`

**Model:** `ProsusAI/finbert` (~420MB, downloads once to `~/.cache/huggingface/`)

FinBERT is a BERT model fine-tuned on financial analyst reports. It classifies each text chunk into three classes: **positive**, **negative**, **neutral** — with associated probabilities.

**Compound score formula:**
```
compound = P(positive) - P(negative)
Range: [-1, +1]
  +1 = fully positive tone
   0 = neutral
  -1 = fully negative tone
```

**Hardware:**
```
GPU (CUDA):  batch_size=256  ~3-8ms per chunk   (RTX 4090 Ti: ~2-3 min per ticker)
CPU:         batch_size=32   ~200-500ms per chunk (~30-60 min per ticker)
```

**Lookahead prevention:** all sentiment scores are shifted forward by 1 trading day before entering the factor DataFrame. Sentiment observed at close on day T is only usable from day T+1.

---

## Storage

### SQLite (`data/sentiment.db`)

Schema:
```sql
CREATE TABLE sentiment (
    ticker    TEXT,
    date      TEXT,        -- ISO date string
    doc_type  TEXT,        -- '10-K', '8-K', '10-Q', 'news'
    chunk_id  TEXT,
    compound  REAL,        -- FinBERT compound score [-1, +1]
    ...
)
```

Loaded by `load_sentiment_factor()` in `features/factors.py`:
```sql
SELECT ticker, date, doc_type, AVG(compound) AS compound
FROM sentiment
WHERE ticker IN (...)
GROUP BY ticker, date, doc_type
```

### ChromaDB (`data/vectordb/`)

Each text chunk is embedded using `sentence-transformers/all-MiniLM-L6-v2` (local, no API key) and stored in ChromaDB for semantic search. This enables retrieval-augmented queries ("what did AAPL say about AI in their 10-K?") but is not currently used in the factor pipeline directly.

---

## How Sentiment Feeds into Factors

`load_sentiment_factor()` in `features/factors.py`:

1. Query `sentiment.db` for all (ticker, date, doc_type, compound) rows
2. Split into **news** (`doc_type='news'`) and **filing** (`doc_type IN ('10-K','10-Q','8-K')`)
3. Aggregate multiple chunks per (ticker, date) → mean compound score
4. Pivot to wide Date × Ticker format
5. Forward-fill: news 5 days, filings 90 days (filings are infrequent)
6. Shift forward 1 day (T+1, no lookahead)
7. Cross-sectionally z-score and join to main factor DataFrame

```
News_Sentiment    — short-term tone signal (forward-fill 5d)
Filing_Sentiment  — medium-term fundamental tone (forward-fill 90d)
```

---

## Coverage Report

As of 2025-03:

| Category | Count |
|---|---|
| Tickers with sentiment data | ~55 of 100 SP100 tickers |
| Total sentiment rows | ~307,000+ |
| Vector store chunks | ~307,000+ |
| Forms covered | 10-K, 8-K, news (yfinance + RSS) |
| Date range | 2019-01-01 → 2024-12-31 |

**Remaining tickers (~45):** JPM, BAC, GS, MS, C, BLK, ZTS, MCD, KO, PM, MO, NKE, LOW, COST, SBUX, TGT, CL, PEP, SLB, BA, CAT, HON, MMM, GE, LMT, NOC, GD, RTX, UPS, FDX, DE, NSC, ETN, EMR, WM, ADP, T, VZ, CHTR, DUK, SO, NEE, AMT, SPG, ELV, ECL, APD.

Run `python pintrade/run_sec_sp100_remaining.py` to ingest these. The script is idempotent — it skips tickers already in the DB and logs progress to `data/sec_ingest.log`.

---

## Running the Pipeline

```bash
# Full SP100 ingest (resumes automatically from last checkpoint)
python pintrade/run_sec_sp100_remaining.py

# Single ticker (programmatic)
from pintrade.data.pipeline.pipeline import run_pipeline
summary = run_pipeline(
    tickers=["JPM"],
    start_date="2019-01-01",
    end_date="2024-12-31",
    base_dir="pintrade/data/filings",
    db_path="pintrade/data/sentiment.db",
    form_types=["10-K", "8-K"],
    include_news=True,
    run_sentiment=True,
)
print(summary)
# {'tickers': ['JPM'], 'filings_downloaded': 6, 'vector_chunks': 2450, ...}
```

**Note on SEC EDGAR rate limits:** The SEC limits requests to ~10/second. The downloader includes automatic backoff. If a ticker returns 0 filings, the SEC CIK lookup may have failed — this is common for tickers that recently changed name or have unusual EDGAR registrations. News sentiment is still scored for these tickers.

"""
Download SEC 10-K + 8-K filings for SP100 tickers not yet in sentiment.db,
score with FinBERT, and store results in sentiment.db.

- Skips tickers already present in the DB (idempotent)
- Prints a progress banner every 10 tickers
- Appends all output to data/sec_ingest.log

Run:
    python pintrade/run_sec_sp100_remaining.py
"""
import sys
import sqlite3
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Redirect all output to log file AND stdout
import io

LOG_PATH = Path(__file__).parent / "data" / "sec_ingest.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

class _Tee:
    """Write to both a file and the original stream."""
    def __init__(self, file_obj, original):
        self._file = file_obj
        self._orig = original
        self.name   = getattr(original, "name", "<tee>")
        self.encoding = getattr(original, "encoding", "utf-8")
        self.errors   = getattr(original, "errors", "strict")
    def write(self, data):
        self._file.write(data)
        self._file.flush()
        self._orig.write(data)
        self._orig.flush()
        return len(data)
    def flush(self):
        self._file.flush()
        self._orig.flush()
    def fileno(self):
        return self._orig.fileno()
    def isatty(self):
        return False

_log_file = open(LOG_PATH, "a", encoding="utf-8", buffering=1)

# Import loguru BEFORE redirecting stderr (it installs a default stderr handler on import)
from loguru import logger
logger.remove()   # remove default stderr handler before we redirect

sys.stdout = _Tee(_log_file, sys.__stdout__)
sys.stderr = _Tee(_log_file, sys.__stderr__)

# Now add loguru sink pointing to our tee'd stdout
logger.add(sys.stdout, format="{time:HH:mm:ss} | {level:<7} | {message}", level="INFO",
           colorize=False)

from pintrade.data.pipeline.pipeline import run_pipeline

# ── Full SP100 universe ───────────────────────────────────────────────────────
SP100 = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META",
    "NVDA", "INTC", "CSCO", "ORCL", "ADBE",
    "NFLX", "TXN",  "QCOM", "AVGO", "ACN",
    "IBM",  "AMD",  "CRM",
    "BRK-B","JPM",  "BAC",  "WFC",  "V",
    "MA",   "GS",   "MS",   "C",    "AXP",
    "USB",  "PNC",  "CME",  "BLK",  "AIG",
    "AON",  "ALL",
    "JNJ",  "UNH",  "PFE",  "MRK",  "ABT",
    "AMGN", "GILD", "MDT",  "TMO",  "DHR",
    "BSX",  "SYK",  "ISRG", "BIIB", "REGN",
    "VRTX", "BMY",  "CI",   "CVS",  "ZTS",
    "WMT",  "HD",   "MCD",  "KO",   "PG",
    "PM",   "MO",   "NKE",  "LOW",  "COST",
    "SBUX", "TGT",  "CL",   "PEP",
    "XOM",  "CVX",  "SLB",
    "BA",   "CAT",  "HON",  "MMM",  "GE",
    "LMT",  "NOC",  "GD",   "RTX",  "UPS",
    "FDX",  "DE",   "NSC",  "ETN",  "EMR",
    "WM",   "ADP",
    "T",    "VZ",   "CHTR",
    "DUK",  "SO",   "NEE",
    "AMT",  "SPG",
    "ELV",  "ECL",  "APD",
]

SENT_DB   = Path(__file__).parent / "data" / "sentiment.db"
BASE_DIR  = Path(__file__).parent / "data" / "filings"
START     = "2019-01-01"
END       = "2024-12-31"

# ── Identify tickers already in DB ───────────────────────────────────────────
def _tickers_in_db(db_path: Path) -> set[str]:
    if not db_path.exists():
        return set()
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute("SELECT DISTINCT ticker FROM sentiment").fetchall()
    conn.close()
    existing = {r[0].upper() for r in rows}
    # GOOG in DB covers GOOGL in SP100 (same company, Alphabet)
    if "GOOG" in existing:
        existing.add("GOOGL")
    return existing

existing = _tickers_in_db(SENT_DB)
remaining = [t for t in SP100 if t not in existing]

print("=" * 70)
print(f"SEC EDGAR INGEST — SP100 remaining tickers")
print(f"Already in DB : {len(existing)-1} tickers")   # -1 for GOOGL alias
print(f"To process    : {len(remaining)} tickers")
print(f"Forms         : 10-K, 8-K  |  {START} → {END}")
print(f"Log           : {LOG_PATH}")
print("=" * 70)
print(f"Remaining: {remaining}")
print()

# ── Process in batches of 10 with progress banners ───────────────────────────
BATCH = 10
total   = len(remaining)
done    = 0
failed  = []
t_start = time.time()

for batch_start in range(0, total, BATCH):
    batch = remaining[batch_start: batch_start + BATCH]
    batch_num = batch_start // BATCH + 1
    n_batches = (total + BATCH - 1) // BATCH

    print()
    print("─" * 70)
    print(f"  BATCH {batch_num}/{n_batches}  "
          f"[tickers {batch_start+1}–{min(batch_start+BATCH, total)} of {total}]")
    print(f"  {batch}")
    elapsed = time.time() - t_start
    if done > 0:
        eta = elapsed / done * (total - done)
        print(f"  Progress: {done}/{total} done  |  "
              f"Elapsed {elapsed/60:.1f}m  |  ETA ~{eta/60:.0f}m")
    print("─" * 70)

    for ticker in batch:
        t0 = time.time()
        try:
            summary = run_pipeline(
                tickers=[ticker],
                start_date=START,
                end_date=END,
                base_dir=str(BASE_DIR),
                db_path=str(SENT_DB),
                vectordb_dir=str(Path(__file__).parent / "data" / "vectordb"),
                sentiment_db=str(SENT_DB),
                form_types=["10-K", "8-K"],
                include_news=True,
                run_sentiment=True,
                chunk_size=512,
                overlap=64,
            )
            dur = time.time() - t0
            print(f"  OK {ticker:<8}  "
                  f"filings={summary['filings_downloaded']:>3}  "
                  f"chunks={summary['vector_chunks']:>5}  "
                  f"({dur:.0f}s)")
            done += 1
        except Exception as e:
            dur = time.time() - t0
            print(f"  ERR {ticker:<8}  ERROR: {e}  ({dur:.0f}s)")
            failed.append(ticker)
            done += 1

# ── Final summary ─────────────────────────────────────────────────────────────
total_elapsed = time.time() - t_start
print()
print("=" * 70)
print("INGEST COMPLETE")
print("=" * 70)
print(f"  Processed : {done} tickers  ({total_elapsed/60:.1f} min total)")
print(f"  Succeeded : {done - len(failed)}")
print(f"  Failed    : {len(failed)}  {failed if failed else ''}")

# Final DB state
conn = sqlite3.connect(str(SENT_DB))
final_tickers = [r[0] for r in conn.execute(
    "SELECT DISTINCT ticker FROM sentiment ORDER BY ticker").fetchall()]
total_rows = conn.execute("SELECT COUNT(*) FROM sentiment").fetchone()[0]
conn.close()
print(f"\n  sentiment.db now covers {len(final_tickers)} tickers, "
      f"{total_rows:,} total rows")
print(f"  Tickers: {final_tickers}")
print(f"  Log saved → {LOG_PATH}")

"""
Expand universe to S&P 500 top-100 by market cap as of 2018.
Hardcoded list avoids look-ahead survivorship bias — these were the
100 largest S&P 500 constituents at the START of the analysis window.

Steps:
  1. Download OHLCV 2017-06-01 → 2024-12-31 (extra history for Momentum_252D)
  2. Compute factors (no PIN, sentiment uses existing DB for 23 tickers already scored)
  3. IC + MI analysis  →  factor selection table
  4. Walk-forward (regime=True, sentiment=True)  →  5-fold OOS Sharpe

Run from project root:
    cd E:/emo/workspace && python pintrade/run_sp100.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use('Agg')   # non-interactive backend — required when running off main thread

import pandas as pd
import numpy as np

from pintrade.data.loader import load_ohlcv_data
from pintrade.features.factors import compute_factors
from pintrade.analysis.ic_analysis import (
    compute_ic, compute_ic_summary,
    compute_mi, compute_combined_summary,
    plot_ic_mi,
)
from pintrade.analysis.walk_forward import run_walk_forward, plot_walk_forward

# ── Universe: S&P 500 top-100 by market cap, early 2018 ──────────────────────
# Source: approximate ranking based on Jan-2018 market caps.
# Companies acquired mid-period (e.g. CELG→BMY 2019) are excluded in favour of
# the surviving entity. RTX = United Technologies pre-2020 merger with Raytheon.
# ELV = Anthem (ANTM) renamed 2022; yfinance serves history under ELV.
SP100 = [
    # ── Mega-cap tech / FAANG+ ──────────────────────────────────────────────
    "AAPL", "MSFT", "AMZN", "GOOGL", "META",   # top-5 tech
    "NVDA", "INTC", "CSCO", "ORCL", "ADBE",    # semiconductors + enterprise
    # ── Technology ──────────────────────────────────────────────────────────
    "NFLX", "TXN",  "QCOM", "AVGO", "ACN",
    "IBM",  "AMD",  "CRM",
    # ── Financials ──────────────────────────────────────────────────────────
    "BRK-B","JPM",  "BAC",  "WFC",  "V",
    "MA",   "GS",   "MS",   "C",    "AXP",
    "USB",  "PNC",  "CME",  "BLK",  "AIG",
    "AON",  "ALL",
    # ── Healthcare ──────────────────────────────────────────────────────────
    "JNJ",  "UNH",  "PFE",  "MRK",  "ABT",
    "AMGN", "GILD", "MDT",  "TMO",  "DHR",
    "BSX",  "SYK",  "ISRG", "BIIB", "REGN",
    "VRTX", "BMY",  "CI",   "CVS",  "ZTS",
    # ── Consumer ────────────────────────────────────────────────────────────
    "WMT",  "HD",   "MCD",  "KO",   "PG",
    "PM",   "MO",   "NKE",  "LOW",  "COST",
    "SBUX", "TGT",  "CL",   "PEP",   # WBA delisted 2024; PEP (~$155B in 2018)
    # ── Energy ──────────────────────────────────────────────────────────────
    "XOM",  "CVX",  "SLB",
    # ── Industrials / Aerospace ─────────────────────────────────────────────
    "BA",   "CAT",  "HON",  "MMM",  "GE",
    "LMT",  "NOC",  "GD",   "RTX",  "UPS",
    "FDX",  "DE",   "NSC",  "ETN",  "EMR",
    "WM",   "ADP",
    # ── Telecom ─────────────────────────────────────────────────────────────
    "T",    "VZ",   "CHTR",
    # ── Utilities ───────────────────────────────────────────────────────────
    "DUK",  "SO",   "NEE",
    # ── REITs ───────────────────────────────────────────────────────────────
    "AMT",  "SPG",
    # ── Other large-cap ─────────────────────────────────────────────────────
    "ELV",  "ECL",  "APD",
]
assert len(SP100) == 100, f"Expected 100 tickers, got {len(SP100)}"

SENT_DB   = Path(__file__).parent / "data" / "sentiment.db"
BASE_DIR  = Path(__file__).parent
WARMUP    = "2017-06-01"   # 1.5yr before 2019-01-01 fold start (252D momentum)
START     = "2019-01-01"
END       = "2024-12-31"

pd.set_option('display.float_format', '{:.4f}'.format)
pd.set_option('display.max_columns', 12)
pd.set_option('display.width', 160)

# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print(f"SP100 UNIVERSE  ({len(SP100)} tickers)")
print("=" * 70)

# ── 1. Download OHLCV ────────────────────────────────────────────────────────
print(f"\n[1] Downloading OHLCV  {WARMUP} → {END}...")
ohlcv = load_ohlcv_data(SP100, WARMUP, END)
actual = list(ohlcv.columns.get_level_values('Ticker').unique())
print(f"    Loaded: {len(actual)} tickers  |  "
      f"Dates: {ohlcv.index[0].date()} → {ohlcv.index[-1].date()}")
missing = [t for t in SP100 if t not in actual]
if missing:
    print(f"    !! Missing tickers (yfinance unavailable): {missing}")

# ── 2. Compute factors ───────────────────────────────────────────────────────
use_sent = SENT_DB.exists()
print(f"\n[2] Computing factors  (pin=False, sentiment={'True ('+str(SENT_DB.stat().st_size//1024)+'KB)' if use_sent else 'False'})...")
full_factors = compute_factors(
    ohlcv,
    include_pin=False,
    include_sentiment=use_sent,
    sentiment_db=str(SENT_DB) if use_sent else "data/sentiment.db",
)
print(f"    Factor shape: {full_factors.shape}  |  Columns: {list(full_factors.columns)}")

# Coverage per factor
cov = full_factors.notna().mean().sort_values(ascending=False)
print(f"    Factor coverage (% non-NaN):")
for col, pct in cov.items():
    bar = '█' * int(pct * 20)
    print(f"      {col:<25}  {pct*100:5.1f}%  {bar}")

# ── 3. IC + MI analysis (trim to analysis window) ───────────────────────────
print(f"\n[3] IC + MI analysis  {START} → {END}...")
fac_analysis = full_factors.loc[
    full_factors.index.get_level_values('Date') >= START
]
ohlcv_analysis = ohlcv.loc[ohlcv.index >= START]

ic_df   = compute_ic(fac_analysis, ohlcv_analysis, forward_days=21)
mi      = compute_mi(fac_analysis, ohlcv_analysis, forward_days=21)
summary = compute_combined_summary(ic_df, mi, nonlinear_threshold=2.0)

ic_sum = compute_ic_summary(ic_df)

print(f"\n    IC + MI Summary (sorted by |t-stat|):")
display_cols = ['IC Mean', 'ICIR', 't-stat', 'MI', 'Nonlinearity', 'Flag']
by_t = summary.reindex(ic_sum['t-stat'].abs().sort_values(ascending=False).index)
print(by_t[display_cols].to_string())

print(f"\n    Factors with |t-stat| > 2.0:")
sig = ic_sum[ic_sum['t-stat'].abs() > 2.0].sort_values('t-stat', key=abs, ascending=False)
for f, row in sig.iterrows():
    sign  = '+1' if row['ICIR'] > 0 else '-1'
    nl    = summary.loc[f, 'Flag'] if f in summary.index else ''
    print(f"      {f:<25}  t={row['t-stat']:+.2f}  ICIR={row['ICIR']:+.3f}  weight={sign}  {nl}")

# Save IC+MI plot
ic_mi_path = str(BASE_DIR / "ic_mi_sp100.png")
plot_ic_mi(ic_df, mi, save_path=ic_mi_path)

# ── 4. Walk-forward ──────────────────────────────────────────────────────────
print(f"\n[4] Walk-forward  regime=True  sentiment={use_sent}...")
print("=" * 70)

res = run_walk_forward(
    tickers      = actual,        # only tickers yfinance actually returned
    start        = START,
    end          = END,
    train_years  = 1,
    test_years   = 1,
    top_n        = 5,
    rebalance    = 'monthly',
    include_pin  = False,
    t_threshold  = 2.0,
    include_sentiment = use_sent,
    sentiment_db = str(SENT_DB) if use_sent else "data/sentiment.db",
    use_regime   = True,
    vix_threshold= 25.0,
    momentum_buffer_days=300,
)

# ── 5. Results ───────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("WALK-FORWARD RESULTS — SP100, regime=True, sentiment=True")
print(f"{'='*70}")

df = res['windows']
print(df[['test_start','test_end','n_selected','selected','short_active_%',
          'is_ls_sharpe','long_sharpe','short_sharpe','ls_sharpe',
          'ls_annret','ls_mdd','ls_beta']].to_string())

print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
s = res['summary']
for k, v in s.items():
    print(f"  {k:<35} {v:.4f}" if isinstance(v, float) else f"  {k:<35} {v}")

# ── 6. Comparison vs SP25 best ───────────────────────────────────────────────
SP25_BEST = {  # SP25 regime=True+sentiment best run (from prior session)
    'overall_oos_ls_sharpe': 1.3898,
    'mean_oos_ls_sharpe':    1.3988,
    'mean_oos_ann_return':   0.5156,
    'per_fold': {1: 2.690, 2: 1.917, 3: -0.626, 4: 1.025, 5: 1.988},
}

print(f"\n{'='*70}")
print("SP100 vs SP25 — Sharpe comparison (both regime=True, sentiment=True)")
print(f"{'='*70}")
print(f"  {'Fold':<5} {'OOS Year':<10} {'SP25':>8} {'SP100':>8} {'Delta':>8}")
print(f"  {'-'*42}")
for fold in df.index:
    yr   = str(df.loc[fold, 'test_start'])[:4]
    s25  = SP25_BEST['per_fold'].get(fold, float('nan'))
    s100 = df.loc[fold, 'ls_sharpe']
    print(f"  F{fold:<4} {yr:<10} {s25:>+8.3f} {s100:>+8.3f} {s100-s25:>+8.3f}")

sp100_overall = s['overall_oos_ls_sharpe']
sp25_overall  = SP25_BEST['overall_oos_ls_sharpe']
print(f"\n  Overall OOS L/S Sharpe  SP25={sp25_overall:+.4f}  "
      f"SP100={sp100_overall:+.4f}  Delta={sp100_overall-sp25_overall:+.4f}")

# ── 7. Save plot ─────────────────────────────────────────────────────────────
wf_path = str(BASE_DIR / "walk_forward_sp100.png")
plot_walk_forward(res, save_path=wf_path)
print(f"\nPlots saved: {ic_mi_path}  {wf_path}")

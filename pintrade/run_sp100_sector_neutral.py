"""
SP100 Sector-Neutral Factor Scoring — Walk-Forward Comparison

Compares three configurations on the SP100 universe (2019-2024):
  1. SP100 Raw          — cross-sectional z-score across all 100 tickers
  2. SP100 Sector-Neutral — z-score within each GICS sector separately
  3. SP25 Baseline      — hardcoded best result from prior run (regime=True)

Sector-neutral normalization removes common sector-level variation so that
the composite score reflects within-sector relative quality, not cross-sector
sector tilts (e.g. tech vs energy beta differences).

GICS Sectors mapped:
  Technology    (18 tickers)   Financials  (17)   Healthcare  (21)
  Consumer      (14 tickers)   Industrials (17)   Energy      ( 3)
  Utilities     ( 3 tickers)   Materials   ( 2)   Other/REIT  ( 5)

Run from project root:
    cd E:/emo/workspace && python pintrade/run_sp100_sector_neutral.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np

from pintrade.data.loader import load_ohlcv_data
from pintrade.analysis.walk_forward import run_walk_forward, plot_walk_forward

# ── Universe (same as run_sp100.py) ──────────────────────────────────────────
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
assert len(SP100) == 100

# ── GICS Sector map ───────────────────────────────────────────────────────────
# Assigned per standard GICS classification as of 2018.
# T/VZ/CHTR = Communication Services (mapped to Technology for simplicity).
# AMT/SPG (REITs) and BRK-B mapped to Financials.
# ELV = Elevance Health → Healthcare.
# ECL/APD = Materials.
SP100_SECTOR_MAP: dict[str, str] = {
    # Technology
    "AAPL": "Technology", "MSFT": "Technology", "AMZN": "Technology",
    "GOOGL": "Technology", "META": "Technology", "NVDA": "Technology",
    "INTC": "Technology",  "CSCO": "Technology", "ORCL": "Technology",
    "ADBE": "Technology",  "NFLX": "Technology", "TXN":  "Technology",
    "QCOM": "Technology",  "AVGO": "Technology", "ACN":  "Technology",
    "IBM":  "Technology",  "AMD":  "Technology", "CRM":  "Technology",
    # Communication Services → Technology bucket (similar factor dynamics)
    "T":    "Technology",  "VZ":   "Technology", "CHTR": "Technology",
    # Financials
    "BRK-B":"Financials",  "JPM":  "Financials", "BAC":  "Financials",
    "WFC":  "Financials",  "V":    "Financials",  "MA":   "Financials",
    "GS":   "Financials",  "MS":   "Financials",  "C":    "Financials",
    "AXP":  "Financials",  "USB":  "Financials",  "PNC":  "Financials",
    "CME":  "Financials",  "BLK":  "Financials",  "AIG":  "Financials",
    "AON":  "Financials",  "ALL":  "Financials",
    # REITs → Financials (no separate bucket)
    "AMT":  "Financials",  "SPG":  "Financials",
    # Healthcare
    "JNJ":  "Healthcare",  "UNH":  "Healthcare",  "PFE":  "Healthcare",
    "MRK":  "Healthcare",  "ABT":  "Healthcare",  "AMGN": "Healthcare",
    "GILD": "Healthcare",  "MDT":  "Healthcare",  "TMO":  "Healthcare",
    "DHR":  "Healthcare",  "BSX":  "Healthcare",  "SYK":  "Healthcare",
    "ISRG": "Healthcare",  "BIIB": "Healthcare",  "REGN": "Healthcare",
    "VRTX": "Healthcare",  "BMY":  "Healthcare",  "CI":   "Healthcare",
    "CVS":  "Healthcare",  "ZTS":  "Healthcare",  "ELV":  "Healthcare",
    # Consumer (Staples + Discretionary merged — both large-cap consumer)
    "WMT":  "Consumer",    "HD":   "Consumer",    "MCD":  "Consumer",
    "KO":   "Consumer",    "PG":   "Consumer",    "PM":   "Consumer",
    "MO":   "Consumer",    "NKE":  "Consumer",    "LOW":  "Consumer",
    "COST": "Consumer",    "SBUX": "Consumer",    "TGT":  "Consumer",
    "CL":   "Consumer",    "PEP":  "Consumer",
    # Energy
    "XOM":  "Energy",      "CVX":  "Energy",      "SLB":  "Energy",
    # Industrials
    "BA":   "Industrials", "CAT":  "Industrials", "HON":  "Industrials",
    "MMM":  "Industrials", "GE":   "Industrials", "LMT":  "Industrials",
    "NOC":  "Industrials", "GD":   "Industrials", "RTX":  "Industrials",
    "UPS":  "Industrials", "FDX":  "Industrials", "DE":   "Industrials",
    "NSC":  "Industrials", "ETN":  "Industrials", "EMR":  "Industrials",
    "WM":   "Industrials", "ADP":  "Industrials",
    # Utilities
    "DUK":  "Utilities",   "SO":   "Utilities",   "NEE":  "Utilities",
    # Materials
    "ECL":  "Materials",   "APD":  "Materials",
}

# Verify coverage
_missing_map = [t for t in SP100 if t not in SP100_SECTOR_MAP]
if _missing_map:
    print(f"  WARNING: tickers missing from sector map: {_missing_map}")

# Print sector sizes
from collections import Counter
_sec_counts = Counter(SP100_SECTOR_MAP[t] for t in SP100 if t in SP100_SECTOR_MAP)
print("Sector composition:")
for sec, cnt in sorted(_sec_counts.items(), key=lambda x: -x[1]):
    print(f"  {sec:<15} {cnt:>3} tickers")

# ── Config ────────────────────────────────────────────────────────────────────
SENT_DB  = Path(__file__).parent / "data" / "sentiment.db"
BASE_DIR = Path(__file__).parent
WARMUP   = "2017-06-01"
START    = "2019-01-01"
END      = "2024-12-31"

pd.set_option('display.float_format', '{:.4f}'.format)
pd.set_option('display.max_columns', 12)
pd.set_option('display.width', 160)

use_sent = SENT_DB.exists()

_WF_COMMON = dict(
    tickers             = SP100,
    start               = START,
    end                 = END,
    train_years         = 1,
    test_years          = 1,
    top_n               = 5,
    rebalance           = 'monthly',
    include_pin         = False,
    t_threshold         = 2.0,
    include_sentiment   = use_sent,
    sentiment_db        = str(SENT_DB) if use_sent else "data/sentiment.db",
    use_regime          = True,
    vix_threshold       = 25.0,
    momentum_buffer_days= 300,
)

# ── SP25 Baseline (hardcoded from prior run — regime=True, sentiment=True) ───
SP25_BEST = {
    'overall_oos_ls_sharpe': 1.3898,
    'mean_oos_ls_sharpe':    1.3988,
    'mean_oos_ann_return':   0.5156,
    'per_fold': {1: 2.690, 2: 1.917, 3: -0.626, 4: 1.025, 5: 1.988},
}

# ─────────────────────────────────────────────────────────────────────────────
print("=" * 72)
print(f"SP100 SECTOR-NEUTRAL vs RAW COMPARISON")
print("=" * 72)

# ── Run 1: SP100 Raw (cross-sectional z-score) ────────────────────────────────
print(f"\n{'─'*72}")
print("RUN 1 / 2 — SP100 RAW  (cross-sectional z-score)")
print(f"{'─'*72}")
res_raw = run_walk_forward(**_WF_COMMON)

# ── Run 2: SP100 Sector-Neutral ───────────────────────────────────────────────
print(f"\n{'─'*72}")
print("RUN 2 / 2 — SP100 SECTOR-NEUTRAL  (within-sector z-score)")
print(f"{'─'*72}")
res_sn = run_walk_forward(**_WF_COMMON, sector_map=SP100_SECTOR_MAP)

# ── Save plots ────────────────────────────────────────────────────────────────
plot_walk_forward(res_raw, save_path=str(BASE_DIR / "walk_forward_sp100_raw.png"))
plot_walk_forward(res_sn,  save_path=str(BASE_DIR / "walk_forward_sp100_sectorneutral.png"))

# ─────────────────────────────────────────────────────────────────────────────
# COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────
df_raw = res_raw['windows']
df_sn  = res_sn['windows']

print(f"\n{'='*72}")
print("THREE-WAY COMPARISON — OOS L/S Sharpe per Fold")
print("SP25 Baseline | SP100 Raw | SP100 Sector-Neutral")
print(f"{'='*72}")
print(f"  {'Fold':<5} {'OOS Year':<10} {'SP25 BL':>9} {'SP100 Raw':>10} "
      f"{'SP100 SN':>10} {'SN-Raw':>8} {'SN-BL':>8}")
print(f"  {'─'*64}")

for fold in df_raw.index:
    yr      = str(df_raw.loc[fold, 'test_start'])[:4]
    bl      = SP25_BEST['per_fold'].get(fold, float('nan'))
    raw_sr  = df_raw.loc[fold, 'ls_sharpe']
    sn_sr   = df_sn.loc[fold, 'ls_sharpe'] if fold in df_sn.index else float('nan')
    delta_r = sn_sr - raw_sr
    delta_b = sn_sr - bl
    print(f"  F{fold:<4} {yr:<10} {bl:>+9.3f} {raw_sr:>+10.3f} "
          f"{sn_sr:>+10.3f} {delta_r:>+8.3f} {delta_b:>+8.3f}")

raw_overall = res_raw['summary']['overall_oos_ls_sharpe']
sn_overall  = res_sn['summary']['overall_oos_ls_sharpe']
bl_overall  = SP25_BEST['overall_oos_ls_sharpe']

print(f"\n  {'─'*64}")
print(f"  {'Overall OOS L/S Sharpe':<30} "
      f"  SP25={bl_overall:>+7.4f}  "
      f"Raw={raw_overall:>+7.4f}  "
      f"SN={sn_overall:>+7.4f}  "
      f"ΔSN-Raw={sn_overall-raw_overall:>+7.4f}  "
      f"ΔSN-BL={sn_overall-bl_overall:>+7.4f}")

print(f"\n  {'Mean OOS L/S Sharpe':<30} "
      f"  SP25={SP25_BEST['mean_oos_ls_sharpe']:>+7.4f}  "
      f"Raw={res_raw['summary']['mean_oos_ls_sharpe']:>+7.4f}  "
      f"SN={res_sn['summary']['mean_oos_ls_sharpe']:>+7.4f}")

print(f"  {'Mean OOS Ann Return':<30} "
      f"  SP25={SP25_BEST['mean_oos_ann_return']:>+7.4f}  "
      f"Raw={res_raw['summary']['mean_oos_ls_ann_return']:>+7.4f}  "
      f"SN={res_sn['summary']['mean_oos_ls_ann_return']:>+7.4f}")

print(f"  {'Mean OOS Max Drawdown':<30} "
      f"  {'N/A':>12}  "
      f"Raw={res_raw['summary']['mean_oos_ls_max_dd']:>+7.4f}  "
      f"SN={res_sn['summary']['mean_oos_ls_max_dd']:>+7.4f}")

# ── Per-fold factor selection comparison ─────────────────────────────────────
print(f"\n{'='*72}")
print("FACTOR SELECTION — Raw vs Sector-Neutral per Fold")
print(f"{'='*72}")
for fold in df_raw.index:
    yr  = str(df_raw.loc[fold, 'test_start'])[:4]
    raw_sel = df_raw.loc[fold, 'selected']
    sn_sel  = df_sn.loc[fold,  'selected'] if fold in df_sn.index else 'N/A'
    print(f"\n  Fold {fold} OOS {yr}:")
    print(f"    Raw : {raw_sel}")
    print(f"    SN  : {sn_sel}")

# ── Summary interpretation ───────────────────────────────────────────────────
print(f"\n{'='*72}")
print("INTERPRETATION")
print(f"{'='*72}")

verdict_raw_vs_bl = "IMPROVED" if raw_overall > bl_overall + 0.05 else \
                    "WORSE"    if raw_overall < bl_overall - 0.05 else "FLAT"
verdict_sn_vs_raw = "IMPROVED" if sn_overall > raw_overall + 0.05 else \
                    "WORSE"    if sn_overall < raw_overall - 0.05 else "FLAT"

print(f"  SP100 Raw  vs SP25 Baseline   : {verdict_raw_vs_bl}  "
      f"({raw_overall-bl_overall:>+.4f} Sharpe)")
print(f"  SP100 SN   vs SP100 Raw       : {verdict_sn_vs_raw}  "
      f"({sn_overall-raw_overall:>+.4f} Sharpe)")

if sn_overall > raw_overall:
    print(f"\n  Sector-neutral normalization HELPED: removing sector-level factor")
    print(f"  variation allowed the model to better identify within-sector alpha.")
else:
    print(f"\n  Sector-neutral normalization DID NOT HELP vs raw SP100.")
    print(f"  Cross-sector variation may itself carry predictive signal, or the")
    print(f"  smaller within-sector peer groups degrade z-score quality.")

print(f"\nPlots saved:")
print(f"  {BASE_DIR / 'walk_forward_sp100_raw.png'}")
print(f"  {BASE_DIR / 'walk_forward_sp100_sectorneutral.png'}")

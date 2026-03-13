"""
Expanded universe factor IC analysis — 25 S&P 500 large caps.

Run from: cd E:/emo/workspace && python pintrade/run_ic_analysis.py

Output:
  - IC summary table (sorted by |ICIR|) printed to console
  - ic_analysis_25.png  — daily IC, cumulative IC, ICIR bar chart
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from pintrade.data.loader import load_ohlcv_data
from pintrade.features.factors import compute_factors
from pintrade.analysis.ic_analysis import (
    compute_ic, compute_ic_summary,
    compute_mi, compute_combined_summary,
    plot_ic_mi,
)

# ── Universe ──────────────────────────────────────────────────────────────────
SP25 = [
    "AAPL", "MSFT", "GOOG", "AMZN", "NVDA",
    "META", "TSLA", "BRK-B", "JPM", "JNJ",
    "V", "PG", "UNH", "HD", "MA",
    "DIS", "BAC", "XOM", "CVX", "WMT",
    "NFLX", "ADBE", "CRM", "AMD", "INTC",
]

WARMUP = "2021-06-01"   # 1.5yr before analysis start (for Momentum_252D)
START  = "2023-01-01"   # analysis period
END    = "2024-12-31"

SENT_DB    = Path(__file__).parent / "data" / "sentiment.db"
SAVE_PNG   = Path(__file__).parent / "ic_mi_analysis_25.png"

# ── 1. Load OHLCV ─────────────────────────────────────────────────────────────
print("=" * 60)
print(f"1. Loading OHLCV for {len(SP25)} tickers  ({WARMUP} → {END})...")
print("=" * 60)
ohlcv = load_ohlcv_data(SP25, WARMUP, END)
print(f"   Shape: {ohlcv.shape}  |  "
      f"Dates: {ohlcv.index[0].date()} → {ohlcv.index[-1].date()}")
actual_tickers = list(ohlcv.columns.get_level_values('Ticker').unique())
print(f"   Tickers loaded: {len(actual_tickers)}  → {actual_tickers}")

# ── 2. Compute factors ────────────────────────────────────────────────────────
print()
print("=" * 60)
print("2. Computing factors (pin=False, sentiment=True if DB exists)...")
print("=" * 60)

use_sentiment = SENT_DB.exists()
if use_sentiment:
    print(f"   sentiment.db found ({SENT_DB.stat().st_size // 1024} KB) — including sentiment factors")
else:
    print("   sentiment.db not found — skipping sentiment factors")

factor_df = compute_factors(
    ohlcv,
    include_pin=False,
    include_sentiment=use_sentiment,
    sentiment_db=SENT_DB,
)
print(f"   Factor shape: {factor_df.shape}")
print(f"   Columns: {list(factor_df.columns)}")

# Trim to analysis window (post-warmup)
factor_df = factor_df.loc[factor_df.index.get_level_values('Date') >= START]
ohlcv_analysis = ohlcv.loc[ohlcv.index >= START]
print(f"   Factor rows (post-warmup): {len(factor_df)}")

# ── 3. Coverage report ────────────────────────────────────────────────────────
print()
print("=" * 60)
print("3. Factor coverage (% non-NaN per factor)...")
print("=" * 60)
coverage = factor_df.notna().mean().sort_values(ascending=False)
for col, pct in coverage.items():
    print(f"   {col:<25} {pct*100:5.1f}%")

# ── 4. IC analysis ────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("4. Computing Rank IC (Spearman) vs 21-day forward returns...")
print("=" * 60)
ic_df = compute_ic(factor_df, ohlcv_analysis, forward_days=21)
print(f"   IC DataFrame shape: {ic_df.shape}")
print(f"   IC date range: {ic_df.index[0].date()} → {ic_df.index[-1].date()}")

# ── 5. MI analysis ────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("5. Computing Mutual Information (panel, k-NN Kraskov estimator)...")
print("=" * 60)
mi = compute_mi(factor_df, ohlcv_analysis, forward_days=21)
print("   MI values (nats):")
for factor, val in mi.sort_values(ascending=False).items():
    bar = '█' * int(val / mi.max() * 20) if not np.isnan(val) else ''
    print(f"   {factor:<25}  {val:.4f}  {bar}")

# ── 6. Combined IC + MI summary ───────────────────────────────────────────────
print()
print("=" * 60)
print("6. IC vs MI Comparison Table (sorted by MI)...")
print("=" * 60)
pd.set_option('display.float_format', '{:.4f}'.format)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 130)

combined = compute_combined_summary(ic_df, mi, nonlinear_threshold=2.0)
print(combined.to_string())

print()
print("   Columns:")
print("     IC Mean      — mean daily cross-sectional Spearman (signed, captures monotone)")
print("     ICIR         — IC / IC_Std (risk-adjusted IC; |ICIR|>0.3 is useful)")
print("     t-stat       — |t|>2.0 → statistically significant at 95%")
print("     MI           — Mutual Information in nats (captures ANY dependency)")
print("     Nonlinearity — MI / |IC Mean| (high → relationship is non-monotone)")
print("     Flag         — '* NONLINEAR': MI >> IC, use with nonlinear model")

nonlinear = combined[combined['Flag'] != '']
if not nonlinear.empty:
    print()
    print("   ── Nonlinear alpha candidates ──────────────────────────────────")
    for f, row in nonlinear.iterrows():
        print(f"   {f:<25}  MI={row['MI']:.4f}  IC={row['IC Mean']:.4f}  "
              f"Nonlinearity={row['Nonlinearity']:.1f}x")
    print()
    print("   These factors carry information that Spearman cannot capture.")
    print("   Consider tree-based or kernel models (XGBoost, RBF-SVM) to exploit them.")

# ── 7. Plot ───────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print(f"7. Saving IC + MI plot → {SAVE_PNG}...")
print("=" * 60)
plot_ic_mi(ic_df, mi, save_path=str(SAVE_PNG))

# ── 8. Weight recommendations ─────────────────────────────────────────────────
print()
print("=" * 60)
print("8. Recommended DEFAULT_WEIGHTS based on IC + MI...")
print("=" * 60)
print("   (linear model: use IC-significant factors; flag nonlinear for ML model)")
print()
ic_sum = combined.copy()
for factor in ic_sum.index:
    t    = ic_sum.loc[factor, 't-stat']
    icir = ic_sum.loc[factor, 'ICIR']
    flag = ic_sum.loc[factor, 'Flag']
    if pd.isna(t) or abs(t) < 2.0:
        note = "0  # not significant"
    elif icir > 0:
        note = "1"
    else:
        note = "-1  # reversal factor"
    if flag:
        note += "  # nonlinear — consider ML model"
    print(f"   '{factor}': {note},")

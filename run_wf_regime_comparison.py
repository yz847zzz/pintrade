"""
3-way walk-forward comparison:
  1. Baseline (no sentiment, regime=True)   — confirm prior 1.37
  2. Sentiment + regime=True                — new
  3. Sentiment + regime=False               — already run (0.332), included for table

Run from project root:
    cd E:/emo/workspace && python pintrade/run_wf_regime_comparison.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from pintrade.analysis.walk_forward import run_walk_forward, plot_walk_forward

SP25 = [
    "AAPL", "MSFT", "GOOG", "AMZN", "NVDA",
    "META", "TSLA", "BRK-B", "JPM", "JNJ",
    "V",    "PG",   "UNH",   "HD",  "MA",
    "DIS",  "BAC",  "XOM",   "CVX", "WMT",
    "NFLX", "ADBE", "CRM",   "AMD", "INTC",
]

SENT_DB = Path(__file__).parent / "data" / "sentiment.db"

pd.set_option('display.float_format', '{:.3f}'.format)
pd.set_option('display.max_columns', 25)
pd.set_option('display.width', 220)

_common = dict(
    tickers=SP25,
    start='2019-01-01',
    end='2024-12-31',
    train_years=1,
    test_years=1,
    top_n=5,
    rebalance='monthly',
    include_pin=False,
    t_threshold=2.0,
    vix_threshold=25.0,
)

# ── Run 1: Baseline + regime=True ────────────────────────────────────────────
print("\n" + "="*70)
print("RUN 1: BASELINE — no sentiment, regime=True")
print("="*70)
res1 = run_walk_forward(**_common, include_sentiment=False, use_regime=True)

# ── Run 2: Sentiment + regime=True ───────────────────────────────────────────
print("\n" + "="*70)
print("RUN 2: SENTIMENT — filing sentiment, regime=True")
print("="*70)
res2 = run_walk_forward(**_common, include_sentiment=True,
                        sentiment_db=str(SENT_DB), use_regime=True)

# ── Sentinel value for Run 3 (already computed) ──────────────────────────────
# Reproducing Run 3 figures from the prior session output
PRIOR_RUN3 = {
    'overall_oos_ls_sharpe': 0.3317,
    'mean_oos_ls_sharpe':    0.1194,
    'pct_positive_oos_ls':   0.60,
    'mean_oos_ls_ann_return': 0.2124,
    'mean_oos_ls_max_dd':    -0.2703,
    'per_fold_ls_sharpe': {1: 2.632, 2: 0.915, 3: -1.256, 4: -2.778, 5: 1.084},
}

# ── Per-fold tables ────────────────────────────────────────────────────────────
for label, res in [("RUN 1 — No sentiment, regime=True", res1),
                   ("RUN 2 — Sentiment,    regime=True",  res2)]:
    print(f"\n{'='*70}")
    print(label)
    print(f"{'='*70}")
    print(res['windows'].to_string())
    print()
    for k, v in res['summary'].items():
        print(f"  {k:<35} {v:.4f}" if isinstance(v, float) else f"  {k:<35} {v}")

# ── 3-way fold-by-fold comparison ─────────────────────────────────────────────
print(f"\n{'='*70}")
print("3-WAY COMPARISON — OOS L/S Sharpe per Fold")
print(f"{'='*70}")

df1 = res1['windows']
df2 = res2['windows']
r3  = PRIOR_RUN3['per_fold_ls_sharpe']

hdr = f"  {'Fold':<5} {'OOS Year':<9} {'(1)NoSent+Regime':>17} {'(2)Sent+Regime':>15} {'(3)Sent+NoRegime':>17}  {'2vs1':>7}  {'2vs3':>7}"
print(hdr)
print("  " + "-"*80)

for fold in df1.index:
    yr   = str(df1.loc[fold, 'test_start'])[:4]
    sr1  = df1.loc[fold, 'ls_sharpe']
    sr2  = df2.loc[fold, 'ls_sharpe'] if fold in df2.index else float('nan')
    sr3  = r3.get(fold, float('nan'))
    d21  = sr2 - sr1
    d23  = sr2 - sr3
    print(f"  F{fold:<4} {yr:<9} {sr1:>+17.3f} {sr2:>+15.3f} {sr3:>+17.3f}  {d21:>+7.3f}  {d23:>+7.3f}")

s1 = res1['summary']['overall_oos_ls_sharpe']
s2 = res2['summary']['overall_oos_ls_sharpe']
s3 = PRIOR_RUN3['overall_oos_ls_sharpe']

print(f"\n  {'Overall':>5} {'':9} {s1:>+17.4f} {s2:>+15.4f} {s3:>+17.4f}  {s2-s1:>+7.4f}  {s2-s3:>+7.4f}")

print(f"\n  Columns: (1) regime suppresses short in bull markets")
print(f"           (2) same + Filing_Sentiment as additional factor")
print(f"           (3) sentiment but NO regime (from prior run)")
print(f"  Delta columns: positive = improvement over reference")

# ── Per-fold regime activity ───────────────────────────────────────────────────
print(f"\n{'='*70}")
print("SHORT LEG ACTIVITY (% days short was active per fold)")
print(f"{'='*70}")
print(f"  {'Fold':<5} {'OOS Year':<9} {'Regime=T (1)':>13} {'Regime=T (2)':>13} {'Regime=F':>10}")
print("  " + "-"*55)
for fold in df1.index:
    yr  = str(df1.loc[fold, 'test_start'])[:4]
    p1  = df1.loc[fold, 'short_active_%']
    p2  = df2.loc[fold, 'short_active_%'] if fold in df2.index else float('nan')
    print(f"  F{fold:<4} {yr:<9} {p1:>12.0f}%  {p2:>12.0f}%  {'100%':>9}")

# ── Plots ──────────────────────────────────────────────────────────────────────
base_dir = Path(__file__).parent
plot_walk_forward(res1, save_path=str(base_dir / "walk_forward_regime_baseline.png"))
plot_walk_forward(res2, save_path=str(base_dir / "walk_forward_regime_sentiment.png"))
print("\nPlots saved.")

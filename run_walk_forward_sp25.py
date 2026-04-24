"""
Walk-forward with and without sentiment factors — SP25 universe 2019-2024.

Run from project root:
    cd E:/emo/workspace && python pintrade/run_walk_forward_sp25.py
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

SENT_DB   = Path(__file__).parent / "data" / "sentiment.db"
USE_SENT  = SENT_DB.exists()

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
)

# ── Run 1: no sentiment (baseline) ───────────────────────────────────────────
print("\n" + "="*70)
print("RUN 1: NO SENTIMENT (baseline)")
print("="*70)
res_base = run_walk_forward(**_common, include_sentiment=False)

# ── Run 2: with sentiment ─────────────────────────────────────────────────────
if USE_SENT:
    print("\n" + "="*70)
    print(f"RUN 2: WITH SENTIMENT  (sentiment.db: {SENT_DB})")
    print("="*70)
    res_sent = run_walk_forward(**_common,
                                include_sentiment=True,
                                sentiment_db=str(SENT_DB))
else:
    print("\nsentiment.db not found — skipping sentiment run")
    res_sent = None

# ── Per-fold results ──────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("BASELINE — Per-fold walk-forward results")
print(f"{'='*70}")
print(res_base['windows'].to_string())

print(f"\n{'='*70}")
print("BASELINE — Summary")
print(f"{'='*70}")
for k, v in res_base['summary'].items():
    print(f"  {k:<35} {v:.4f}" if isinstance(v, float) else f"  {k:<35} {v}")

if res_sent:
    print(f"\n{'='*70}")
    print("WITH SENTIMENT — Per-fold walk-forward results")
    print(f"{'='*70}")
    print(res_sent['windows'].to_string())

    print(f"\n{'='*70}")
    print("WITH SENTIMENT — Summary")
    print(f"{'='*70}")
    for k, v in res_sent['summary'].items():
        print(f"  {k:<35} {v:.4f}" if isinstance(v, float) else f"  {k:<35} {v}")

    # ── Side-by-side L/S Sharpe ────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SENTIMENT IMPACT — L/S Sharpe per Fold")
    print(f"{'='*70}")
    df_b = res_base['windows']
    df_s = res_sent['windows']
    print(f"  {'Fold':<6} {'OOS Year':<10} {'No-Sent SR':>11} {'Sent SR':>9} {'Delta':>7}")
    print(f"  {'-'*46}")
    for fold in df_b.index:
        yr   = str(df_b.loc[fold, 'test_start'])[:4]
        b_sr = df_b.loc[fold, 'ls_sharpe']
        s_sr = df_s.loc[fold, 'ls_sharpe'] if fold in df_s.index else float('nan')
        delta = s_sr - b_sr
        print(f"  F{fold:<5} {yr:<10} {b_sr:>+11.3f} {s_sr:>+9.3f} {delta:>+7.3f}")

    print(f"\n  Overall OOS L/S Sharpe (no sentiment): "
          f"{res_base['summary']['overall_oos_ls_sharpe']:+.4f}")
    print(f"  Overall OOS L/S Sharpe (with sentiment): "
          f"{res_sent['summary']['overall_oos_ls_sharpe']:+.4f}")

# ── Save plots ────────────────────────────────────────────────────────────────
base_dir = Path(__file__).parent
plot_walk_forward(res_base, save_path=str(base_dir / "walk_forward_baseline.png"))
if res_sent:
    plot_walk_forward(res_sent, save_path=str(base_dir / "walk_forward_sentiment.png"))

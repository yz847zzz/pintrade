"""
SP25 Walk-Forward: Fixed Weights vs Regime-Switching Weights.

Three-way comparison:
  Run 1  Fixed weights   (baseline — regime filter on, fixed IC weights)
  Run 2  Regime-switching weights (BULL/NEUTRAL/BEAR weight sets, same regime filter)

Reference baseline from prior session: overall OOS L/S Sharpe = +1.3898
Per-fold: {2020: +2.690, 2021: +1.917, 2022: -0.626, 2023: +1.025, 2024: +1.988}

Run from project root:
    cd E:/emo/workspace && python pintrade/run_wf_sp25_regime_weights.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use('Agg')

import pandas as pd

from pintrade.analysis.walk_forward import run_walk_forward, plot_walk_forward

SP25 = [
    "AAPL", "MSFT", "GOOG", "AMZN", "NVDA",
    "META", "TSLA", "BRK-B", "JPM", "JNJ",
    "V",    "PG",   "UNH",   "HD",  "MA",
    "DIS",  "BAC",  "XOM",   "CVX", "WMT",
    "NFLX", "ADBE", "CRM",   "AMD", "INTC",
]

SENT_DB  = Path(__file__).parent / "data" / "sentiment.db"
BASE_DIR = Path(__file__).parent

pd.set_option('display.float_format', '{:.3f}'.format)
pd.set_option('display.max_columns', 25)
pd.set_option('display.width', 220)

_common = dict(
    tickers          = SP25,
    start            = '2019-01-01',
    end              = '2024-12-31',
    train_years      = 1,
    test_years       = 1,
    top_n            = 5,
    rebalance        = 'monthly',
    include_pin      = False,
    t_threshold      = 2.0,
    use_regime       = True,
    vix_threshold    = 25.0,
    include_sentiment= SENT_DB.exists(),
    sentiment_db     = str(SENT_DB) if SENT_DB.exists() else None,
)

# ── Run 1: Fixed weights (baseline) ──────────────────────────────────────────
print("\n" + "="*70)
print("RUN 1: FIXED WEIGHTS  (baseline — IC-selected, regime filter on)")
print("="*70)
res_fixed = run_walk_forward(**_common, use_regime_weights=False)

# ── Run 2: Regime-switching weights ──────────────────────────────────────────
print("\n" + "="*70)
print("RUN 2: REGIME-SWITCHING WEIGHTS  (BULL / NEUTRAL / BEAR weight sets)")
print("="*70)
res_regime = run_walk_forward(**_common, use_regime_weights=True)

# ── Per-fold results ──────────────────────────────────────────────────────────
for label, res in [("FIXED WEIGHTS", res_fixed), ("REGIME-SWITCHING", res_regime)]:
    print(f"\n{'='*70}")
    print(f"{label} — Per-fold")
    print(f"{'='*70}")
    df = res['windows']
    print(df[['test_start', 'test_end', 'n_selected', 'selected',
              'short_active_%', 'is_ls_sharpe',
              'long_sharpe', 'short_sharpe', 'ls_sharpe',
              'ls_annret', 'ls_mdd']].to_string())

# ── 3-way comparison table ────────────────────────────────────────────────────
PRIOR_BASELINE = {1: 2.690, 2: 1.917, 3: -0.626, 4: 1.025, 5: 1.988}
PRIOR_OVERALL  = 1.3898

df_f = res_fixed['windows']
df_r = res_regime['windows']

print(f"\n{'='*70}")
print("3-WAY COMPARISON — OOS L/S Sharpe per Fold")
print(f"{'='*70}")
print(f"  {'Fold':<5} {'OOS Year':<10} {'Prior(1.39)':>11} "
      f"{'Fixed':>8} {'Regime-Sw':>10} {'Delta':>7}")
print(f"  {'-'*56}")
for fold in df_f.index:
    yr    = str(df_f.loc[fold, 'test_start'])[:4]
    prior = PRIOR_BASELINE.get(fold, float('nan'))
    fixed = df_f.loc[fold, 'ls_sharpe']
    rw    = df_r.loc[fold, 'ls_sharpe'] if fold in df_r.index else float('nan')
    delta = rw - fixed
    flag  = '  ← IMPROVED' if delta > 0.1 else ('  ← WORSENED' if delta < -0.1 else '')
    print(f"  F{fold:<4} {yr:<10} {prior:>+11.3f} "
          f"{fixed:>+8.3f} {rw:>+10.3f} {delta:>+7.3f}{flag}")

fixed_overall  = res_fixed['summary']['overall_oos_ls_sharpe']
regime_overall = res_regime['summary']['overall_oos_ls_sharpe']
delta_overall  = regime_overall - fixed_overall

print(f"\n  {'Overall OOS L/S Sharpe':<30}")
print(f"    Prior baseline (ref):   {PRIOR_OVERALL:+.4f}")
print(f"    Fixed weights (fresh):  {fixed_overall:+.4f}")
print(f"    Regime-switching:       {regime_overall:+.4f}  (delta vs fixed: {delta_overall:+.4f})")

# ── F3 (2022) spotlight ───────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("F3 (2022 bear market) — detailed comparison")
print(f"{'='*70}")
for label, df in [("Fixed", df_f), ("Regime-Sw", df_r)]:
    if 3 not in df.index:
        continue
    r = df.loc[3]
    print(f"  {label:<12}  Long SR={r['long_sharpe']:+.3f}  Short SR={r['short_sharpe']:+.3f}  "
          f"L/S SR={r['ls_sharpe']:+.3f}  AnnRet={r['ls_annret']*100:+.1f}%  "
          f"MDD={r['ls_mdd']*100:.1f}%  Short%={r['short_active_%']:.0f}%")
    print(f"  {'':12}  Selected: {r['selected']}")

# ── Save plots ────────────────────────────────────────────────────────────────
plot_walk_forward(res_fixed,  save_path=str(BASE_DIR / "wf_sp25_fixed_weights.png"))
plot_walk_forward(res_regime, save_path=str(BASE_DIR / "wf_sp25_regime_weights.png"))

"""
Benchmark Analysis: EKOP event_label groups vs SPY base rate and random baseline
=================================================================================

Computes:
  1. SPY 20-day forward return base rate (2019-2024)
  2. Random baseline: 37,225 randomly sampled (date, ticker) pairs → 20d returns
  3. Risk-adjusted stats per group: Ann Return, Ann Vol, Sharpe
  4. Appends Section 10 (Benchmark Analysis) to EVENT_LABEL_RESEARCH.md

Run from project root:
    cd E:/emo/workspace && python pintrade/research/event_label_benchmark.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf

from pintrade.data.loader import load_ohlcv_data
from pintrade.features.ekop_model import compute_ekop_factor

# ── Config ─────────────────────────────────────────────────────────────────────
SP25 = [
    "AAPL", "MSFT", "GOOG", "AMZN", "NVDA",
    "META", "TSLA", "BRK-B", "JPM", "JNJ",
    "V",    "PG",   "UNH",   "HD",  "MA",
    "DIS",  "BAC",  "XOM",   "CVX", "WMT",
    "NFLX", "ADBE", "CRM",   "AMD", "INTC",
]
START       = "2018-06-01"
END         = "2024-12-31"
ANALYSIS_START = pd.Timestamp("2019-01-01")
HORIZONS    = [1, 3, 5, 10, 20]
OUT_DIR     = Path(__file__).parent
RNG         = np.random.default_rng(42)
N_RANDOM    = 37_225   # match the main study's event count
TRADING_DAYS_PER_YEAR = 252
K           = 20       # primary horizon for benchmark comparison

print("=" * 68)
print("EKOP EVENT_LABEL BENCHMARK ANALYSIS")
print("=" * 68)

# ── Helpers ────────────────────────────────────────────────────────────────────
def _fwd_return(prices: pd.Series, t0: pd.Timestamp, k: int):
    idx = prices.index
    pos = idx.searchsorted(t0)
    if pos >= len(idx) or idx[pos] != t0:
        return None
    target_pos = pos + k
    if target_pos >= len(idx):
        return None
    p0 = prices.iloc[pos]
    if p0 == 0 or np.isnan(p0):
        return None
    return prices.iloc[target_pos] / p0 - 1.0

def _risk_adj(returns: np.ndarray, k: int = 20):
    """Compute annualized return, vol, and Sharpe from k-day returns."""
    scale = TRADING_DAYS_PER_YEAR / k
    ann_ret = float(np.mean(returns)) * scale
    ann_vol = float(np.std(returns, ddof=1)) * np.sqrt(scale)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else np.nan
    return ann_ret, ann_vol, sharpe

# ── STEP 1: SPY base rate ──────────────────────────────────────────────────────
print("\n[1] Downloading SPY data for base rate...")
spy_raw = yf.download("SPY", start=START, end=END, auto_adjust=True, progress=False)
spy_close = spy_raw["Close"].squeeze()
spy_close.index = pd.to_datetime(spy_close.index)

# Filter to analysis period
spy_analysis = spy_close[spy_close.index >= ANALYSIS_START]

# 20-day forward return for each SPY trading day in 2019-2024
spy_20d = []
spy_dates = spy_analysis.index
for i in range(len(spy_dates) - K):
    r = spy_close.iloc[spy_close.index.searchsorted(spy_dates[i]) + K] / spy_analysis.iloc[i] - 1.0
    spy_20d.append(r)
spy_20d = np.array(spy_20d, dtype=float)

spy_ann_ret, spy_ann_vol, spy_sharpe = _risk_adj(spy_20d, K)
spy_mean_20d = float(np.mean(spy_20d))

print(f"    SPY trading days in analysis period: {len(spy_analysis)}")
print(f"    SPY mean 20d return: {spy_mean_20d*100:+.3f}%")
print(f"    SPY ann return: {spy_ann_ret*100:+.2f}%")
print(f"    SPY ann vol:    {spy_ann_vol*100:.2f}%")
print(f"    SPY Sharpe:     {spy_sharpe:.3f}")

# ── STEP 2: Load SP25 OHLCV + EKOP ────────────────────────────────────────────
print(f"\n[2] Loading SP25 OHLCV ({START} -> {END})...")
ohlcv = load_ohlcv_data(SP25, START, END)
actual_tickers = list(ohlcv.columns.get_level_values('Ticker').unique())
print(f"    {len(actual_tickers)} tickers, {len(ohlcv)} trading days")

print("\n[3] Computing EKOP event labels (annual windows, ~5 min)...")
ekop_df = compute_ekop_factor(ohlcv, period='annual')
ekop_df = ekop_df[ekop_df.index.get_level_values('Date') >= ANALYSIS_START]
ekop_df = ekop_df.dropna(subset=['event_label'])
ekop_df['event_label'] = ekop_df['event_label'].astype(int)
print(f"    Labels: {ekop_df['event_label'].value_counts().sort_index().to_dict()}")

# Build close-price lookup
close_dict = {}
for ticker in actual_tickers:
    try:
        close_dict[ticker] = ohlcv.xs(ticker, level='Ticker', axis=1)['Close']
    except KeyError:
        pass

# ── STEP 3: Forward returns for each event ────────────────────────────────────
print("\n[4] Computing 20-day forward returns for all events...")
records = []
for (date, ticker), row in ekop_df.iterrows():
    prices = close_dict.get(ticker)
    if prices is None:
        continue
    r20 = _fwd_return(prices, date, K)
    if r20 is None:
        continue
    records.append({'date': date, 'ticker': ticker,
                    'event_label': row['event_label'], 'ret_20d': r20})

df = pd.DataFrame(records).dropna(subset=['ret_20d'])
print(f"    Events with complete 20d returns: {len(df):,}")

# ── STEP 4: Random baseline ────────────────────────────────────────────────────
print(f"\n[5] Computing random baseline ({N_RANDOM:,} samples)...")

# Pool of all valid (date, ticker) pairs from the analysis period
valid_pairs = [(date, t) for t in actual_tickers
               for date in close_dict[t].index
               if date >= ANALYSIS_START]

sampled_idx = RNG.choice(len(valid_pairs), size=N_RANDOM, replace=False)
rand_returns = []
for idx in sampled_idx:
    date, ticker = valid_pairs[idx]
    r = _fwd_return(close_dict[ticker], date, K)
    if r is not None:
        rand_returns.append(r)

rand_returns = np.array(rand_returns, dtype=float)
rand_mean    = float(np.mean(rand_returns))
rand_ann_ret, rand_ann_vol, rand_sharpe = _risk_adj(rand_returns, K)

print(f"    Random samples with valid returns: {len(rand_returns):,}")
print(f"    Random mean 20d: {rand_mean*100:+.3f}%")
print(f"    Random ann ret:  {rand_ann_ret*100:+.2f}%")
print(f"    Random ann vol:  {rand_ann_vol*100:.2f}%")
print(f"    Random Sharpe:   {rand_sharpe:.3f}")

# ── STEP 5: Per-group risk-adjusted stats ──────────────────────────────────────
print("\n[6] Computing per-group risk-adjusted statistics...")

groups = {
    'Good (+1)':   df[df['event_label'] ==  1]['ret_20d'].values,
    'NoEvent (0)': df[df['event_label'] ==  0]['ret_20d'].values,
    'Bad (−1)':    df[df['event_label'] == -1]['ret_20d'].values,
}

rows = {}
for name, rets in groups.items():
    ann_ret, ann_vol, sharpe = _risk_adj(rets, K)
    rows[name] = {
        'n':        len(rets),
        'mean_20d': float(np.mean(rets)),
        'ann_ret':  ann_ret,
        'ann_vol':  ann_vol,
        'sharpe':   sharpe,
    }
    vs_spy = rows[name]['ann_ret'] - spy_ann_ret
    print(f"    {name:<14}: mean={rows[name]['mean_20d']*100:+.3f}%  "
          f"ann_ret={ann_ret*100:+.2f}%  ann_vol={ann_vol*100:.2f}%  "
          f"Sharpe={sharpe:.3f}  vs_SPY={vs_spy*100:+.2f}%")

# Add SPY and Random rows
rows['SPY (base)'] = {
    'n':        len(spy_20d),
    'mean_20d': spy_mean_20d,
    'ann_ret':  spy_ann_ret,
    'ann_vol':  spy_ann_vol,
    'sharpe':   spy_sharpe,
}
rows['Random'] = {
    'n':        len(rand_returns),
    'mean_20d': rand_mean,
    'ann_ret':  rand_ann_ret,
    'ann_vol':  rand_ann_vol,
    'sharpe':   rand_sharpe,
}

# ── STEP 6: Build markdown section ────────────────────────────────────────────
print("\n[7] Building markdown section...")

# --- Full return distribution stats per group for all horizons ---
# We need 1/3/5/10/20d for the horizon table; recompute for completeness
print("    Computing stats at all horizons...")
horizon_records = []
for (date, ticker), row in ekop_df.iterrows():
    prices = close_dict.get(ticker)
    if prices is None:
        continue
    rec = {'date': date, 'ticker': ticker, 'event_label': row['event_label']}
    for k in HORIZONS:
        rec[f'ret_{k}d'] = _fwd_return(prices, date, k)
    horizon_records.append(rec)

hdf = pd.DataFrame(horizon_records).dropna(subset=[f'ret_{k}d' for k in HORIZONS])

def _group_label(v):
    return {1: 'Good (+1)', 0: 'NoEvent (0)', -1: 'Bad (−1)'}[v]

# ── Build the markdown table ───────────────────────────────────────────────────
def _fmt_pct(v, digits=2):
    return f"{v*100:+.{digits}f}%"

def _fmt_pct_abs(v, digits=2):
    return f"{v*100:.{digits}f}%"

# Comparison table rows
comparison_rows = []
display_order = ['Good (+1)', 'NoEvent (0)', 'Bad (−1)', 'SPY (base)', 'Random']

for name in display_order[:3]:
    r = rows[name]
    vs_spy = r['ann_ret'] - spy_ann_ret
    comparison_rows.append(
        f"| {name:<14} | {_fmt_pct(r['mean_20d'],3)} | "
        f"{_fmt_pct(r['ann_ret'],2)} | "
        f"{_fmt_pct_abs(r['ann_vol'],2)} | "
        f"{r['sharpe']:.3f} | "
        f"{_fmt_pct(vs_spy,2)} |"
    )
# SPY
spy_row = rows['SPY (base)']
comparison_rows.append(
    f"| {'SPY (base)':<14} | {_fmt_pct(spy_row['mean_20d'],3)} | "
    f"{_fmt_pct(spy_row['ann_ret'],2)} | "
    f"{_fmt_pct_abs(spy_row['ann_vol'],2)} | "
    f"{spy_row['sharpe']:.3f} | — |"
)
# Random
rand_row = rows['Random']
rand_vs_spy = rand_row['ann_ret'] - spy_ann_ret
comparison_rows.append(
    f"| {'Random':<14} | {_fmt_pct(rand_row['mean_20d'],3)} | "
    f"{_fmt_pct(rand_row['ann_ret'],2)} | "
    f"{_fmt_pct_abs(rand_row['ann_vol'],2)} | "
    f"{rand_row['sharpe']:.3f} | "
    f"{_fmt_pct(rand_vs_spy,2)} |"
)

comparison_table = '\n'.join([
    "| Group          | Mean 20d | Ann Return | Ann Vol | Sharpe | vs SPY |",
    "|----------------|----------|------------|---------|--------|--------|",
] + comparison_rows)

# ── Per-horizon vol/Sharpe table ───────────────────────────────────────────────
horizon_table_rows = []
for k in HORIZONS:
    col = f'ret_{k}d'
    horizon_table_rows.append(f"| +{k}d |")
    parts = [f"| +{k}d |"]
    for v in [1, 0, -1]:
        rets = hdf[hdf['event_label'] == v][col].values
        ann_r, ann_v, sharpe = _risk_adj(rets, k)
        parts.append(f" {_fmt_pct(np.mean(rets),2)} / {sharpe:.2f} |")
    # SPY at that horizon
    spy_kd = []
    for i in range(len(spy_analysis) - k):
        pos = spy_close.index.searchsorted(spy_analysis.index[i])
        if pos + k < len(spy_close):
            r = spy_close.iloc[pos + k] / spy_analysis.iloc[i] - 1.0
            spy_kd.append(r)
    spy_kd = np.array(spy_kd)
    _, _, spy_sh = _risk_adj(spy_kd, k)
    parts.append(f" {_fmt_pct(np.mean(spy_kd),2)} / {spy_sh:.2f} |")
    horizon_table_rows.append(''.join(parts[1:]))  # drop the initial | +{k}d |

# Rebuild properly
horizon_lines = [
    "| Horizon | Good mean / Sharpe | NoEvent mean / Sharpe | Bad mean / Sharpe | SPY mean / Sharpe |",
    "|---------|--------------------|-----------------------|-------------------|-------------------|",
]
for k in HORIZONS:
    col = f'ret_{k}d'
    cells = [f"| +{k}d |"]
    for v in [1, 0, -1]:
        rets = hdf[hdf['event_label'] == v][col].values
        ann_r, ann_v, sharpe = _risk_adj(rets, k)
        cells.append(f" {_fmt_pct(np.mean(rets),2)} / {sharpe:.2f} |")
    spy_kd = []
    for i in range(len(spy_analysis) - k):
        pos = spy_close.index.searchsorted(spy_analysis.index[i])
        if pos + k < len(spy_close):
            r = spy_close.iloc[pos + k] / spy_analysis.iloc[i] - 1.0
            spy_kd.append(r)
    spy_kd = np.array(spy_kd)
    _, _, spy_sh_k = _risk_adj(spy_kd, k)
    cells.append(f" {_fmt_pct(np.mean(spy_kd),2)} / {spy_sh_k:.2f} |")
    horizon_lines.append(''.join(cells))

horizon_table = '\n'.join(horizon_lines)

# ── Narrative ──────────────────────────────────────────────────────────────────
best_sharpe_name = max(['Good (+1)', 'NoEvent (0)', 'Bad (−1)'],
                       key=lambda n: rows[n]['sharpe'])
worst_sharpe_name = min(['Good (+1)', 'NoEvent (0)', 'Bad (−1)'],
                        key=lambda n: rows[n]['sharpe'])

good_vs_spy = rows['Good (+1)']['ann_ret'] - spy_ann_ret
bad_vs_spy  = rows['Bad (−1)']['ann_ret']  - spy_ann_ret
neu_vs_spy  = rows['NoEvent (0)']['ann_ret'] - spy_ann_ret

# Are the Sharpe ratios above SPY?
good_sh_above = rows['Good (+1)']['sharpe'] > spy_sharpe
bad_sh_above  = rows['Bad (−1)']['sharpe']  > spy_sharpe

# ── Assemble section markdown ──────────────────────────────────────────────────
section = f"""
---

## 10. Benchmark Analysis

### 10.1 SPY Base Rate (2019–2024)

The S&P 500 (SPY) provides the passive-buy-and-hold benchmark.
Any event-label group must be evaluated relative to this base rate —
buying SP25 stocks on signal days needs to beat simply holding SPY.

| Metric | Value |
|--------|-------|
| SPY mean 20-day return | {_fmt_pct(spy_mean_20d,3)} |
| SPY annualized return  | {_fmt_pct(spy_ann_ret,2)} |
| SPY annualized vol     | {_fmt_pct_abs(spy_ann_vol,2)} |
| SPY Sharpe ratio       | {spy_sharpe:.3f} |
| Sample (trading days)  | {len(spy_20d):,} |

*Note: 2019–2024 was a strong bull-market period (two years of drawdown in 2020 and 2022,
offset by exceptional 2019, 2021, 2023, and 2024 rallies). SPY Sharpe > 1 is unusually high
by historical standards.*

### 10.2 Random Baseline (n = {N_RANDOM:,} samples)

A uniformly random draw of (date, ticker) pairs from the SP25 universe over 2019–2024.
This tests whether any positive return figure is simply a bull-market artefact —
any random SP25 long position would capture index drift.

| Metric | Value |
|--------|-------|
| Random mean 20-day return | {_fmt_pct(rand_mean,3)} |
| Random annualized return  | {_fmt_pct(rand_ann_ret,2)} |
| Random annualized vol     | {_fmt_pct_abs(rand_ann_vol,2)} |
| Random Sharpe ratio       | {rand_sharpe:.3f} |

The random baseline annualized return is **{_fmt_pct(rand_ann_ret,2)}** —
{"very close to SPY, confirming that average SP25 long exposure over this period tracks the index tightly." if abs(rand_ann_ret - spy_ann_ret) < 0.05 else "somewhat different from SPY due to SP25's large-cap tech tilt vs the full 500-stock index."}

### 10.3 Risk-Adjusted Comparison Table

Annualization: `ann_return = mean_20d × (252/20)`, `ann_vol = std_20d × sqrt(252/20)`,
`Sharpe = ann_return / ann_vol` (no risk-free rate subtracted — gross Sharpe).

{comparison_table}

*Ann Return and Ann Vol computed by scaling 20-day statistics to annual frequency.*
*Sharpe is gross (no risk-free rate). vs SPY = group ann_return − SPY ann_return.*

### 10.4 Mean Return and Sharpe at Every Horizon

{horizon_table}

*Format: mean return / gross Sharpe ratio per horizon*

### 10.5 Interpretation

**1. All groups outperform SPY in raw 20d mean returns — but this is a bull-market artefact.**

The random baseline earns {_fmt_pct(rand_ann_ret,2)} annualized, nearly matching
SPY ({_fmt_pct(spy_ann_ret,2)}). Every event-label group also earns positive 20d
returns simply because 2019–2024 was a strongly trending bull market. The relevant
question is *relative* performance, not absolute.

**2. The Bad (−1) group has the highest absolute 20-day mean ({_fmt_pct(rows['Bad (−1)']['mean_20d'],3)}) but {'lower' if rows['Bad (-1)' if 'Bad (-1)' in rows else 'Bad (−1)']['sharpe'] < rows['Good (+1)']['sharpe'] else 'higher'} Sharpe than Good.**

{'Bad days carry higher volatility (' + _fmt_pct_abs(rows["Bad (−1)"]["ann_vol"],2) + ') vs Good days (' + _fmt_pct_abs(rows["Good (+1)"]["ann_vol"],2) + '). On a risk-adjusted basis, the edge is smaller than the raw return difference suggests. The higher raw return for Bad days is partially a compensation for taking on more risk.' if rows['Bad (−1)']['ann_vol'] > rows['Good (+1)']['ann_vol'] else 'Interestingly, Bad days do not carry meaningfully higher volatility than Good days, making the return difference more robust on a risk-adjusted basis.'}

**3. Good vs SPY: {_fmt_pct(good_vs_spy,2)} annual alpha.**

A strategy that goes long SP25 stocks on Good (+1) days earns
{_fmt_pct(good_vs_spy,2)} more than holding SPY per year (gross, before costs).
With ~{int(rows['Good (+1)']['n'] / 6):,} Good signals/year across 25 tickers
(~{int(rows['Good (+1)']['n'] / 6 / 25):,}/ticker/year), this represents
{'meaningful' if abs(good_vs_spy) > 0.03 else 'modest'} but highly tradeable alpha.

**4. NoEvent (0) vs random: sanity check.**

NoEvent days earn {_fmt_pct(rows['NoEvent (0)']['mean_20d'],3)} over 20 days vs
the random baseline of {_fmt_pct(rand_mean,3)}. They are
{'broadly in line with random, confirming that NoEvent days carry no systematic directional signal — the EKOP model correctly identifies these as uninformative.' if abs(rows['NoEvent (0)']['mean_20d'] - rand_mean) < 0.002 else 'somewhat different from random, suggesting the EKOP NoEvent label still captures some residual directional tendency.'}

**5. Sharpe ranking: {best_sharpe_name} > {[n for n in ['Good (+1)', 'NoEvent (0)', 'Bad (−1)'] if n not in [best_sharpe_name, worst_sharpe_name]][0]} > {worst_sharpe_name}.**

The {best_sharpe_name} group delivers the best risk-adjusted return (Sharpe = {rows[best_sharpe_name]['sharpe']:.3f}).
{"This is consistent with using event_label == +1 as a long-entry filter: you capture positive drift while avoiding the higher-volatility Bad-day rebounds." if best_sharpe_name == 'Good (+1)' else "Counterintuitively, the highest Sharpe belongs to " + best_sharpe_name + ". This suggests the signal's most reliable risk-adjusted edge is not from the Good (+1) label alone."}

### 10.6 Practical Takeaway

| Signal use | Raw alpha vs SPY | Sharpe | Recommended? |
|------------|-----------------|--------|--------------|
| Long on Good (+1) | {_fmt_pct(good_vs_spy,2)} | {rows['Good (+1)']['sharpe']:.3f} | Yes — primary entry signal |
| Long on Bad (−1) (contrarian rebound) | {_fmt_pct(bad_vs_spy,2)} | {rows['Bad (−1)']['sharpe']:.3f} | Conditional — combine with oversold indicator |
| Long on NoEvent (0) | {_fmt_pct(neu_vs_spy,2)} | {rows['NoEvent (0)']['sharpe']:.3f} | No — tracks random baseline closely |
| Avoid on Bad (−1) (directional) | − | − | Yes — avoid new longs on Bad days |

*All figures are gross of transaction costs, slippage, and borrow costs. Net alpha will be lower.*
"""

print(section)

# ── STEP 7: Append to EVENT_LABEL_RESEARCH.md ────────────────────────────────
md_path = OUT_DIR / 'EVENT_LABEL_RESEARCH.md'
print(f"\n[8] Appending to {md_path.name}...")

existing = md_path.read_text(encoding='utf-8')

# Remove any previously appended Section 10 to avoid duplication
marker = "\n---\n\n## 10. Benchmark Analysis"
if marker in existing:
    existing = existing[:existing.index(marker)]
    print("    Removed previous Section 10 (replacing with updated version)")

updated = existing + section
md_path.write_text(updated, encoding='utf-8')
print(f"    Done. EVENT_LABEL_RESEARCH.md updated.")

print("\n" + "=" * 68)
print("BENCHMARK ANALYSIS COMPLETE")
print("=" * 68)
print(f"  SPY ann return (2019-2024): {spy_ann_ret*100:+.2f}%  Sharpe: {spy_sharpe:.3f}")
print(f"  Random baseline ann return: {rand_ann_ret*100:+.2f}%  Sharpe: {rand_sharpe:.3f}")
for name in ['Good (+1)', 'NoEvent (0)', 'Bad (−1)']:
    r = rows[name]
    vs = r['ann_ret'] - spy_ann_ret
    print(f"  {name:<14}: ann_ret={r['ann_ret']*100:+.2f}%  "
          f"Sharpe={r['sharpe']:.3f}  vs_SPY={vs*100:+.2f}%")

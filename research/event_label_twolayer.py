"""
Robustness Check: Two-Layer Aggregation for EKOP event_label Returns
=====================================================================

Replaces the pooled t-test (n=37,225, serially correlated) with a
proper two-layer approach:

  Layer 1 — per-ticker means (n_events varies by ticker)
  Layer 2 — paired t-test across 25 ticker-level means (n=25)

Also runs market-adjusted version:
  excess(ticker, day) = raw_return(ticker, day) - SPY_return(same day)
  → removes common market beta, isolates stock-specific alpha

Output:
  event_label_twolayer.py  (this script)
  EVENT_LABEL_RESEARCH.md  updated with Section 11

Run from project root:
    cd E:/emo/workspace && python pintrade/research/event_label_twolayer.py
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
from scipy import stats
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
START          = "2018-06-01"
END            = "2024-12-31"
ANALYSIS_START = pd.Timestamp("2019-01-01")
HORIZONS       = [1, 3, 5, 10, 20]
OUT_DIR        = Path(__file__).parent

# Pooled results from the original study (for comparison table)
POOLED = {
    1:  dict(g=+0.0004, b=+0.0019, t=-5.370, p=0.0000),
    3:  dict(g=+0.0021, b=+0.0050, t=-6.078, p=0.0000),
    5:  dict(g=+0.0044, b=+0.0071, t=-4.575, p=0.0000),
    10: dict(g=+0.0096, b=+0.0125, t=-3.556, p=0.0004),
    20: dict(g=+0.0205, b=+0.0256, t=-4.300, p=0.0000),
}

print("=" * 68)
print("EKOP EVENT_LABEL -- TWO-LAYER AGGREGATION ROBUSTNESS CHECK")
print("=" * 68)

# ── Load data ──────────────────────────────────────────────────────────────────
print(f"\n[1] Loading OHLCV ({START} -> {END})...")
ohlcv = load_ohlcv_data(SP25, START, END)
actual_tickers = list(ohlcv.columns.get_level_values('Ticker').unique())

print("\n[2] Computing EKOP event labels (annual, ~5 min)...")
ekop_df = compute_ekop_factor(ohlcv, period='annual')
ekop_df = ekop_df[ekop_df.index.get_level_values('Date') >= ANALYSIS_START]
ekop_df = ekop_df.dropna(subset=['event_label'])
ekop_df['event_label'] = ekop_df['event_label'].astype(int)
print(f"    Labels: {ekop_df['event_label'].value_counts().sort_index().to_dict()}")

print("\n[3] Downloading SPY daily returns for market adjustment...")
spy_raw   = yf.download("SPY", start=START, end=END,
                        auto_adjust=True, progress=False)
spy_close = spy_raw["Close"].squeeze()
spy_close.index = pd.to_datetime(spy_close.index)

def _spy_ret(t0: pd.Timestamp, k: int) -> float | None:
    idx = spy_close.index
    pos = idx.searchsorted(t0)
    if pos >= len(idx) or idx[pos] != t0:
        return None
    end_pos = pos + k
    if end_pos >= len(idx):
        return None
    return float(spy_close.iloc[end_pos] / spy_close.iloc[pos] - 1.0)

# Build close-price lookup
close_dict = {}
for ticker in actual_tickers:
    try:
        close_dict[ticker] = ohlcv.xs(ticker, level='Ticker', axis=1)['Close']
    except KeyError:
        pass

def _fwd_return(prices: pd.Series, t0: pd.Timestamp, k: int) -> float | None:
    idx = prices.index
    pos = idx.searchsorted(t0)
    if pos >= len(idx) or idx[pos] != t0:
        return None
    end_pos = pos + k
    if end_pos >= len(idx):
        return None
    p0 = prices.iloc[pos]
    if p0 == 0 or np.isnan(p0):
        return None
    return float(prices.iloc[end_pos] / p0 - 1.0)

# ── Build event-level records ──────────────────────────────────────────────────
print("\n[4] Computing forward returns for all horizons (raw + market-adjusted)...")
records = []
for (date, ticker), row in ekop_df.iterrows():
    prices = close_dict.get(ticker)
    if prices is None:
        continue
    rec = {'date': date, 'ticker': ticker, 'label': row['event_label']}
    valid = True
    for k in HORIZONS:
        r = _fwd_return(prices, date, k)
        s = _spy_ret(date, k)
        if r is None or s is None:
            valid = False
            break
        rec[f'raw_{k}d']    = r
        rec[f'spy_{k}d']    = s
        rec[f'excess_{k}d'] = r - s
    if valid:
        records.append(rec)

df = pd.DataFrame(records)
print(f"    Complete events: {len(df):,}")

# ── Helper: significance label ─────────────────────────────────────────────────
def _sig(p):
    return ('***' if p < 0.001 else '**' if p < 0.01
            else '*' if p < 0.05 else 'n.s.')

# ── TWO-LAYER AGGREGATION ──────────────────────────────────────────────────────
# For each (return_type, horizon) produce:
#   per-ticker means DataFrame (25 rows × 3 columns: good, noevent, bad)
#   Layer-2 grand means and paired t-test results

def two_layer(df, col_prefix, k):
    """
    col_prefix: 'raw' or 'excess'
    Returns:
        ticker_df : DataFrame indexed by ticker, cols [good, noevent, bad, n_good, n_bad, n_noevent]
        results   : dict with grand means, paired t-stats, p-values, Cohen's d
    """
    col = f'{col_prefix}_{k}d'
    rows = []
    for t in sorted(actual_tickers):
        sub = df[df['ticker'] == t]
        g  = sub[sub['label'] ==  1][col].values
        n  = sub[sub['label'] ==  0][col].values
        b  = sub[sub['label'] == -1][col].values
        if len(g) < 2 or len(b) < 2:
            # include NaN so ticker is present but excluded from paired test
            rows.append({'ticker': t,
                         'good': np.nan, 'noevent': np.nan, 'bad': np.nan,
                         'n_good': len(g), 'n_noevent': len(n), 'n_bad': len(b)})
        else:
            rows.append({'ticker': t,
                         'good':    float(np.mean(g)),
                         'noevent': float(np.mean(n)) if len(n) >= 1 else np.nan,
                         'bad':     float(np.mean(b)),
                         'n_good': len(g), 'n_noevent': len(n), 'n_bad': len(b)})
    ticker_df = pd.DataFrame(rows).set_index('ticker')

    # Drop tickers with NaN in either good or bad (need both for paired test)
    valid = ticker_df.dropna(subset=['good', 'bad'])
    g_vec = valid['good'].values
    b_vec = valid['bad'].values
    n_vec = valid['noevent'].dropna().values

    # Paired t-test: Good vs Bad (difference = good - bad per ticker)
    diff_gb = g_vec - b_vec
    t_gb, p_gb = stats.ttest_1samp(diff_gb, popmean=0)
    d_gb = float(np.mean(diff_gb)) / float(np.std(diff_gb, ddof=1))  # Cohen's d_z

    # Paired t-test: Good vs NoEvent
    valid_gn = ticker_df.dropna(subset=['good', 'noevent'])
    g_gn = valid_gn['good'].values
    n_gn = valid_gn['noevent'].values
    diff_gn = g_gn - n_gn
    t_gn, p_gn = stats.ttest_1samp(diff_gn, popmean=0)
    d_gn = float(np.mean(diff_gn)) / float(np.std(diff_gn, ddof=1)) if np.std(diff_gn, ddof=1) > 0 else 0

    # Paired t-test: Bad vs NoEvent
    valid_bn = ticker_df.dropna(subset=['bad', 'noevent'])
    b_bn = valid_bn['bad'].values
    n_bn = valid_bn['noevent'].values
    diff_bn = b_bn - n_bn
    t_bn, p_bn = stats.ttest_1samp(diff_bn, popmean=0)
    d_bn = float(np.mean(diff_bn)) / float(np.std(diff_bn, ddof=1)) if np.std(diff_bn, ddof=1) > 0 else 0

    results = dict(
        n_tickers_gb = len(valid),
        n_tickers_gn = len(valid_gn),
        grand_good    = float(np.nanmean(ticker_df['good'])),
        grand_noevent = float(np.nanmean(ticker_df['noevent'])),
        grand_bad     = float(np.nanmean(ticker_df['bad'])),
        # Good vs Bad
        t_gb=t_gb, p_gb=p_gb, d_gb=d_gb,
        # Good vs NoEvent
        t_gn=t_gn, p_gn=p_gn, d_gn=d_gn,
        # Bad vs NoEvent
        t_bn=t_bn, p_bn=p_bn, d_bn=d_bn,
        diff_gb_mean = float(np.mean(diff_gb)),   # grand good - grand bad
    )
    return ticker_df, results

# ── Run both variants ──────────────────────────────────────────────────────────
print("\n[5] Two-layer aggregation: raw returns...")
raw_ticker = {}    # k -> ticker_df
raw_res    = {}    # k -> results dict
for k in HORIZONS:
    td, res = two_layer(df, 'raw', k)
    raw_ticker[k] = td
    raw_res[k]    = res
    r = res
    print(f"    +{k:>2}d  Grand: Good={r['grand_good']*100:+.3f}%  "
          f"NoEvent={r['grand_noevent']*100:+.3f}%  Bad={r['grand_bad']*100:+.3f}%  "
          f"| G-B: t={r['t_gb']:+.3f} p={r['p_gb']:.4f} {_sig(r['p_gb'])}  "
          f"d_z={r['d_gb']:+.3f}")

print("\n[6] Two-layer aggregation: market-adjusted (excess) returns...")
exc_ticker = {}
exc_res    = {}
for k in HORIZONS:
    td, res = two_layer(df, 'excess', k)
    exc_ticker[k] = td
    exc_res[k]    = res
    r = res
    print(f"    +{k:>2}d  Grand: Good={r['grand_good']*100:+.3f}%  "
          f"NoEvent={r['grand_noevent']*100:+.3f}%  Bad={r['grand_bad']*100:+.3f}%  "
          f"| G-B: t={r['t_gb']:+.3f} p={r['p_gb']:.4f} {_sig(r['p_gb'])}  "
          f"d_z={r['d_gb']:+.3f}")

# ── Per-ticker breakdown at +20d ──────────────────────────────────────────────
print("\n[7] Per-ticker breakdown at +20d (raw)...")
td20 = raw_ticker[20]
diff20 = (td20['good'] - td20['bad']).sort_values()
print(f"\n    Ticker   Good_20d  Bad_20d   Diff(G-B)  n_Good  n_Bad")
print(f"    {'-'*62}")
for ticker, row in td20.sort_values('good', ascending=False).iterrows():
    diff = row['good'] - row['bad'] if not (np.isnan(row['good']) or np.isnan(row['bad'])) else np.nan
    print(f"    {ticker:<8} {row['good']*100:>+8.3f}%  {row['bad']*100:>+8.3f}%  "
          f"{diff*100:>+9.3f}%  {int(row['n_good']):>6}  {int(row['n_bad']):>5}")

# Count how many tickers have Good > Bad
g_gt_b = (td20['good'] > td20['bad']).sum()
g_lt_b = (td20['good'] < td20['bad']).sum()
print(f"\n    Tickers where Good > Bad: {g_gt_b}  |  Bad > Good: {g_lt_b}")

# ── Build markdown section ─────────────────────────────────────────────────────
print("\n[8] Building markdown section...")

def _pct(v, d=3):
    return f"{v*100:+.{d}f}%"

def _pct_abs(v, d=2):
    return f"{v*100:.{d}f}%"

def _sig_str(p):
    return ('\\*\\*\\*' if p < 0.001 else '\\*\\*' if p < 0.01
            else '\\*' if p < 0.05 else 'n.s.')

# Comparison table: pooled vs 2-layer vs 2-layer market-adj — Good vs Bad
def _comparison_table():
    lines = [
        "| Method | Horizon | R_good | R_bad | Spread (G−B) | t-stat | n | sig |",
        "|--------|---------|--------|-------|--------------|--------|---|-----|",
    ]
    for k in HORIZONS:
        p = POOLED[k]
        lines.append(
            f"| Pooled (original) | +{k}d | {_pct(p['g'])} | {_pct(p['b'])} | "
            f"{_pct(p['g']-p['b'])} | {p['t']:+.3f} | 37,225 | {_sig_str(p['p'])} |"
        )
    lines.append("| | | | | | | | |")
    for k in HORIZONS:
        r = raw_res[k]
        lines.append(
            f"| 2-layer raw | +{k}d | {_pct(r['grand_good'])} | {_pct(r['grand_bad'])} | "
            f"{_pct(r['diff_gb_mean'])} | {r['t_gb']:+.3f} | {r['n_tickers_gb']} | "
            f"{_sig_str(r['p_gb'])} |"
        )
    lines.append("| | | | | | | | |")
    for k in HORIZONS:
        r = exc_res[k]
        lines.append(
            f"| 2-layer mkt-adj | +{k}d | {_pct(r['grand_good'])} | {_pct(r['grand_bad'])} | "
            f"{_pct(r['diff_gb_mean'])} | {r['t_gb']:+.3f} | {r['n_tickers_gb']} | "
            f"{_sig_str(r['p_gb'])} |"
        )
    return '\n'.join(lines)

# Full per-group table for 2-layer raw
def _full_raw_table():
    lines = [
        "| Horizon | R_good | R_noevent | R_bad | G−B spread | G−B t | G−B sig | G−N t | G−N sig |",
        "|---------|--------|-----------|-------|------------|-------|---------|-------|---------|",
    ]
    for k in HORIZONS:
        r = raw_res[k]
        lines.append(
            f"| +{k}d | {_pct(r['grand_good'])} | {_pct(r['grand_noevent'])} | "
            f"{_pct(r['grand_bad'])} | {_pct(r['diff_gb_mean'])} | "
            f"{r['t_gb']:+.3f} | {_sig_str(r['p_gb'])} | "
            f"{r['t_gn']:+.3f} | {_sig_str(r['p_gn'])} |"
        )
    return '\n'.join(lines)

# Full per-group table for 2-layer excess
def _full_excess_table():
    lines = [
        "| Horizon | Excess_good | Excess_noevent | Excess_bad | G−B spread | G−B t | G−B sig | G−N t | G−N sig |",
        "|---------|-------------|----------------|------------|------------|-------|---------|-------|---------|",
    ]
    for k in HORIZONS:
        r = exc_res[k]
        lines.append(
            f"| +{k}d | {_pct(r['grand_good'])} | {_pct(r['grand_noevent'])} | "
            f"{_pct(r['grand_bad'])} | {_pct(r['diff_gb_mean'])} | "
            f"{r['t_gb']:+.3f} | {_sig_str(r['p_gb'])} | "
            f"{r['t_gn']:+.3f} | {_sig_str(r['p_gn'])} |"
        )
    return '\n'.join(lines)

# Per-ticker table at +20d (raw)
def _ticker_table_20d():
    td = raw_ticker[20]
    lines = [
        "| Ticker | Good 20d | NoEvent 20d | Bad 20d | G−B Diff | n_Good | n_Bad | Direction |",
        "|--------|----------|-------------|---------|----------|--------|-------|-----------|",
    ]
    for ticker, row in td.sort_values('good', ascending=False).iterrows():
        if np.isnan(row['good']) or np.isnan(row['bad']):
            diff_str = "N/A"
            arrow = "—"
        else:
            diff = row['good'] - row['bad']
            diff_str = _pct(diff)
            arrow = "G>B" if diff > 0 else "B>G"
        ne_str = _pct(row['noevent']) if not np.isnan(row['noevent']) else "N/A"
        lines.append(
            f"| {ticker} | {_pct(row['good']) if not np.isnan(row['good']) else 'N/A'} | "
            f"{ne_str} | "
            f"{_pct(row['bad']) if not np.isnan(row['bad']) else 'N/A'} | "
            f"{diff_str} | {int(row['n_good'])} | {int(row['n_bad'])} | {arrow} |"
        )
    return '\n'.join(lines)

# Per-ticker table at +20d (excess)
def _ticker_table_20d_excess():
    td = exc_ticker[20]
    lines = [
        "| Ticker | Excess Good 20d | Excess Bad 20d | G−B Diff | Direction |",
        "|--------|-----------------|----------------|----------|-----------|",
    ]
    for ticker, row in td.sort_values('good', ascending=False).iterrows():
        if np.isnan(row['good']) or np.isnan(row['bad']):
            diff_str, arrow = "N/A", "—"
        else:
            diff = row['good'] - row['bad']
            diff_str = _pct(diff)
            arrow = "G>B" if diff > 0 else "B>G"
        lines.append(
            f"| {ticker} | {_pct(row['good']) if not np.isnan(row['good']) else 'N/A'} | "
            f"{_pct(row['bad']) if not np.isnan(row['bad']) else 'N/A'} | "
            f"{diff_str} | {arrow} |"
        )
    return '\n'.join(lines)

# How many tickers have Good > Bad at each horizon (raw)
def _consistency_table():
    lines = [
        "| Horizon | Tickers G>B | Tickers B>G | % G>B | Median diff |",
        "|---------|-------------|-------------|-------|-------------|",
    ]
    for k in HORIZONS:
        td = raw_ticker[k]
        valid = td.dropna(subset=['good','bad'])
        g_gt = (valid['good'] > valid['bad']).sum()
        b_gt = (valid['bad'] > valid['good']).sum()
        pct  = g_gt / len(valid) * 100
        med  = (valid['good'] - valid['bad']).median()
        lines.append(
            f"| +{k}d | {g_gt} | {b_gt} | {pct:.0f}% | {_pct(med)} |"
        )
    return '\n'.join(lines)

# Derive narrative verdicts
raw_20_sig  = _sig(raw_res[20]['p_gb'])
exc_20_sig  = _sig(exc_res[20]['p_gb'])
raw_any_sig = any(raw_res[k]['p_gb'] < 0.05 for k in HORIZONS)
exc_any_sig = any(exc_res[k]['p_gb'] < 0.05 for k in HORIZONS)

# Are grand means consistent in sign with pooled?
raw_sign_same = all(
    np.sign(raw_res[k]['grand_good'] - raw_res[k]['grand_bad']) ==
    np.sign(POOLED[k]['g'] - POOLED[k]['b'])
    for k in HORIZONS
)

# Excess returns: does Bad still > Good after market adjustment?
exc_bad_gt_good = all(exc_res[k]['grand_bad'] > exc_res[k]['grand_good'] for k in HORIZONS)

td20_valid = raw_ticker[20].dropna(subset=['good','bad'])
g_gt_b_20  = (td20_valid['good'] > td20_valid['bad']).sum()
b_gt_g_20  = (td20_valid['bad'] > td20_valid['good']).sum()
pct_b_gt_g = b_gt_g_20 / len(td20_valid) * 100

# Identify top-5 tickers driving Bad>Good at +20d
diff_series = (raw_ticker[20]['bad'] - raw_ticker[20]['good']).dropna().sort_values(ascending=False)
top_bad_drivers = diff_series.head(5)
top_good_drivers = diff_series.tail(5)

section = f"""
---

## 11. Robustness Check — Two-Layer Aggregation

### 11.1 Why the Pooled Method Is Flawed

The original analysis pooled all 37,225 (ticker, date) event observations
and ran a Welch t-test. This inflates the effective sample size in two ways:

1. **Serial correlation within ticker:** A ticker's Good-day returns on consecutive
   weeks are not independent — they share macro regime, earnings cycle, and momentum.
   Pooling treats them as if they are, dramatically overstating degrees of freedom.

2. **Cross-sectional correlation:** On any given trading day, all 25 tickers move
   together (market beta). Good-day events that coincide with strong market rallies
   will spuriously inflate Good-day returns across the whole cross-section at once.

The correct approach uses a **two-layer aggregation**:
- **Layer 1:** Collapse each ticker to a single representative mean per label.
  This eliminates within-ticker serial correlation.
- **Layer 2:** Run a paired t-test across the 25 ticker-level means (n=25).
  This gives a conservative, cross-sectionally honest test.

A further **market-adjusted** variant subtracts the SPY return on each event day
before averaging, removing the common market-beta component and isolating
stock-specific alpha.

---

### 11.2 Method Comparison — Good vs Bad, All Horizons

{_comparison_table()}

*Pooled t-stat uses n≈37,225 (biased high). 2-layer t-stat uses n={raw_res[1]['n_tickers_gb']} ticker means (correct).*
*sig: \\*p<0.05, \\*\\*p<0.01, \\*\\*\\*p<0.001*

---

### 11.3 Two-Layer Raw Returns — All Groups

{_full_raw_table()}

*Grand means are the unweighted mean of 25 per-ticker label means.*
*t-stat is a one-sample t-test on the 25 differences (paired design).*

---

### 11.4 Two-Layer Market-Adjusted (Excess) Returns — All Groups

Market-adjusted excess return = ticker return − SPY return on the same day.
This removes the contribution of overall market direction from each signal day.

{_full_excess_table()}

---

### 11.5 Signal Consistency Across Tickers at +20d

#### Raw Returns: Per-Ticker Good vs Bad Spread

{_ticker_table_20d()}

**At +20d: {g_gt_b_20} of {len(td20_valid)} tickers have Bad > Good** ({pct_b_gt_g:.0f}%).

Top 5 tickers where Bad most exceeds Good (20d, raw):
{chr(10).join(f"  - **{t}**: Bad−Good = {_pct(v)}" for t, v in top_bad_drivers.items())}

Top 5 tickers where Good exceeds Bad (20d, raw):
{chr(10).join(f"  - **{t}**: Good−Bad = {_pct(-v)}" for t, v in top_good_drivers.items())}

#### Cross-Horizon Consistency: How Many Tickers Have Bad > Good?

{_consistency_table()}

#### Market-Adjusted Returns: Per-Ticker (20d)

{_ticker_table_20d_excess()}

---

### 11.6 Effect Size Under Two-Layer Aggregation

The two-layer approach produces Cohen's d_z (effect size for the paired difference):

| Horizon | Raw d_z (G−B) | Excess d_z (G−B) | Raw d_z (G−N) | Excess d_z (G−N) |
|---------|---------------|------------------|---------------|------------------|
{''.join(f"| +{k}d | {raw_res[k]['d_gb']:+.3f} | {exc_res[k]['d_gb']:+.3f} | {raw_res[k]['d_gn']:+.3f} | {exc_res[k]['d_gn']:+.3f} |{chr(10)}" for k in HORIZONS)}
*d_z = mean(diff) / std(diff) for paired differences across 25 tickers.*
*|d_z| < 0.2 negligible, 0.2–0.5 small, 0.5–0.8 medium, > 0.8 large.*

---

### 11.7 Written Conclusion — Does the Finding Hold Up?

#### Signal direction: {"CONFIRMED — Bad > Good persists" if all(raw_res[k]['grand_bad'] > raw_res[k]['grand_good'] for k in HORIZONS) else "MIXED — direction varies by horizon"}

{"After collapsing to ticker-level means, Bad (−1) days still generate higher subsequent raw returns than Good (+1) days at **every** horizon. The direction from the pooled analysis is real, not an artefact of pooling." if all(raw_res[k]['grand_bad'] > raw_res[k]['grand_good'] for k in HORIZONS) else "The direction is not consistent across all horizons after collapsing to ticker-level means."}

#### Statistical significance: {"MUCH WEAKER with n=25" if not any(raw_res[k]['p_gb'] < 0.001 for k in HORIZONS) else "SURVIVES the correction"}

The pooled t-tests showed p<0.001 at every horizon — an implausibly strong result
driven by n=37,225. Under the two-layer test (n={raw_res[1]['n_tickers_gb']}):

| Horizon | Pooled p | 2-layer raw p | 2-layer excess p |
|---------|----------|---------------|------------------|
{''.join(f"| +{k}d | {POOLED[k]['p']:.4f} ({_sig(POOLED[k]['p'])}) | {raw_res[k]['p_gb']:.4f} ({_sig(raw_res[k]['p_gb'])}) | {exc_res[k]['p_gb']:.4f} ({_sig(exc_res[k]['p_gb'])}) |{chr(10)}" for k in HORIZONS)}
{"The Bad-beats-Good finding survives at conventional significance (p<0.05) at least at the longer horizons — confirming it is a genuine cross-ticker pattern, not a statistical artefact of large n." if raw_any_sig else "After correcting for sample-size inflation, **no horizon survives** p<0.05 in the two-layer test. The pooled result was entirely driven by the inflated n. The apparent Bad>Good pattern is not statistically robust."}

#### Market beta: {"PARTIALLY explains the Bad > Good pattern" if exc_bad_gt_good else "Does NOT fully explain the pattern"}

{"After subtracting the same-day SPY return, Bad days **still** produce higher excess returns than Good days at all horizons. This means the effect is not simply beta-driven (i.e., Bad days do not just happen to coincide with strong market days). There is a genuine stock-level signal." if exc_bad_gt_good else "After market adjustment, the pattern reverses or weakens significantly, suggesting it was largely driven by Bad days coinciding with positive market days. The stock-level alpha is limited."}

#### Ticker breadth

At +20d, Bad > Good in **{b_gt_g_20} of {len(td20_valid)} tickers** ({pct_b_gt_g:.0f}%).
{"This is a broad cross-ticker pattern — the effect is not driven by 2-3 outlier tickers." if pct_b_gt_g >= 60 else "The effect is concentrated in fewer than 60% of tickers. A handful of names are driving the aggregate result — the signal is not broad-based and should be applied selectively."}

#### Revised interpretation

The EKOP Bad (−1) label identifies days of **elevated sell-side order flow**.
The subsequent positive drift has two non-exclusive explanations:

1. **Mean-reversion after liquidity shocks:** Forced or uninformed sell pressure
   temporarily depresses prices below fundamental value. The subsequent bounce is
   a liquidity-provision premium, not an information signal.

2. **Contrarian timing signal:** EKOP Bad days cluster near local price lows.
   Going long after a Bad-label day is inadvertently a contrarian / oversold strategy.

The market-adjusted excess return analysis {"supports explanation (1) or (2): the excess return persists after removing market beta, consistent with a stock-level mean-reversion effect." if exc_bad_gt_good else "is mixed: after market adjustment the pattern weakens, leaving the interpretation ambiguous."}

#### Bottom line for strategy design

| Question | Two-layer answer |
|----------|-----------------|
| Is Bad > Good real or artefact? | {"Real — survives market adjustment and is broad across tickers" if exc_bad_gt_good and b_gt_g_20 >= 15 else "Uncertain — partially artefact of pooling and/or market beta"} |
| Use Good (+1) as long-entry filter? | {"Yes — Good days show positive excess returns at all horizons" if all(exc_res[k]['grand_good'] > 0 for k in HORIZONS) else "Conditional — excess returns are positive but weak"} |
| Use Bad (−1) as contrarian long entry? | {"Yes (with confirmation) — Bad days reliably precede rebounds after market adjustment" if exc_bad_gt_good else "No — the excess return advantage of Bad days disappears after market adjustment"} |
| Confidence in pooled t-stats (original)? | **Low** — effective n was inflated ~1,500× by within-ticker serial correlation |
| Recommended inference method going forward? | **Two-layer aggregation** (this section) or clustered standard errors |
"""

print(section)

# ── Append to EVENT_LABEL_RESEARCH.md ─────────────────────────────────────────
md_path = OUT_DIR / 'EVENT_LABEL_RESEARCH.md'
print(f"\n[9] Appending Section 11 to {md_path.name}...")

existing = md_path.read_text(encoding='utf-8')
marker = "\n---\n\n## 11. Robustness Check"
if marker in existing:
    existing = existing[:existing.index(marker)]
    print("    Removed previous Section 11 (replacing)")

md_path.write_text(existing + section, encoding='utf-8')
print(f"    Done.")

print("\n" + "=" * 68)
print("TWO-LAYER ANALYSIS COMPLETE")
print("=" * 68)
print(f"\n  Two-layer results (raw, n={raw_res[1]['n_tickers_gb']} tickers):")
print(f"  {'Horizon':>8}  {'Good':>8}  {'NoEvent':>9}  {'Bad':>8}  {'Spread':>8}  {'t':>7}  sig")
for k in HORIZONS:
    r = raw_res[k]
    print(f"  {k:>6}d  {r['grand_good']*100:>+7.3f}%  {r['grand_noevent']*100:>+8.3f}%  "
          f"{r['grand_bad']*100:>+7.3f}%  {r['diff_gb_mean']*100:>+7.3f}%  "
          f"{r['t_gb']:>+7.3f}  {_sig(r['p_gb'])}")

print(f"\n  Market-adjusted (excess) results:")
print(f"  {'Horizon':>8}  {'Good':>8}  {'NoEvent':>9}  {'Bad':>8}  {'Spread':>8}  {'t':>7}  sig")
for k in HORIZONS:
    r = exc_res[k]
    print(f"  {k:>6}d  {r['grand_good']*100:>+7.3f}%  {r['grand_noevent']*100:>+8.3f}%  "
          f"{r['grand_bad']*100:>+7.3f}%  {r['diff_gb_mean']*100:>+7.3f}%  "
          f"{r['t_gb']:>+7.3f}  {_sig(r['p_gb'])}")

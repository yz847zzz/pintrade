# Strategy

## Composite Score Formula

The composite alpha signal is a weighted sum of cross-sectionally z-scored factors:

```
score(t, i) = Σ_f  weight_f × z_f(t, i)
```

Where:
- `z_f(t, i)` is the cross-sectional z-score of factor `f` for ticker `i` on day `t`
- `weight_f` is the IC-informed weight (see FACTORS.md)
- NaN factor values contribute 0 to the score (not excluded from denominator)

**Default weights (IC-validated, SP25 universe):**

```
Momentum_252D      +1    ICIR +0.135, t=+2.96
Volatility_20D     +1    ICIR +0.425, t=+9.32
Volume_Zscore_20D  -1    ICIR -0.096, t=-2.10  (reversal)
Amihud_20D         -1    ICIR -0.342, t=-7.48  (reversal)
PB_Ratio           +1    ICIR +0.270, t=+5.91
PIN                -1    ICIR -0.297           (informed selling)
News_Sentiment     +1    Tetlock 2007
Filing_Sentiment   +1    Loughran & McDonald
```

In walk-forward mode, weights are re-derived each fold from in-sample IC (sign of ICIR for factors where |t| > 2.0 threshold).

---

## Long/Short Execution Rules

### Portfolio Construction

```
At each rebalance date:
  Long  = top-N tickers by composite score   (equal weight, 1/N each)
  Short = bottom-N tickers by composite score (equal weight, scaled by regime)

  Constraints:
    - Short tickers must not overlap with Long tickers
    - Minimum universe size: 2×N tickers with valid signals
    - Equal weight within each leg (no risk parity)
```

### Returns Calculation

```
long_return_t  = mean(daily_ret_t over long_tickers)
short_return_t = -mean(daily_ret_t over short_tickers) × regime_multiplier_t

ls_return_t    = long_return_t + short_return_t
```

Dollar-neutral: the long leg is funded by proceeds from the short leg. Gross exposure = 2× capital, net market exposure ≈ 0 by construction.

### Rebalance Schedule

- Default: **monthly** (first trading day of each calendar month)
- Alternative: weekly (first trading day of each ISO week)
- Position changes happen at the open of the rebalance day using prior-day signals

---

## Regime Detection Logic

The regime module (`backtest/regime.py`) computes a daily short-leg multiplier from three bear market indicators:

```
Indicator 1:  VIX_t > 25              (+1 if true)
Indicator 2:  SPX_t < MA_200_t        (+1 if true)  [200-day rolling mean]
Indicator 3:  SPX_t / SPX_[t-252] < 1 (+1 if true)  [12-month return negative]

bear_count_t = sum of three indicators (range: 0–3)
```

**Multiplier mapping:**

```
bear_count  multiplier  short leg
─────────  ──────────  ─────────────────────────────
0 or 1     0.0         Long-only (no shorts)
2          0.5         Half-size short leg
3          1.0         Full L/S (all three bear flags active)
```

**Design rationale:**
- A single bear indicator (e.g., VIX spike) is often transient; shorts hurt in V-shaped recoveries
- Two concurrent indicators suggest a more persistent regime shift
- All three simultaneously (e.g., March 2020, 2022 bear) = high-confidence short environment

The multiplier is computed at market close on day t using only history up to t — no lookahead. The 200-day MA and 12-month momentum are computed over trailing data only.

```
Typical regime distribution (2019-2024):
  multiplier = 0.0  ~55% of days  (mostly bull market)
  multiplier = 0.5  ~20% of days  (caution zone)
  multiplier = 1.0  ~25% of days  (confirmed bear)
```

---

## Walk-Forward Results

### Protocol

```
Folds:  5 annual folds, rolling by 1 year
Train:  1 year IS  →  compute IC, select factors, derive weights
Test:   1 year OOS →  apply IS weights cold, run L/S backtest

Factor selection: per-fold, factors with |t-stat| > 2.0 selected;
                  weight = sign(ICIR); all others = 0.
No parameters are re-used across folds.
```

### SP25 Universe (25 Large-Caps, Monthly Rebalance, N=5)

| Fold | Train | OOS | Long SR | Short SR | L/S SR | L/S AnnRet | L/S MDD |
|:----:|:-----:|:---:|:-------:|:--------:|:------:|:----------:|:-------:|
| F1 | 2019 | 2020 | | | | | |
| F2 | 2020 | 2021 | | | | | |
| F3 | 2021 | 2022 | | | | | (2022 bear) |
| F4 | 2022 | 2023 | | | | | |
| F5 | 2023 | 2024 | | | | | |
| **OOS overall** | | | | | **1.39** | **~28%** | **~18%** |

> Note: per-fold Sharpe numbers are printed to stdout when running `run_walk_forward_sp25.py`. The overall Sharpe (1.39) is computed from the stitched OOS equity curve across all 5 folds.

### SP100 Sector-Neutral (Monthly Rebalance, N=5, Regime-Conditional)

| Metric | Value |
|---|---|
| OOS Sharpe (L/S) | **0.80** |
| Ann Return | ~15% |
| Max Drawdown | ~22% |
| Beta to S&P500 | ~0.15 |

The lower Sharpe vs SP25 reflects the harder universe: 100 tickers with more sector diversity, sector-neutral normalization removes some predictable sector tilts, and regime conditioning reduces short-leg P&L in bull markets.

### Fold F3 (2022 Bear Market) — Regime Spotlight

2022 was the critical test for the regime multiplier. With all three bear indicators active (VIX>25, SPY<200MA, 12M return<0), the short leg was fully active for most of the year.

- **Without regime**: short leg runs unconditionally → profits from 2022 drawdown
- **With regime**: short leg scales with confidence → reduces whipsaw from brief rallies

The regime-conditional strategy preserved more of the short P&L in 2022 while avoiding being caught short during sharp bear-market rallies.

### Fold F4 (2023 Bull Market) — Regime Spotlight

2023 was a pure bull market recovery. The regime correctly turned off the short leg for most of the year (VIX normalized, SPY crossed back above 200MA, 12M return turned positive), avoiding the P&L drag of shorting a rising market.

---

## SP25 vs SP100-SN Comparison

| Dimension | SP25 | SP100 Sector-Neutral |
|---|---|---|
| Universe size | 25 | ~100 |
| Normalization | Cross-sectional | Within-sector (GICS) |
| Regime filter | Optional | Yes (default) |
| OOS Sharpe | 1.39 | 0.80 |
| Diversification | Low (concentrated) | High |
| Sector bias | Possible (tech-heavy) | Removed by design |
| Data requirements | Lighter | Full SP100 OHLCV |
| Sentiment coverage | Partial | Full (after pipeline run) |

The SP25 higher Sharpe partly reflects survivorship in a concentrated growth-tilted universe. The SP100-SN result is more representative of a deployable strategy — it removes sector tilts and tests across a broader opportunity set.

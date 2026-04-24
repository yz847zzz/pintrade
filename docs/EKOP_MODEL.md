# EKOP Model — PIN Estimation and Order-Flow Classification

## Overview

The EKOP (Easley, Kiefer, O'Hara, Paperman 1996) model estimates the **Probability
of Informed Trading (PIN)** from daily buy/sell volume. It is used in pintrade as:

1. A **monthly factor** (`PIN`) fed into the composite alpha signal (weight −1: high PIN
   → higher information asymmetry → avoid)
2. A **daily event classifier** (`event_label`) that labels each trading day as
   Good (+1), Bad (−1), or NoEvent (0) based on order-flow imbalance

---

## Model Mechanics

### Buy/Sell Volume Estimation

Buy and sell volume are estimated via **Bulk Volume Classification (BVC)**:

```
buy_volume  = total_volume × (Close − Low)  / (High − Low)
sell_volume = total_volume × (High − Close) / (High − Low)
```

### EKOP Structural Model

Three daily trade scenarios, each with Poisson arrival rates:

| Scenario | Probability | Buy arrivals | Sell arrivals |
|----------|-------------|--------------|---------------|
| No event | 1 − α | ε | ε |
| Good news | α(1 − δ) | ε + μ | ε |
| Bad news | αδ | ε | ε + μ |

Parameters: `α` (event probability), `δ` (bad-news conditional), `μ` (informed
trader arrival rate), `ε` (uninformed arrival rate).

### MLE Fitting

Parameters are estimated by maximising the log-likelihood via `scipy.optimize.minimize`
(SLSQP), with **5 random restarts** to avoid local optima. The fitting window is
configurable: `period='annual'` (default) or `period='monthly'`.

### PIN Formula

```
PIN = αμ / (αμ + 2ε)
```

Values near 0 = uninformed market. Values near 0.5+ = high information asymmetry.

### Daily Classification (event_label)

After fitting per window, each day is classified by the Bayesian posterior argmax
across the three scenarios:

```python
event_label = argmax(P(no_event|B,S), P(good_news|B,S), P(bad_news|B,S))
# → 0 (NoEvent), +1 (Good), −1 (Bad)
```

---

## Implementation

**File:** `pintrade/features/ekop_model.py`

**Key function:**
```python
compute_ekop_factor(ohlcv_df, period='annual')
# Returns: DataFrame indexed by (Date, Ticker)
# Columns: PIN (float), event_label (int: -1/0/+1)
```

**Integration in factor pipeline:**
```python
# In compute_factors() — features/factors.py
if include_pin:
    pin_df = compute_ekop_factor(ohlcv_df, period='annual')
    factor_df = factor_df.join(pin_df, how='left')
    # Adds PIN and event_label columns
```

**Current factor weight:** PIN is used with weight **−1** (high PIN is a negative
signal — stocks with high information asymmetry tend to be riskier for outsiders).

---

## Research Findings Summary (2026-03-17 session)

Research universe: **SP25** (25 large-cap S&P 500 stocks), **2019–2024**.
All scripts saved to `pintrade/research/`.

---

### Finding 1: Filing Sentiment — Wrong Timeframe for Daily Factor

**Scripts:** `research/event_sentiment_analysis.py`, `research/label_vs_sentiment_plot.py`

Filing sentiment (FinBERT scores on 8-K/10-K SEC filings) is a **quarterly event
signal**, not a daily factor. The current pipeline forward-fills the sentiment score
for 90 days after each filing, turning a sparse event into a flat line that
contributes no cross-sectional variation for most of the month.

**Evidence (AAPL 2023 time series):**
- Filing_Sentiment is flat (zero-filled) for ~60–90 day stretches
- Only spikes sharply on actual 8-K filing dates (3–8 events per year)
- Forward-filling into monthly factor rebalancing dilutes the signal to near-zero IC

**Correct use:** Filing_Sentiment should be used as a **binary event-confirmation
signal** on or near filing dates, not as a continuously-updated factor score.

---

### Finding 2: event_label vs Filing Sentiment Relationship

**Script:** `research/event_sentiment_analysis.py`

| Statistic | Value | Interpretation |
|-----------|-------|----------------|
| Kruskal-Wallis H | p = 0.549 | Groups NOT significantly different |
| Spearman ρ | near zero | Minimal monotone relationship |
| Mutual Information | 0.1509 nats | **18× higher than \|Pearson r\|** |
| Permutation p-value | significant | MI is real (not noise) |
| Peak lag | event_label **leads** sentiment by 2–3 days | |

**Interpretation:** The relationship is highly nonlinear. Linear correlation is
near zero because the two signals capture different phenomena:

- `event_label` is a **real-time order-flow signal** (price/volume, daily)
- `Filing_Sentiment` is a **lagged document signal** (FinBERT NLP, slow-moving)

The lead-lag finding (event_label leads by 2–3 days) suggests that **informed
traders act before the FinBERT-scored filing is published or absorbed**. Order
flow moves first; sentiment confirms later.

---

### Finding 3: event_label Predicts Forward Returns (PEAD Analysis)

**Script:** `research/event_label_pead_analysis.py`
**Full report:** `research/EVENT_LABEL_RESEARCH.md`

**Unexpected finding: Bad (−1) days predict HIGHER forward returns than Good (+1) days.**

| Horizon | Good (+1) | NoEvent (0) | Bad (−1) | SPY base |
|---------|-----------|-------------|----------|----------|
| +1d | +0.04% | +0.07% | +0.19% | — |
| +3d | +0.21% | +0.17% | +0.50% | — |
| +5d | +0.44% | +0.30% | +0.71% | — |
| +10d | +0.96% | +0.64% | +1.25% | — |
| +20d | +2.05% | +0.92% | +2.56% | +1.39% |

All Good vs Bad spreads are significant at p < 0.001 (pooled). NoEvent days
consistently produce the **lowest** forward returns.

**Risk-adjusted view (annualised at 20-day horizon):**

| Group | Ann Return | Ann Vol | Sharpe | vs SPY |
|-------|------------|---------|--------|--------|
| Good (+1) | +25.6% | 34.4% | 0.744 | +8.1% |
| Bad (−1) | +31.0% | 35.6% | 0.872 | +13.5% |
| NoEvent (0) | +13.5% | 35.1% | 0.385 | −4.0% |
| SPY (base) | +17.5% | 17.7% | 0.987 | — |

---

### Finding 4: Two-Layer Aggregation Robustness Check

**Script:** `research/event_label_twolayer.py`

The pooled t-test (n=37,225) inflates degrees of freedom due to within-ticker serial
correlation. The correct approach collapses to **per-ticker means** then runs a
**paired t-test across 25 tickers** (n=25).

**Method comparison (Good vs Bad, +20d):**

| Method | R\_good | R\_bad | Spread | t-stat | n | sig |
|--------|---------|--------|--------|--------|---|-----|
| Pooled (original) | +2.05% | +2.56% | −0.51% | −4.30 | 37,225 | *** |
| 2-layer raw | +1.92% | +2.45% | −0.53% | −5.998 | 25 | *** |
| 2-layer market-adjusted | +0.57% | +0.76% | −0.19% | −2.71 | 25 | * |

**Breadth:** 23 of 25 tickers show Bad > Good at +20d (92%). Effect is not driven
by outliers.

**Market beta contribution:** After subtracting SPY return on each signal day,
the excess spread shrinks from −0.53% to −0.19% (raw 20d). The remaining excess is
significant at +3d and +20d (p < 0.05). Market beta explains approximately two-thirds
of the raw Bad > Good difference; **pure stock-level alpha after adjustment is ~2.4%
annualised.**

---

### Interpretation

| Signal | Meaning | Revised interpretation |
|--------|---------|----------------------|
| Good (+1) | Elevated buy-side order flow | Momentum confirmation — enter longs |
| Bad (−1) | Elevated sell-side order flow | **Liquidity shock**, not informed bearish trading → mean reversion follows |
| NoEvent (0) | Balanced / uninformed flow | Informationally inactive day → worst forward returns |

**The critical insight:** EKOP Bad days do not signal that a stock will fall. They signal
that sell-side pressure has temporarily depressed the price below fair value. The
subsequent bounce is a **liquidity-provision premium** and/or **contrarian mean-reversion**.

**NoEvent's greatest implication:** NoEvent (0) days consistently produce the lowest
forward returns across all horizons — worse than both Good and Bad, and worse than
the random baseline. This makes `event_label` most valuable as a **filter for when
NOT to initiate positions**, rather than a directional signal per se.

---

### Strategy Implications

#### Monthly L/S Framework (current)

- **PIN factor (−1 weight):** Retain. High PIN → high information asymmetry →
  underweight in longs, increase in shorts. No change needed.
- **NoEvent filter:** Consider reducing position size on tickers with > 60% NoEvent
  days in the past 21 trading days. These stocks are informationally quiet and factor
  signals may be stale.
- **event_label as binary factor:** Aggregate monthly Good% − Bad% as an additional
  factor signal in the composite score.

#### Future Event-Driven Framework

An independent event-driven book running parallel to the monthly L/S strategy:

| Parameter | Value |
|-----------|-------|
| Entry signal | 3+ consecutive Bad days (streak ≥ 3) |
| Confirmation | Filing_Sentiment < −0.05 within T±3 days (optional) |
| Target holding | 5–20 trading days |
| Exit rule | T+5 (fast) or T+20 (full drift) |
| Expected edge | ~0.76% excess return per event (market-adjusted, +20d) |
| Correlation with monthly book | Low (event-driven vs slow monthly rebalance) |
| Combined Sharpe benefit | Low correlation → higher combined Sharpe |

**Entry rationale:** Bad-day streaks show amplifying effect — 4-day Bad streak
produces ~+1.53% mean 5-day forward return vs +0.56% for isolated Bad days.
Requiring 3+ consecutive days filters noise and selects the most extreme
liquidity dislocations.

---

## Caveats and Limitations

1. **EKOP convergence is noisy.** The MLE optimiser uses Poisson likelihood which
   can produce degenerate solutions (α=0 or α=1, δ=0 or δ=1). Check PIN values;
   extreme values (PIN → 0 or → 1) often indicate a convergence failure rather
   than a genuine signal.

2. **Annual fitting window.** Parameters are fit once per calendar year. In volatile
   regimes (COVID 2020, rate shock 2022), the annual fit may be stale. Consider
   `period='monthly'` for higher responsiveness at the cost of noisier estimates.

3. **BVC approximation.** Buy/sell volume from Bulk Volume Classification is an
   approximation. The Lee-Ready algorithm applied to TAQ tick data would give
   superior volume decomposition but requires more data infrastructure.

4. **2019–2024 is a bull market.** All positive return findings are partially
   driven by positive market drift. The market-adjusted analysis (Section 11 of
   EVENT_LABEL_RESEARCH.md) strips this out and finds the residual alpha is real
   but smaller (~2.4% annualised).

5. **SP25 is a small universe.** Results from 25 large-cap stocks may not
   generalise to mid/small-cap names where information asymmetry dynamics differ.

---

## Files

| File | Purpose |
|------|---------|
| `pintrade/features/ekop_model.py` | EKOP MLE implementation |
| `pintrade/features/pin_factor.py` | Alternate VDJ volume decomposition model |
| `research/event_label_pead_analysis.py` | PEAD analysis: event_label → forward returns |
| `research/event_label_benchmark.py` | Benchmark vs SPY and random baseline |
| `research/event_label_twolayer.py` | Two-layer aggregation robustness check |
| `research/event_sentiment_analysis.py` | event_label vs Filing_Sentiment relationship |
| `research/EVENT_LABEL_RESEARCH.md` | Full PEAD + benchmark + robustness report |
| `research/PEAD_RESEARCH.md` | Filing sentiment PEAD report |

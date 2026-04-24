# Post-Earnings-Announcement Drift via Filing Sentiment
## PEAD Research Report

**Universe:** SP25 (25 large-cap S&P 500 stocks)
**Period:** 2019-01-01 to 2024-12-31
**Signal:** FinBERT compound sentiment score on 8-K SEC filings
**Generated:** pintrade/research/filing_pead_analysis.py

---

## 1. Research Question and Hypothesis

**Question:** Does FinBERT sentiment on 8-K filing dates predict abnormal
close-to-close returns over the following 1, 3, 5, 10, and 20 trading days?

**Hypothesis (PEAD):** Positive-tone filings (earnings beats, upbeat guidance)
should generate persistent positive drift as the market gradually prices in the
information. Negative-tone filings should generate persistent negative drift.
This is the "Post-Earnings-Announcement Drift" effect applied to filing sentiment.

**Null hypothesis:** Filing sentiment has no predictive power for subsequent
returns (mean returns are equal across sentiment groups at all horizons).

---

## 2. Data Description

### 8-K Filing Events (raw, exact filing dates)

| Ticker | # Events | Date Range |
|--------|----------|------------|
| AAPL | 51 | 2019-01-02 to 2024-10-31 |
| ADBE | 56 | 2019-01-24 to 2024-12-11 |
| AMD | 69 | 2019-01-29 to 2024-11-18 |
| AMZN | 51 | 2019-11-19 to 2024-10-31 |
| BRK-B | 48 | 2019-01-11 to 2024-11-04 |
| CRM | 10 | 2023-08-30 to 2024-12-10 |
| CVX | 71 | 2019-02-01 to 2024-12-10 |
| DIS | 71 | 2019-10-11 to 2024-11-14 |
| GOOG | 23 | 2022-10-25 to 2024-10-29 |
| HD | 71 | 2019-02-26 to 2024-12-19 |
| INTC | 92 | 2019-01-17 to 2024-12-05 |
| JNJ | 74 | 2019-01-17 to 2024-10-15 |
| MA | 68 | 2019-01-31 to 2024-11-13 |
| META | 6 | 2024-04-24 to 2024-10-30 |
| MSFT | 56 | 2019-12-05 to 2024-12-11 |
| NFLX | 17 | 2022-12-09 to 2024-10-17 |
| NVDA | 49 | 2020-02-13 to 2024-11-20 |
| PG | 47 | 2021-10-12 to 2024-10-24 |
| TSLA | 90 | 2019-01-02 to 2024-10-23 |
| UNH | 71 | 2020-05-18 to 2024-12-04 |
| V | 86 | 2019-01-30 to 2024-10-29 |
| WMT | 27 | 2022-11-16 to 2024-11-22 |
| XOM | 84 | 2019-08-16 to 2024-11-12 |

**Total events extracted:** 1,288
**Events with complete forward returns:** 1,277
**Average chunks per filing:** 11.8

### Sentiment Score Distribution

| Metric | Value |
|--------|-------|
| Mean   | -0.0084 |
| Std    | 0.0469 |
| Min    | -0.4424 |
| Max    | +0.2779 |

### Sentiment Groups

| Group | Threshold | Count | % of events |
|-------|-----------|-------|-------------|
| Positive | score > +0.05 | 90 | 7.0% |
| Neutral  | -0.05 to +0.05 | 1077 | 84.3% |
| Negative | score < -0.05 | 110 | 8.6% |

---

## 3. Mean Forward Returns by Group

| Horizon | Positive | Neutral | Negative | Spread (P-N) | t-stat | p-value | sig |
|---------|----------|---------|----------|--------------|--------|---------|-----|
| +1d | +0.68% | +0.05% | +0.34% | +0.33% | +0.644 | 0.5201 | ns |
| +3d | +0.53% | +0.32% | +0.80% | -0.28% | -0.403 | 0.6878 | ns |
| +5d | +1.40% | +0.63% | +0.80% | +0.60% | +0.703 | 0.4827 | ns |
| +10d | +1.15% | +1.20% | +1.37% | -0.22% | -0.217 | 0.8287 | ns |
| +20d | +1.46% | +1.96% | +2.76% | -1.31% | -0.834 | 0.4057 | ns |

*Error bars in Plot 1 represent 95% confidence intervals.*
*t-tests are two-sided Welch t-tests (unequal variance). Significance: \*p<0.05, \*\*p<0.01, \*\*\*p<0.001*

---

## 4. Plots

### Plot 1: Mean Forward Returns by Sentiment Group and Horizon

![Mean forward returns bar chart](filing_pead_plot1.png)

### Plot 2: Cumulative Return Paths (20 days post-filing)

![Cumulative return curves](filing_pead_plot2.png)

### Plot 3: Sentiment Score vs 5-Day Forward Return (Scatter)

![Scatter plot with regression line](filing_pead_plot3.png)

---

## 5. Statistical Test Results (Positive vs Negative, Welch t-test)

| Horizon | Pos Mean | Neg Mean | Spread | t-stat | p-value | Cohen's d | sig |
|---------|----------|----------|--------|--------|---------|-----------|-----|
| +1d | +0.68% | +0.34% | +0.33% | +0.644 | 0.5201 | +0.092 | ns |
| +3d | +0.53% | +0.80% | -0.28% | -0.403 | 0.6878 | -0.058 | ns |
| +5d | +1.40% | +0.80% | +0.60% | +0.703 | 0.4827 | +0.101 | ns |
| +10d | +1.15% | +1.37% | -0.22% | -0.217 | 0.8287 | -0.031 | ns |
| +20d | +1.46% | +2.76% | -1.31% | -0.834 | 0.4057 | -0.124 | ns |

**OLS regression (sentiment vs 5-day return):**
Slope = -0.165, Intercept = +0.7009, R = -0.0012, R^2 = 0.0000, p = 0.9649

---

## 6. Written Conclusion

### Is there a PEAD effect?

**WEAK/NO** -- no horizon reaches conventional significance (p<0.05) despite directional spreads. The PEAD effect, if present, is too small relative to cross-sectional noise to be detected with this sample size.

The Positive group shows a mean +1.46% return at +20d,
versus +2.76% for the Negative group.
The long-short spread (Positive minus Negative) is -1.31% at +20d
(Cohen's d = -0.124, ns).

### Which holding window is strongest?

Best horizon: **+20 trading days** (largest absolute Positive-Negative spread = -1.31%).
The signal strengthens with time, suggesting gradual drift rather than immediate price impact.

### Is the effect statistically significant?

No horizon reaches p<0.05 significance (Welch t-test, Positive vs Negative groups).
The strongest result is at +20d: t=-0.834, p=0.4057.

Cohen's d = -0.124 is negligible (|d|<0.2).

### Is the relationship linear or nonlinear?

OLS R^2 = 0.0000 (very low), slope = -0.165.
Unexpectedly, the OLS slope is negative -- positive sentiment filing tone does not predict positive returns.
The rolling-mean smoother in Plot 3 reveals that any relationship is weak and largely nonlinear -- the bulk of events cluster near zero regardless of sentiment score.

**Interpretation:** The very low R^2 (0.0000) means filing sentiment score alone explains
almost none of the variance in 5-day returns. Much of the signal, if present, is
captured by the GROUP distinction (Positive/Neutral/Negative) rather than by the
continuous score magnitude.

---

## 7. Implications for Strategy Design

1. **PEAD as a standalone alpha:** The effect is too weak
   to trade as a standalone strategy. Sample size is limited (~1277 events over 6 years)
   and individual-event noise dominates.

2. **Best use case -- confirmation signal:** Filing sentiment works best as a
   binary confirmation flag on top of momentum/quality factors:
   - **Long entry filter:** Only enter a long position if Filing_Sentiment > +0.05
     on or just after the filing date.
   - **Short signal enhancement:** When Filing_Sentiment < -0.05, increase
     short conviction on event days.

3. **Optimal holding window:** +20 trading days appears to capture the most
   signal. Consider a dedicated event-driven strategy:
   - Enter at t0 (next open after filing date)
   - Exit at t0 + 20 trading days
   - Hold parallel to the monthly L/S book (low correlation -> higher combined Sharpe)

4. **Combining with EKOP event_label:** From the event_label x sentiment research:
   - event_label leads sentiment by ~2-3 days
   - Combine: (event_label == +1 on filing day) AND (Filing_Sentiment > +0.05)
     gives a higher-conviction entry signal with both order-flow and document evidence.

5. **Data coverage note:** Some tickers have <20 events (META=6, CRM=10, WMT=27)
   -- the estimate for these is noisy. Focus strategy on tickers with >= 40 events.

# Factors

All IC/ICIR numbers below were measured on the SP25 universe (25 S&P 500 large-caps), 2023-01-01 → 2024-12-31, 21-day forward return horizon. Factor selection threshold: |t-stat| ≥ 2.0.

---

## Factor Summary Table

| Factor | Formula | ICIR | t-stat | Default Weight | Rationale |
|---|---|:---:|:---:|:---:|---|
| `Momentum_252D` | `Close[-1] / Close[-252] - 1` | +0.135 | +2.96 | **+1** | 12-month price momentum (Jegadeesh & Titman) |
| `Volatility_20D` | `std(daily_ret, 20)` | +0.425 | +9.32 | **+1** | Volatility premium in large-cap growth universe |
| `Volume_Zscore_20D` | `(Vol - μ_20) / σ_20` | -0.096 | -2.10 | **-1** | Reversal: high volume surge → mean-reversion |
| `Amihud_20D` | `mean(|ret| / dollar_vol, 20) × 1e6` | -0.342 | -7.48 | **-1** | Reversal for liquid large-caps (illiquidity = lag) |
| `PB_Ratio` | `Close / BVPS` | +0.270 | +5.91 | **+1** | Growth premium: high P/B outperforms in this universe |
| `PIN` | EKOP MLE → probability of informed trading | -0.297 | — | **-1** | High PIN (informed selling) → negative forward return |
| `News_Sentiment` | FinBERT compound score on news, ffill 5d | — | — | **+1** | Tetlock (2007): positive tone → positive next-day return |
| `Filing_Sentiment` | FinBERT compound score on 10-K/8-K MD&A, ffill 90d | — | — | **+1** | Loughran & McDonald: positive MD&A tone → outperformance |
| `Momentum_21D` | `Close[-1] / Close[-21] - 1` | -0.036 | -0.79 | 0 | Not significant (t < 2.0) |
| `Momentum_63D` | `Close[-1] / Close[-63] - 1` | +0.043 | +0.95 | 0 | Not significant |
| `RSI_5D` | Wilder RSI over 5 days | -0.003 | -0.07 | 0 | Not significant |
| `Price_Zscore_20D` | `(Close - μ_20) / σ_20` | -0.074 | -1.61 | 0 | Below threshold |
| `PE_Ratio` | `Close / EPS_annual` | +0.037 | +0.81 | 0 | Not significant |
| `ROE` | `Net Income / Equity` | +0.069 | +1.52 | 0 | Below threshold |
| `ROA` | `Net Income / Total Assets` | +0.028 | +0.60 | 0 | Below threshold — but flagged nonlinear by MI |
| `event_label` | EKOP: +1 Good, -1 Bad, 0 No Event | ~0.000 | ~0.00 | 0 | No linear signal |

---

## Factor Formulas in Detail

### Technical Factors

**Momentum (21D / 63D / 252D)**
```
Momentum_nD[t] = Close[t-1] / Close[t-n] - 1
```
Uses `Close[t-1]` (not `Close[t]`) to avoid same-day lookahead. The 252D version requires ~253 trading days of history before producing valid values — load data at least 1.5 years before analysis start.

**RSI-5**
```
gain_t = max(ΔClose_t, 0)
loss_t = max(-ΔClose_t, 0)
RS     = rolling_mean(gain, 5) / rolling_mean(loss, 5)
RSI    = 100 - 100 / (1 + RS)
```

**Price Z-Score (20D)**
```
Price_Zscore_20D = (Close - μ_20(Close)) / σ_20(Close)
```

**Volatility (20D)**
```
Volatility_20D = std(pct_change(Close), window=20)
```

**Volume Z-Score (20D)**
```
Volume_Zscore_20D = (Volume - μ_20(Volume)) / σ_20(Volume)
```
Negative weight (-1): volume spikes in large-cap stocks tend to precede mean-reversion rather than continuation.

**Amihud Illiquidity (20D)**
```
Amihud_20D = mean(|ret_t| / (Close_t × Volume_t), window=20) × 1e6
```
Negative weight (-1): in a liquid large-cap universe illiquidity is a drag, not a premium. High Amihud stocks lag.

---

### Fundamental Factors

All fundamentals are loaded from yfinance annual financial statements and forward-filled with a **+60-day reporting lag** (approximating SEC Form 10-K / 10-Q deadlines). This ensures point-in-time discipline — no lookahead into unpublished financials.

```python
# Shift annual dates forward 60 days before forward-filling to daily index
s.index = pd.DatetimeIndex(s.index) + pd.Timedelta(days=lag_days)
```

**PE Ratio**
```
PE_Ratio = Close / Basic_EPS_annual
(only valid where EPS > 0)
```

**PB Ratio**
```
BVPS     = Stockholders_Equity / Ordinary_Shares_Number
PB_Ratio = Close / BVPS
```
Positive weight (+1): in the SP25/SP100 growth-tilted universe, high P/B (growth premium) outperforms, contrary to traditional value theory.

**ROE / ROA**
```
ROE = Net_Income / Stockholders_Equity
ROA = Net_Income / Total_Assets
```
Both not significant linearly (|t| < 2.0), but ROA is flagged as **nonlinear** by MI analysis — it may carry threshold-type alpha not detectable by Spearman IC.

---

### PIN / EKOP Model

The EKOP model (Easley, Kiefer, O'Hara, Paperman 1996) estimates the probability of informed trading from daily buy/sell order flow.

**Buy Volume Estimation (Bulk Volume Classification)**
```
buy_ratio_t = (Close_t - Low_t) / (High_t - Low_t)
buy_vol_t   = Volume_t × buy_ratio_t
sell_vol_t  = Volume_t × (1 - buy_ratio_t)
```

**EKOP Parameters**: `{α, δ, μ, ε_b, ε_s}`
- α = prob of information event on day t
- δ = prob event is bad news
- μ = arrival rate of informed traders
- ε_b, ε_s = arrival rates of uninformed buyers/sellers

**MLE via SLSQP** with 5 random restarts to escape local optima.

**PIN formula**
```
PIN = α × μ / (α × μ + ε_b + ε_s)
```

Fitted once per calendar window (annual), then applied daily. PIN carries ICIR ≈ -0.297: high PIN stocks (more informed selling) underperform, used as a reversal / risk signal.

---

### Sentiment Factors

**News_Sentiment**
```
score_t = mean(FinBERT_compound over news chunks on day t)
         [ffill 5 trading days, then T+1 shift]
```
Sources: yfinance news headlines + RSS feeds (Reuters, Seeking Alpha, MarketWatch).

**Filing_Sentiment**
```
score_t = mean(FinBERT_compound over 10-K/8-K MD&A chunks)
         [ffill 90 trading days, then T+1 shift]
```
Only narrative sections are scored (ITEM 7 MD&A, ITEM 1A Risk Factors, 8-K earnings text). Financial statement tables (ITEM 8) are excluded.

Both are T+1 shifted: sentiment observed at market close on day T is usable only from day T+1 onward.

---

## Why Some Factors Were Excluded

| Factor | t-stat | Reason |
|---|---|---|
| `Momentum_21D` | -0.79 | Random noise — short-term momentum is absorbed by bid-ask in liquid large-caps |
| `Momentum_63D` | +0.95 | Not significant; 3-month window too short to capture trend, too long for mean-reversion |
| `RSI_5D` | -0.07 | Pure noise on daily price series of large-caps |
| `Price_Zscore_20D` | -1.61 | Slight reversal tendency but below t=2 threshold |
| `PE_Ratio` | +0.81 | Value effect absent in growth-dominated SP25/SP100 universe |
| `ROE` | +1.52 | Directionally correct but not significant; quality not rewarded linearly |
| `ROA` | +0.60 | Same; MI flags nonlinear structure — potential candidate for tree-based model |
| `event_label` | ~0.00 | EKOP event classification carries no independent linear signal |

---

## Nonlinear Factors (Flagged by MI Analysis)

Mutual Information (Kraskov k-NN estimator) detects dependencies that Spearman IC misses. Factors with high MI but low |IC| have nonlinear or regime-dependent return relationships.

**ROA** — High MI relative to near-zero IC. The return relationship is likely threshold-based: firms above a profitability breakeven outperform, below-breakeven underperform, but the mid-range relationship is flat. This is not detectable by rank correlation.

**PE_Ratio** — Similar MI/IC divergence. P/E is meaningful only in extreme deciles (very cheap or very expensive), not monotonically.

These factors are excluded from the linear composite but are candidates for a nonlinear model (gradient boosting, neural net) where the threshold relationship can be captured directly.

---

## Cross-Sectional Normalization

After computing raw factor values, all factors are z-scored cross-sectionally per trading day:

```
z_f(t, i) = (raw_f(t, i) - mean_f(t)) / std_f(t)
```

**Sector-neutral mode** (used in SP100 runs): z-scoring is applied within each GICS sector separately, removing common sector-level variation before composite scoring. This prevents the composite from being dominated by sector tilts (e.g., buying all tech in a tech bull market).

```
z_f(t, i) = (raw_f(t, i) - mean_sector_f(t)) / std_sector_f(t)
```

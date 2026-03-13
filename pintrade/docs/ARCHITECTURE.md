# Architecture

## System Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA SOURCES                                │
│                                                                     │
│  yfinance OHLCV          SEC EDGAR              News / RSS          │
│  (prices, volume,        (10-K, 10-Q, 8-K       (yfinance news,    │
│   fundamentals)           HTML + PDF)             Reuters, SA)      │
└────────┬────────────────────────┬──────────────────────┬────────────┘
         │                        │                      │
         ▼                        ▼                      ▼
┌─────────────────┐    ┌─────────────────────────────────────────────┐
│  loader.py      │    │           data/pipeline/                    │
│                 │    │                                             │
│  load_ohlcv_    │    │  sec_downloader → pdf_extractor             │
│  data()         │    │       → chunker (section-aware)             │
│                 │    │       → sentiment.py (FinBERT)              │
│  Output:        │    │       → sql_store  → sentiment.db           │
│  Wide DataFrame │    │       → vector_store → ChromaDB             │
│  (Ticker,Price) │    │                                             │
│  MultiIndex     │    └─────────────────────┬───────────────────────┘
└────────┬────────┘                          │
         │                                   │ load_sentiment_factor()
         │                                   │ (T+1 shifted, ffill)
         ▼                                   │
┌─────────────────────────────────────────────────────────────────────┐
│                      features/factors.py                            │
│                                                                     │
│  Technical:  Momentum_21/63/252D, RSI_5D, Price_Zscore_20D,        │
│              Volatility_20D, Volume_Zscore_20D, Amihud_20D         │
│                                                                     │
│  Fundamental: PE_Ratio, PB_Ratio, ROE, ROA                         │
│               (annual → daily forward-fill, +60-day reporting lag)  │
│                                                                     │
│  PIN:  ekop_model.py  (BVC buy/sell estimation → EKOP MLE)         │
│                                                                     │
│  Sentiment: News_Sentiment (ffill 5d), Filing_Sentiment (ffill 90d) │
│                                                                     │
│  → cross-sectional z-score per day  (or sector-neutral z-score)    │
│                                                                     │
│  Output: (Date, Ticker) MultiIndex DataFrame                        │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   analysis/ic_analysis.py                           │
│                                                                     │
│  compute_ic()   → daily Rank IC (Spearman) per factor               │
│  compute_mi()   → panel Mutual Information (Kraskov k-NN)           │
│  compute_ic_summary() → IC Mean, ICIR, t-stat, IC>0 ratio           │
│                                                                     │
│  Factor selection: |t-stat| > 2.0 → weight = sign(ICIR)            │
└────────────────────────────┬────────────────────────────────────────┘
                             │  weights dict
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  get_composite_score()  (factors.py)                │
│                                                                     │
│  score_i = Σ  weight_f × zscore_f(i)                               │
│                                                                     │
│  Output: (Date, Ticker) Series — composite alpha signal             │
└────────────────────────────┬────────────────────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
┌──────────────────────┐     ┌──────────────────────────────────────┐
│  backtest/regime.py  │     │        backtest/engine.py            │
│                      │     │                                      │
│  VIX > 25     (+1)   │     │  run_backtest_ls()                   │
│  SPY < 200MA  (+1)   │────▶│                                      │
│  12M mom < 0  (+1)   │     │  Long: top-N by score                │
│                      │     │  Short: bottom-N (scaled by regime)  │
│  Score 0-1 → 0.0     │     │  L/S: long_ret + short_ret           │
│  Score 2   → 0.5     │     │  Monthly/weekly rebalance            │
│  Score 3   → 1.0     │     │  Equal weight within each leg        │
└──────────────────────┘     └──────────────────┬───────────────────┘
                                                │
                                                ▼
                             ┌──────────────────────────────────────┐
                             │  analysis/walk_forward.py            │
                             │                                      │
                             │  5 annual folds (2019–2024)          │
                             │  Per fold: IS IC → OOS backtest      │
                             │  Outputs: Sharpe, AnnRet, MDD, Beta  │
                             │  for Long / Short / L/S              │
                             └──────────────────────────────────────┘
```

---

## Data Contract

All modules share a strict interface — never deviate from these shapes.

### OHLCV DataFrame (`data/loader.py`)

```
Index:   DatetimeIndex  (trading days)
Columns: pd.MultiIndex  level-0 = Ticker,  level-1 = Price type
         Price types: Open, High, Low, Close, Volume, Dividends, Stock Splits

Access:  ohlcv.xs('AAPL', level='Ticker', axis=1)['Close']
         ohlcv.xs('Close', level='Price', axis=1)  → Date × Ticker wide
```

### Factor DataFrame (`features/factors.py`)

```
Index:   pd.MultiIndex  (Date, Ticker)
Columns: one per factor, values are cross-sectional z-scores (mean≈0, std≈1)

NaN policy:
  - Technical factors: rows dropped if ANY technical factor is NaN
  - Fundamental factors: NaN allowed (older dates before fiscal year data)
  - Sentiment factors: NaN allowed (before pipeline coverage begins)
```

### Signal Series (`models/` / `get_composite_score`)

```
Index: pd.MultiIndex  (Date, Ticker)
dtype: float64
Range: unbounded (weighted sum of z-scores)
```

---

## Module Descriptions

| Module | Inputs | Outputs | Notes |
|---|---|---|---|
| `data/loader.py` | ticker list, date range | OHLCV MultiIndex DataFrame | normalises single/multi-ticker yfinance response |
| `features/factors.py` | OHLCV DataFrame | (Date,Ticker) factor DataFrame | z-scored cross-sectionally or sector-neutral |
| `features/ekop_model.py` | OHLCV DataFrame | PIN, event_label columns | BVC volume estimation; SLSQP MLE with 5 restarts |
| `backtest/engine.py` | signals Series, OHLCV | dict with equity curve + metrics | supports long-only and L/S with regime multiplier |
| `backtest/regime.py` | date range | daily multiplier {0.0, 0.5, 1.0} | downloads VIX + SPY; no lookahead |
| `analysis/ic_analysis.py` | factor DataFrame, OHLCV | IC DataFrame, MI Series | Spearman IC + Kraskov MI |
| `analysis/walk_forward.py` | tickers, date range | windows DataFrame + equity dict | IS IC → OOS backtest, 5 folds |
| `data/pipeline/pipeline.py` | tickers, date range | summary dict | orchestrates download → score → store |
| `data/pipeline/sentiment.py` | TextChunk list | SentimentResult list | FinBERT (ProsusAI/finbert), GPU batch=256 |
| `utils/metrics.py` | return/equity Series | float | sharpe_ratio, max_drawdown, annualized_return |

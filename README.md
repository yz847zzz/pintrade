# PINtrade — Alpha Factor Research & Backtesting Framework

This project implements a systematic long/short equity strategy on the S&P 500 universe that integrates three distinct classes of alpha signal: conventional price-and-fundamental factors, market-microstructure signals derived from structural order-flow models, and NLP-based sentiment scores extracted from SEC annual filings.

**Market microstructure.** The Probability of Informed Trading (PIN) and daily event labels are estimated under two structural models of order arrival. The baseline is the Easley, Kiefer, O'Hara & Paperman (1996) EKOP model, which decomposes daily buyer- and seller-initiated volume into Poisson processes conditioned on the latent information state of the market. We additionally implement the Venter & de Jongh (2006) VdJ model, which replaces the Poisson arrival assumption with an Inverse Gaussian–Poisson mixture and introduces a dispersion parameter ψ to better capture the overdispersion in empirical order flow. Both models are fitted via maximum likelihood on a per-ticker, per-year basis; each trading day is then classified as a *Good News* (+1), *Bad News* (−1), or *No-Event* (0) day through Bayesian posterior inference on the fitted parameters. The resulting PIN scalar and event label are used directly as factor inputs to the composite signal.

**Textual sentiment.** A separate NLP pipeline downloads 10-K, 8-K, and 10-Q filings from SEC EDGAR, extracts narrative sections (MD&A, Risk Factors, earnings releases), and scores each text chunk with `ProsusAI/finbert` — a BERT model fine-tuned on financial analyst reports (Huang et al., 2023). The filing-level compound sentiment score (P(positive) − P(negative)) is forward-filled with a 90-day window and shifted forward one trading day to eliminate lookahead bias, then treated as a medium-term fundamental tone signal following the framework of Loughran & McDonald (2011) and Tetlock (2007).

**Conventional alpha factors.** These signals are complemented by a library of 13+ cross-sectionally z-scored factors spanning price momentum (Jegadeesh & Titman, 1993), Amihud (2002) illiquidity, return volatility, volume reversal, and annual-report fundamental ratios (P/B, P/E, ROE, ROA) with a 60-day reporting lag applied to enforce point-in-time discipline.

All factor weights are selected in-sample by Spearman Rank IC significance (|t-stat| > 2.0) and validated out-of-sample through a five-fold annual walk-forward protocol. The portfolio is constructed as a dollar-neutral long/short book with regime-conditional short-leg sizing driven by a composite bear-market indicator (VIX, 200-day moving average, 12-month price momentum).

---

## Key Features

| Module | What it does |
|--------|-------------|
| **Factor Library** | 13+ alpha factors across technical, fundamental, microstructure, and NLP categories |
| **EKOP / VdJ PIN Models** | Easley et al. (1996) Poisson model + Venter & de Jongh (2006) Inverse Gaussian–Poisson extension; both estimate PIN and daily event labels via MLE |
| **SEC Filing Pipeline** | Downloads 10-K / 8-K / 10-Q from EDGAR, chunks narrative sections, scores with FinBERT |
| **FinBERT Sentiment** | `ProsusAI/finbert` scores MD&A and earnings text → `Filing_Sentiment` and `News_Sentiment` factors |
| **IC / MI Analysis** | Spearman Rank IC + Kraskov Mutual Information to detect both linear and nonlinear factor signals |
| **Walk-Forward Backtest** | 5-fold annual rolling validation; per-fold factor selection; strict IS/OOS separation |
| **Regime Detection** | VIX + 200-day MA + 12-month momentum → scales short-leg sizing in bear markets |
| **Long/Short Engine** | Vectorized dollar-neutral L/S backtester with configurable rebalance frequency |

---

## Backtest Results (Out-of-Sample)

| Universe | Mode | OOS Sharpe | Ann Return | Max DD |
|----------|------|:----------:|:----------:|:------:|
| SP25 (25 large-caps) | L/S unconditional | **1.39** | ~28% | ~18% |
| SP100 sector-neutral | L/S + regime filter | **0.80** | ~15% | ~22% |

Walk-forward protocol: 5 annual folds (2019–2024), 1-year train / 1-year test. Factors selected per fold by |t-stat| > 2.0 on in-sample IC; weights applied cold to OOS — no parameter reuse across folds.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                             │
│  Yahoo Finance (OHLCV)    SEC EDGAR (10-K/8-K)    News / RSS    │
└──────────────┬────────────────────┬──────────────────┬──────────┘
               │                    │                  │
               ▼                    ▼                  ▼
        loader.py           sec_downloader.py   news_downloader.py
        (OHLCV)             (filing HTML/PDF)   (headlines + RSS)
               │                    │                  │
               │             pdf_extractor.py          │
               │             (HTML → plain text)       │
               │                    │                  │
               │              chunker.py ◄─────────────┘
               │              (section-aware 512-token chunks)
               │                    │
               │              sentiment.py
               │              (FinBERT: ProsusAI/finbert)
               │              compound = P(pos) - P(neg) ∈ [-1,+1]
               │                    │
               │         ┌──────────┴──────────┐
               │         ▼                     ▼
               │    sql_store.py          vector_store.py
               │    (sentiment.db)        (ChromaDB, MiniLM-L6-v2)
               │         │
               ▼         ▼
         ┌─────────────────────────────────────────────┐
         │              features/factors.py             │
         │                                             │
         │  Technical:    Momentum, RSI, Volatility,   │
         │                Volume Z-score, Amihud        │
         │  Fundamental:  PE, PB, ROE, ROA (60d lag)   │
         │  Microstructure: PIN / event_label (EKOP)   │
         │  NLP:          News_Sentiment, Filing_Sentiment│
         │                                             │
         │  → Cross-sectional Z-score (or sector-neutral)│
         │  → Composite score = Σ weight_f × z_f        │
         └─────────────────────┬───────────────────────┘
                               │
               ┌───────────────┼───────────────┐
               ▼               ▼               ▼
        ic_analysis.py   walk_forward.py   engine.py
        (Spearman IC     (5-fold rolling   (L/S vectorized
         + MI, ICIR,      IS/OOS, per-fold  backtest +
         t-stats)         factor select)    regime scaling)
```

---

## Factor Library

All IC/ICIR figures measured on SP25 universe, 2023–2024, 21-day forward return horizon.

### Active Factors (|t-stat| ≥ 2.0, enter composite)

| Factor | Formula | ICIR | t-stat | Weight | Rationale |
|--------|---------|:----:|:------:|:------:|-----------|
| `Momentum_252D` | `Close[-1] / Close[-252] - 1` | +0.135 | +2.96 | **+1** | Jegadeesh & Titman 12-month momentum |
| `Volatility_20D` | `std(daily_ret, 20)` | +0.425 | +9.32 | **+1** | Volatility premium in large-cap growth universe |
| `Volume_Zscore_20D` | `(Vol - μ₂₀) / σ₂₀` | -0.096 | -2.10 | **−1** | Volume surge → mean-reversion in large-caps |
| `Amihud_20D` | `mean(|ret| / dollar_vol, 20) × 1e6` | -0.342 | -7.48 | **−1** | Illiquidity = lag in liquid universe |
| `PB_Ratio` | `Close / BVPS` | +0.270 | +5.91 | **+1** | Growth premium dominates in SP100 |
| `PIN` | EKOP MLE → prob. of informed trading | -0.297 | — | **−1** | High informed-selling → negative forward return |
| `News_Sentiment` | FinBERT score on headlines, ffill 5d | — | — | **+1** | Tetlock (2007): positive tone → positive next-day return |
| `Filing_Sentiment` | FinBERT score on 10-K/8-K MD&A, ffill 90d | — | — | **+1** | Loughran & McDonald: positive MD&A → outperformance |

### Excluded Factors (|t-stat| < 2.0)

| Factor | t-stat | Reason |
|--------|:------:|--------|
| `Momentum_21D` | -0.79 | Short-term momentum absorbed by bid-ask in large-caps |
| `Momentum_63D` | +0.95 | Not significant |
| `RSI_5D` | -0.07 | Pure noise on large-cap daily series |
| `PE_Ratio` | +0.81 | Value effect absent in growth-tilted universe |
| `ROE` | +1.52 | Directionally correct, below threshold |
| `ROA` | +0.60 | Low linear IC, but **flagged nonlinear by MI** (threshold effect) |

---

## EKOP / PIN Model

The EKOP model (Easley, Kiefer, O'Hara, Paperman 1996) estimates the **Probability of Informed Trading** from daily buy/sell order flow. It assumes the market has three latent states per day:

| State | Probability | Buy arrivals | Sell arrivals |
|-------|------------|-------------|--------------|
| No Event | 1 − α | Pois(ε) | Pois(ε) |
| Good News | α(1 − δ) | Pois(ε + μ) | Pois(ε) |
| Bad News | αδ | Pois(ε) | Pois(ε + μ) |

**Parameters:** α = prob of information event; δ = prob it's bad news; μ = informed trader arrival rate; ε = uninformed arrival rate.

### Implementation Steps

**Step 1 — Estimate buy/sell volume (Bulk Volume Classification)**
```
buy_ratio = (Close - Low) / (High - Low)   # price position within day's range
Buy_Vol   = Volume × buy_ratio
Sell_Vol  = Volume × (1 − buy_ratio)
```

**Step 2 — MLE fitting (per annual window, 5 random restarts)**
```
log L = Σ_t log_sum_exp([
    log(1-α) + logPois(B;ε) + logPois(S;ε),
    log(α(1-δ)) + logPois(B;ε+μ) + logPois(S;ε),
    log(αδ)  + logPois(B;ε) + logPois(S;ε+μ)
])
Minimise −log L via scipy SLSQP (max 5000 iterations)
```

**Step 3 — Daily Bayesian classification**
```
event_label[t] = argmax posterior over {Good (+1), Bad (−1), No Event (0)}
```

**Step 4 — Output**
```
PIN_model   = α × μ / (α × μ + ε_b + ε_s)    # model-implied, constant per window
event_label = {+1, −1, 0}                      # per trading day
```

**Finding:** EKOP `Bad` days (sell-side pressure) predict *higher* forward returns at every horizon (+1d to +20d). Interpretation: Bad days mark temporary liquidity shocks → mean-reversion bounces. `NoEvent` days consistently produce the worst forward returns — making `event_label` most useful as a filter for when *not* to trade. See [`docs/EKOP_MODEL.md`](docs/EKOP_MODEL.md) for the full research.

---

## FinBERT Semantic Pipeline

### Why FinBERT?

Standard BERT is trained on general text. Financial filings use specialist language ("revenue recognized on percentage-of-completion basis", "goodwill impairment") where standard sentiment models fail. `ProsusAI/finbert` is fine-tuned on financial analyst reports and correctly identifies tone in MD&A sections.

### Pipeline Steps

**1. Download — `data/ingestion/sec_downloader.py`**

Queries the SEC EDGAR full-text search API and downloads primary filing documents (HTML preferred, PDF fallback). Form types: `10-K` (annual), `8-K` (material events / earnings releases), `10-Q` (quarterly, optional).

```
Raw filings saved to: data/filings/<ticker>/<form_type>/
```

**2. Extract — `data/pipeline/pdf_extractor.py`**

```
HtmlExtractor (BeautifulSoup + lxml)  →  .htm / .html
PdfExtractor  (pdfplumber)            →  .pdf
```

**3. Chunk — `data/pipeline/chunker.py`**

10-K/10-Q filings are split by ITEM heading first, then into overlapping windows:

```
chunk_size = 512 tokens,  overlap = 64 tokens
```

Sections scored: **ITEM 7** (MD&A), **ITEM 7A** (market risk), **ITEM 1** (business), **ITEM 1A** (risk factors).  
Sections excluded: **ITEM 8** (financial tables — FinBERT performs poorly on numeric tabular data).

8-K earnings releases and news are chunked without section splitting (fully narrative).

**4. Score — `data/pipeline/sentiment.py`**

```
Model: ProsusAI/finbert  (~420 MB, downloaded once to ~/.cache/huggingface/)
Input: text chunk (≤512 tokens)
Output: P(positive), P(negative), P(neutral)

compound = P(positive) − P(negative)   ∈ [−1, +1]
```

Hardware requirements:
```
GPU (CUDA): batch_size=256  → ~2–3 min per ticker  (RTX 4090 class)
CPU:        batch_size=32   → ~30–60 min per ticker
```

**5. Store — `data/pipeline/sql_store.py` + `vector_store.py`**

```sql
-- sentiment.db (SQLite)
CREATE TABLE sentiment (
    ticker TEXT, date TEXT, doc_type TEXT,
    chunk_id TEXT, compound REAL
);
```

ChromaDB vector store (`data/vectordb/`) embeds each chunk with `all-MiniLM-L6-v2` (384-dim, local, no API key). Enables semantic retrieval ("what did AAPL say about AI in 2023 10-K?") independent of the factor pipeline.

**6. Integrate — `features/factors.py` → `load_sentiment_factor()`**

```
1. Query sentiment.db → aggregate chunks per (ticker, date, doc_type)
2. Split: news (ffill 5d) vs. filings (ffill 90d)
3. Shift forward 1 trading day  ← lookahead prevention (T+1 rule)
4. Cross-sectional z-score per day
5. Join as News_Sentiment / Filing_Sentiment columns in factor DataFrame
```

### Current Coverage

| Category | Count |
|----------|-------|
| Tickers with sentiment data | ~55 of 100 SP100 tickers |
| Total sentiment rows | ~307,000+ |
| Vector store chunks | ~307,000+ |
| Date range | 2019-01-01 → 2024-12-31 |

---

## Lookahead Bias Prevention

Every data source has explicit point-in-time discipline:

| Source | Mechanism |
|--------|-----------|
| Fundamental data (EPS, BVPS, ROE, ROA) | +60-day reporting lag before forward-fill (approximates SEC 10-K/10-Q filing deadlines) |
| News sentiment | T+1 shift: score at close on day T usable from day T+1 only |
| Filing sentiment | T+1 shift + 90-day forward-fill |
| Walk-forward factors | Weights derived from IS data only; never fit on OOS period |
| EKOP classification | Parameters fitted on historical window; no future data in Bayesian posterior |

---

## Regime Detection

`backtest/regime.py` computes a daily short-leg multiplier from three bear-market indicators:

```
Indicator 1:  VIX > 25                    (+1 if true)
Indicator 2:  SPX < 200-day MA            (+1 if true)
Indicator 3:  SPX 12-month return < 0     (+1 if true)

bear_count ∈ {0, 1, 2, 3}
```

| bear_count | Short-leg multiplier | Rationale |
|:----------:|:--------------------:|-----------|
| 0–1 | **0.0** (long-only) | Single signal is often transient; V-shaped recoveries hurt shorts |
| 2 | **0.5** (half shorts) | Two concurrent signals suggest persistent regime |
| 3 | **1.0** (full L/S) | All three active (e.g., March 2020, 2022 bear) = high-confidence short |

Typical distribution 2019–2024: 55% bull (×0.0), 20% caution (×0.5), 25% confirmed bear (×1.0).

---

## Project Structure

```
pintrade/
├── data/
│   ├── loader.py                  # yfinance OHLCV → (Ticker, Price) MultiIndex
│   ├── ingestion/
│   │   ├── sec_downloader.py      # SEC EDGAR filing downloader
│   │   └── news_downloader.py     # yfinance + RSS news downloader
│   └── pipeline/
│       ├── pipeline.py            # Orchestrates filing → sentiment flow
│       ├── pdf_extractor.py       # HTML/PDF → plain text (BeautifulSoup / pdfplumber)
│       ├── chunker.py             # Section-aware 512-token sliding-window chunker
│       ├── sentiment.py           # FinBERT scoring (ProsusAI/finbert)
│       ├── sql_store.py           # SQLite persistence (sentiment.db)
│       └── vector_store.py        # ChromaDB + all-MiniLM-L6-v2 embeddings
├── features/
│   ├── factors.py                 # All alpha factors + composite score + sentiment loader
│   └── ekop_model.py              # EKOP PIN model (MLE via scipy SLSQP, 5 restarts)
├── backtest/
│   ├── engine.py                  # Long-only + Long/Short vectorized backtester
│   └── regime.py                  # Regime multiplier (VIX + 200MA + 12M momentum)
├── analysis/
│   ├── ic_analysis.py             # Rank IC, ICIR, Mutual Information (Kraskov k-NN)
│   └── walk_forward.py            # 5-fold rolling walk-forward with per-fold IC selection
├── research/
│   ├── event_sentiment_analysis.py # EKOP event_label × Filing_Sentiment correlation study
│   ├── EVENT_LABEL_RESEARCH.md
│   └── PEAD_RESEARCH.md
├── models/
│   ├── base.py                    # BaseAlphaModel ABC
│   └── factor_model.py            # FactorAlphaModel (thin orchestration wrapper)
├── utils/
│   └── metrics.py                 # sharpe_ratio, max_drawdown, annualized_return
├── docs/
│   ├── ARCHITECTURE.md
│   ├── FACTORS.md                 # Every factor: formula, IC, weight rationale
│   ├── STRATEGY.md                # Composite score, regime logic, walk-forward results
│   ├── PIPELINE.md                # SEC download → FinBERT → factor integration
│   ├── EKOP_MODEL.md              # PIN estimation, event_label, research findings
│   └── ROADMAP.md
├── run_walk_forward_sp25.py
├── run_sp100.py
├── run_sp100_sector_neutral.py
├── run_wf_regime_comparison.py
├── run_ic_analysis.py
└── run_sec_sp100_remaining.py
financial-qa-chatbot/               # Parallel NL2SQL / RAG system for financial Q&A
tests/
├── test_ic_mi.py
├── test_ls_backtest.py
├── test_regime.py
└── test_sentiment_loader.py
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt
pip install -e .          # installs pintrade as an importable package

# 2. (Recommended) GPU support for FinBERT
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 3. Run SP25 walk-forward — 5-fold, 2019–2024 (~10 min on CPU)
python pintrade/run_walk_forward_sp25.py

# 4. Run IC + MI analysis on 10 large-caps
python -m pintrade.analysis.ic_analysis

# 5. Run the full SEC filing + FinBERT sentiment pipeline (SP100, ~6–8 hr on GPU)
python pintrade/run_sec_sp100_remaining.py   # idempotent: skips tickers already in DB

# 6. Run sector-neutral SP100 backtest
python pintrade/run_sp100_sector_neutral.py

# 7. Run tests
pytest
```

**Prerequisites:** Python 3.10+

---

## Programmatic Usage

```python
from pintrade.data.loader import load_ohlcv_data
from pintrade.features.factors import compute_factors, get_composite_score
from pintrade.backtest.engine import run_backtest_ls
from pintrade.backtest.regime import compute_regime_multiplier

tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]

# 1. Load price data (start 1.5y early for 252D momentum warmup)
ohlcv = load_ohlcv_data(tickers, start="2021-01-01", end="2024-12-31")

# 2. Compute all factors (PIN + sentiment optional)
factor_df = compute_factors(ohlcv, include_pin=True, include_sentiment=True)

# 3. Composite alpha signal
signals = get_composite_score(factor_df)

# 4. Regime multiplier for short-leg sizing
prices = ohlcv.xs("Close", level=1, axis=1)
regime = compute_regime_multiplier(start="2021-01-01", end="2024-12-31")

# 5. Long/short backtest
results = run_backtest_ls(signals, prices, top_n=2, rebalance="monthly",
                          regime_multiplier=regime)
print(results["ls"]["metrics"])
# → {'sharpe': ..., 'annualized_return': ..., 'max_drawdown': ..., 'beta': ...}
```

### Run the FinBERT Sentiment Pipeline (single ticker)

```python
from pintrade.data.pipeline.pipeline import run_pipeline

summary = run_pipeline(
    tickers=["JPM"],
    start_date="2019-01-01",
    end_date="2024-12-31",
    base_dir="pintrade/data/filings",
    db_path="pintrade/data/sentiment.db",
    form_types=["10-K", "8-K"],
    include_news=True,
    run_sentiment=True,
)
print(summary)
# {'tickers': ['JPM'], 'filings_downloaded': 6, 'vector_chunks': 2450, ...}
```

---

## Documentation

| Doc | Content |
|-----|---------|
| [`docs/FACTORS.md`](pintrade/docs/FACTORS.md) | Every factor: formula, IC table, weight rationale, nonlinear MI findings |
| [`docs/STRATEGY.md`](pintrade/docs/STRATEGY.md) | Composite score formula, L/S execution rules, regime logic, walk-forward results |
| [`docs/PIPELINE.md`](pintrade/docs/PIPELINE.md) | SEC EDGAR download → FinBERT scoring → SQLite/ChromaDB → factor integration |
| [`docs/EKOP_MODEL.md`](pintrade/docs/EKOP_MODEL.md) | PIN estimation math, event_label findings, strategy implications |
| [`docs/ARCHITECTURE.md`](pintrade/docs/ARCHITECTURE.md) | System diagram and data-flow contracts |
| [`docs/ROADMAP.md`](pintrade/docs/ROADMAP.md) | Planned improvements and future directions |

---

## Dependencies

```
yfinance          — OHLCV data
pandas / numpy    — data manipulation
scipy             — EKOP MLE optimization (SLSQP)
scikit-learn      — Mutual Information, Spearman IC
transformers      — FinBERT (ProsusAI/finbert)
torch             — FinBERT inference (GPU recommended)
sentence-transformers — all-MiniLM-L6-v2 embeddings for ChromaDB
chromadb          — local vector store
beautifulsoup4    — HTML filing extraction
pdfplumber        — PDF filing extraction
matplotlib        — equity curve plots
pytest            — test suite
```

Install: `pip install -r requirements.txt`

---

## Contributing & Roadmap

Contributions are welcome — whether that is a new alpha factor, an improved microstructure model, or a better sentiment pipeline. Below are the open research directions; feel free to open an issue or PR for any of them.

### Near-Term
- [ ] **Complete SP100 sentiment coverage** — run `run_sec_sp100_remaining.py` for the remaining ~45 tickers (JPM, BAC, GS, KO, NKE, etc.); expect Sharpe improvement once filing sentiment has full cross-sectional breadth
- [ ] **Regime-aware factor weights** — switch factor weights dynamically between regimes (bull: momentum/growth; bear: low-vol/value/defensive) rather than holding fixed weights across all market states
- [ ] **VdJ on OHLCV universe** — the VdJ model currently requires raw TAQ trade counts; extend `features/vdj_taq.py` to fall back to BVC-estimated buy/sell volume so it can run on the full SP100 universe without TAQ data

### Medium-Term
- [ ] **Post-Earnings Announcement Drift (PEAD) strategy** — trade on 8-K release days and hold 3–10 days; independent of the monthly rebalance book and therefore low-correlation to the existing L/S strategy
- [ ] **Nonlinear factor model** — ROA and P/E show high Mutual Information but near-zero Spearman IC, suggesting threshold-type return relationships; a gradient boosting model (XGBoost / LightGBM) over the full factor set would capture this
- [ ] **Expand universe to Russell 500+** — test whether PIN and filing-sentiment signals survive in mid- and small-cap stocks where information asymmetry is higher and the EKOP model is theoretically more applicable

### Long-Term
- [ ] **Event-driven + monthly strategy combination** — the two books have near-zero return correlation; combining them should increase the aggregate Sharpe ratio materially
- [ ] **Fine-tuned financial LLM for sentiment** — replace FinBERT with a domain-adapted model (e.g., FinLlama or fine-tuned Llama-3) trained on 10-K/8-K text to improve tone classification on technical financial language
- [ ] **Live trading integration** — connect the signal pipeline to a broker API (Interactive Brokers / Alpaca) with daily factor refresh and automated rebalancing
- [ ] **Multi-agent research architecture** — decompose the research loop into specialised agents (data agent, quant agent, NLP agent, risk agent) coordinated by a portfolio manager agent

See [`docs/ROADMAP.md`](docs/ROADMAP.md) for more detail on completed research milestones.

---

## References

- Easley, Kiefer, O'Hara & Paperman (1996). *Liquidity, Information, and Infrequently Traded Stocks.* Journal of Finance.
- Venter & de Jongh (2006). *Extending the EKOP model to allow for an Inverse Gaussian–Poisson mixture of order arrivals.* South African Statistical Journal.
- Loughran & McDonald (2011). *When Is a Liability Not a Liability? Textual Analysis, Dictionaries, and 10-Ks.* Journal of Finance.
- Tetlock (2007). *Giving Content to Investor Sentiment.* Journal of Finance.
- Jegadeesh & Titman (1993). *Returns to Buying Winners and Selling Losers.* Journal of Finance.
- Amihud (2002). *Illiquidity and stock returns.* Journal of Financial Markets.
- Huang et al. (2023). *FinBERT: A Large Language Model for Extracting Information from Financial Text.* (ProsusAI/finbert)

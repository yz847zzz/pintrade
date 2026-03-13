# pintrade

pintrade is a quantitative equity research framework that constructs composite alpha signals from technical, fundamental, and NLP-derived factors, then validates them through IC analysis and walk-forward backtesting. It targets long/short equity strategies on S&P 500 universes, with regime-conditional short sizing (VIX + 200MA + 12-month momentum) to manage drawdown in bear markets. The full pipeline runs from raw OHLCV prices and SEC EDGAR filings through FinBERT sentiment scoring to a self-financing L/S equity curve — all with strict point-in-time discipline and no lookahead bias.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt
pip install -e .          # installs pintrade as an importable package

# 2. Run the SP25 walk-forward (5-fold, 2019-2024)
python pintrade/run_walk_forward_sp25.py

# 3. Run IC + MI analysis on 10 large-caps
python -m pintrade.analysis.ic_analysis

# 4. Run the full SEC filing + sentiment pipeline (SP100, ~6-8hr GPU)
python pintrade/run_sec_sp100_remaining.py

# 5. Run sector-neutral SP100 backtest
python pintrade/run_sp100_sector_neutral.py
```

**Prerequisites:** Python 3.10+. For FinBERT GPU acceleration (recommended):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## Project Structure

```
pintrade/
├── data/
│   ├── loader.py              # yfinance OHLCV downloader → (Ticker, Price) MultiIndex
│   ├── ingestion/
│   │   ├── sec_downloader.py  # SEC EDGAR filing downloader (10-K, 8-K, 10-Q)
│   │   └── news_downloader.py # yfinance + RSS news downloader
│   └── pipeline/
│       ├── pipeline.py        # Orchestrates full filing → sentiment flow
│       ├── pdf_extractor.py   # HTML/PDF text extraction
│       ├── chunker.py         # Section-aware text chunking
│       ├── sentiment.py       # FinBERT scoring (ProsusAI/finbert)
│       ├── sql_store.py       # SQLite persistence (sentiment.db)
│       └── vector_store.py    # ChromaDB vector store
├── features/
│   ├── factors.py             # All alpha factors + composite score
│   └── ekop_model.py          # EKOP PIN model (MLE via SLSQP)
├── backtest/
│   ├── engine.py              # Long-only + Long/Short backtester
│   └── regime.py              # Regime multiplier (VIX + 200MA + momentum)
├── analysis/
│   ├── ic_analysis.py         # Rank IC, ICIR, MI analysis
│   └── walk_forward.py        # Rolling walk-forward with per-fold IC selection
├── models/
│   ├── base.py                # BaseAlphaModel ABC
│   └── factor_model.py        # FactorAlphaModel (thin wrapper)
├── utils/
│   └── metrics.py             # sharpe_ratio, max_drawdown, annualized_return
├── run_walk_forward_sp25.py
├── run_sp100.py
├── run_sp100_sector_neutral.py
├── run_wf_regime_comparison.py
├── run_ic_analysis.py
└── run_sec_sp100_remaining.py
tests/
├── test_ic_mi.py
├── test_ls_backtest.py
├── test_regime.py
└── test_sentiment_loader.py
```

---

## Key Results

| Universe       | Mode              | OOS Sharpe | Ann Return | Max DD |
|----------------|-------------------|:----------:|:----------:|:------:|
| SP25 (25 stocks) | L/S unconditional | **1.39**  | ~28%       | ~18%   |
| SP100 sector-neutral | L/S + regime  | **0.80**  | ~15%       | ~22%   |

Walk-forward protocol: 5 annual folds (2019–2024), 1-year train / 1-year test, factors selected per fold by |t-stat| > 2.0 on in-sample IC. No data snooping — weights are determined from IS data only and applied cold to OOS.

---

## Documentation

- [Architecture](docs/ARCHITECTURE.md) — system diagram and data flow
- [Factors](docs/FACTORS.md) — every factor: formula, IC, weight rationale
- [Strategy](docs/STRATEGY.md) — composite score, regime logic, walk-forward results
- [Pipeline](docs/PIPELINE.md) — SEC download, FinBERT scoring, sentiment integration
- [Roadmap](docs/ROADMAP.md) — planned improvements and future directions

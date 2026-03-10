# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt
pip install -e .   # install pintrade as a package (required for imports to resolve)

# Run tests
pytest

# Run a single test file
pytest tests/test_foo.py

# Run a single test
pytest tests/test_foo.py::test_bar

# Run a module's __main__ block directly (for quick manual checks)
python -m pintrade.data.loader
python -m pintrade.features.factors
python -m pintrade.features.ekop_model
python -m pintrade.backtest.engine
```

## Architecture

`pintrade` is a quantitative equity research framework. The canonical data flow is:

```
load_ohlcv_data()
    → compute_factors()         # includes optional EKOP/PIN via compute_ekop_factor()
    → get_composite_score()     # equal-weight or custom-weight factor blend
    → run_backtest()            # long top-N, equal-weight, monthly/weekly rebalance
```

### Data contract (critical — maintained across all modules)

- **OHLCV DataFrame** (`data/loader.py`): wide format, `Date` index, **`(Ticker, Price)` MultiIndex columns** (level 0 = Ticker, level 1 = Price type). `load_ohlcv_data()` normalizes both single- and multi-ticker yfinance responses to this shape.
- **Factor DataFrame** (`features/factors.py`): `(Date, Ticker)` MultiIndex as the row index, one column per factor. Values are cross-sectionally z-scored daily before returning.
- **Signal Series** (`models/`): `(Date, Ticker)` MultiIndex, float composite score. This is what `BaseAlphaModel.generate_signals()` must return.

### Module responsibilities

| Module | Purpose |
|---|---|
| `data/loader.py` | Downloads OHLCV via yfinance; normalises to consistent MultiIndex shape |
| `features/factors.py` | Computes momentum (21/63/252-day), RSI-5, price z-score, volatility, volume z-score; applies cross-sectional z-scoring; optionally joins PIN |
| `features/ekop_model.py` | EKOP (Easley-Kiefer-O'Hara-Paperman 1996) PIN model — MLE via scipy SLSQP with 5 restarts; classifies each day as Good (+1) / Bad (−1) / No Event (0) |
| `features/pin_factor.py` | Duplicate of `ekop_model.py` (same content) |
| `models/base.py` | `BaseAlphaModel` ABC: `fit(df)` + `generate_signals(df) → Series` |
| `models/factor_model.py` | `FactorAlphaModel` — thin wrapper: calls `compute_factors` + `get_composite_score` |
| `backtest/engine.py` | Vectorized long-only backtest; equal-weight top-N; saves equity curve PNG |
| `utils/metrics.py` | `sharpe_ratio`, `max_drawdown`, `annualized_return` |
| `analysis/ic_analysis.py` | Rank IC (Spearman) per factor; IC summary table; IC/ICIR plots |

### PIN / EKOP notes

- Buy/sell volume is estimated via **Bulk Volume Classification** (BVC): `buy_ratio = (Close − Low) / (High − Low)`.
- EKOP is fitted **once per calendar window** (annual or monthly), then every day in that window is classified using the fitted parameters.
- `compute_ekop_factor()` returns a `(Date, Ticker)` indexed DataFrame with columns `PIN` (model-implied) and `event_label` (+1/−1/0).
- `compute_factors()` joins the PIN result to the main factor DataFrame when `include_pin=True` (default).

### Momentum lookback caveat

The 252-day momentum factor (`Momentum_252D`) requires at least ~253 trading days of history before any valid values appear. Load data starting at least 1.5 years before the intended analysis start date to avoid all-NaN factor rows being dropped.

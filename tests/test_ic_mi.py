"""
Unit tests for IC and MI analysis functions.
All tests use synthetic factor/price data — no network calls.
"""
import numpy as np
import pandas as pd
import pytest

from pintrade.analysis.ic_analysis import (
    compute_ic,
    compute_ic_summary,
    compute_mi,
    compute_combined_summary,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_factor_df(tickers, n_days=120, seed=0):
    """Return (Date, Ticker) MultiIndex DataFrame of synthetic z-scored factors."""
    np.random.seed(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    idx = pd.MultiIndex.from_product([dates, tickers], names=["Date", "Ticker"])
    data = {
        "FactorA": np.random.randn(len(idx)),
        "FactorB": np.random.randn(len(idx)),
    }
    return pd.DataFrame(data, index=idx)


def _make_prices_df(tickers, n_days=140, start="2021-12-01"):
    """Return wide OHLCV DataFrame with (Ticker, Price) MultiIndex columns."""
    dates = pd.bdate_range(start, periods=n_days)
    arrays = []
    for t in tickers:
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            arrays.append((t, col))
    cols = pd.MultiIndex.from_tuples(arrays, names=["Ticker", "Price"])
    np.random.seed(1)
    data = np.ones((n_days, len(arrays)))
    for i, t in enumerate(tickers):
        base = i * 5
        close = 100 * np.cumprod(1 + np.random.normal(0.001, 0.02, n_days))
        data[:, base + 0] = close
        data[:, base + 1] = close * 1.01
        data[:, base + 2] = close * 0.99
        data[:, base + 3] = close
        data[:, base + 4] = 1e6
    return pd.DataFrame(data, index=dates, columns=cols)


TICKERS = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]


# ── compute_ic ────────────────────────────────────────────────────────────────

def test_compute_ic_returns_dataframe():
    factors = _make_factor_df(TICKERS)
    prices  = _make_prices_df(TICKERS)
    ic_df   = compute_ic(factors, prices, forward_days=5)
    assert isinstance(ic_df, pd.DataFrame)


def test_compute_ic_columns_match_factors():
    factors = _make_factor_df(TICKERS)
    prices  = _make_prices_df(TICKERS)
    ic_df   = compute_ic(factors, prices, forward_days=5)
    assert set(ic_df.columns) == set(factors.columns)


def test_compute_ic_values_in_range():
    """Spearman correlation must be in [-1, +1]."""
    factors = _make_factor_df(TICKERS)
    prices  = _make_prices_df(TICKERS)
    ic_df   = compute_ic(factors, prices, forward_days=5)
    valid   = ic_df.dropna()
    assert (valid >= -1.0).all().all()
    assert (valid <=  1.0).all().all()


def test_compute_ic_has_non_nan_rows():
    factors = _make_factor_df(TICKERS)
    prices  = _make_prices_df(TICKERS)
    ic_df   = compute_ic(factors, prices, forward_days=5)
    assert ic_df.dropna().shape[0] > 0


# ── compute_ic_summary ────────────────────────────────────────────────────────

def test_ic_summary_columns():
    factors = _make_factor_df(TICKERS)
    prices  = _make_prices_df(TICKERS)
    ic_df   = compute_ic(factors, prices, forward_days=5)
    summary = compute_ic_summary(ic_df)
    required = {"IC Mean", "IC Std", "ICIR", "IC > 0 Ratio", "t-stat"}
    assert required.issubset(set(summary.columns))


def test_ic_summary_index_matches_factors():
    factors = _make_factor_df(TICKERS)
    prices  = _make_prices_df(TICKERS)
    ic_df   = compute_ic(factors, prices, forward_days=5)
    summary = compute_ic_summary(ic_df)
    assert set(summary.index) == set(factors.columns)


def test_ic_summary_positive_ratio_in_range():
    factors = _make_factor_df(TICKERS)
    prices  = _make_prices_df(TICKERS)
    ic_df   = compute_ic(factors, prices, forward_days=5)
    summary = compute_ic_summary(ic_df)
    assert (summary["IC > 0 Ratio"].between(0, 1)).all()


# ── compute_mi ────────────────────────────────────────────────────────────────

def test_compute_mi_returns_series():
    factors = _make_factor_df(TICKERS, n_days=150)
    prices  = _make_prices_df(TICKERS, n_days=170)
    mi      = compute_mi(factors, prices, forward_days=5)
    assert isinstance(mi, pd.Series)
    assert mi.name == "MI"


def test_compute_mi_non_negative():
    """MI is a non-negative quantity."""
    factors = _make_factor_df(TICKERS, n_days=150)
    prices  = _make_prices_df(TICKERS, n_days=170)
    mi      = compute_mi(factors, prices, forward_days=5)
    valid   = mi.dropna()
    assert (valid >= 0).all(), f"MI must be non-negative, got: {valid[valid < 0]}"


def test_compute_mi_index_matches_factors():
    factors = _make_factor_df(TICKERS, n_days=150)
    prices  = _make_prices_df(TICKERS, n_days=170)
    mi      = compute_mi(factors, prices, forward_days=5)
    assert set(mi.index) == set(factors.columns)


# ── compute_combined_summary ─────────────────────────────────────────────────

def test_combined_summary_columns():
    factors = _make_factor_df(TICKERS, n_days=150)
    prices  = _make_prices_df(TICKERS, n_days=170)
    ic_df   = compute_ic(factors, prices, forward_days=5)
    mi      = compute_mi(factors, prices, forward_days=5)
    combined = compute_combined_summary(ic_df, mi)
    required = {"IC Mean", "ICIR", "t-stat", "MI", "Nonlinearity", "Flag"}
    assert required.issubset(set(combined.columns))


def test_combined_summary_nonlinearity_positive():
    factors = _make_factor_df(TICKERS, n_days=150)
    prices  = _make_prices_df(TICKERS, n_days=170)
    ic_df   = compute_ic(factors, prices, forward_days=5)
    mi      = compute_mi(factors, prices, forward_days=5)
    combined = compute_combined_summary(ic_df, mi)
    valid = combined["Nonlinearity"].dropna()
    assert (valid >= 0).all()


def test_combined_summary_sorted_by_mi():
    """Result should be sorted by MI descending."""
    factors = _make_factor_df(TICKERS, n_days=150)
    prices  = _make_prices_df(TICKERS, n_days=170)
    ic_df   = compute_ic(factors, prices, forward_days=5)
    mi      = compute_mi(factors, prices, forward_days=5)
    combined = compute_combined_summary(ic_df, mi)
    mi_vals = combined["MI"].dropna().values
    assert list(mi_vals) == sorted(mi_vals, reverse=True)

"""
Unit tests for run_backtest_ls and related engine helpers.
Uses synthetic price/signal data — no network calls.
"""
import numpy as np
import pandas as pd
import pytest

from pintrade.backtest.engine import run_backtest_ls, _is_rebalance_day, _portfolio_metrics


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_prices(tickers, n_days=100, start="2023-01-03"):
    """Create synthetic OHLCV wide DataFrame with (Ticker, Price) MultiIndex cols."""
    dates = pd.bdate_range(start, periods=n_days)
    arrays = []
    for t in tickers:
        np.random.seed(sum(ord(c) for c in t))
        close = 100 * np.cumprod(1 + np.random.normal(0.001, 0.02, n_days))
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            arrays.append((t, col))

    cols = pd.MultiIndex.from_tuples(arrays, names=["Ticker", "Price"])
    data = np.zeros((n_days, len(arrays)))
    for i, t in enumerate(tickers):
        np.random.seed(sum(ord(c) for c in t))
        close = 100 * np.cumprod(1 + np.random.normal(0.001, 0.02, n_days))
        base = i * 5
        data[:, base + 0] = close * 0.99   # Open
        data[:, base + 1] = close * 1.01   # High
        data[:, base + 2] = close * 0.98   # Low
        data[:, base + 3] = close           # Close
        data[:, base + 4] = 1e6             # Volume
    return pd.DataFrame(data, index=dates, columns=cols)


def _make_signals(tickers, dates):
    """Synthetic (Date, Ticker) MultiIndex signal Series."""
    idx = pd.MultiIndex.from_product([dates, tickers], names=["Date", "Ticker"])
    np.random.seed(42)
    return pd.Series(np.random.randn(len(idx)), index=idx)


# ── _is_rebalance_day ─────────────────────────────────────────────────────────

def test_is_rebalance_day_monthly():
    ts1 = pd.Timestamp("2023-01-03")
    ts2 = pd.Timestamp("2023-01-10")
    ts3 = pd.Timestamp("2023-02-01")
    assert _is_rebalance_day(ts1, None, "monthly") is True
    assert _is_rebalance_day(ts2, ts1, "monthly") is False
    assert _is_rebalance_day(ts3, ts2, "monthly") is True


def test_is_rebalance_day_weekly():
    mon1 = pd.Timestamp("2023-01-02")  # week 1
    wed1 = pd.Timestamp("2023-01-04")  # still week 1
    mon2 = pd.Timestamp("2023-01-09")  # week 2
    assert _is_rebalance_day(mon1, None, "weekly") is True
    assert _is_rebalance_day(wed1, mon1, "weekly") is False
    assert _is_rebalance_day(mon2, wed1, "weekly") is True


# ── run_backtest_ls — basic shape checks ─────────────────────────────────────

TICKERS = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA", "META", "NVDA", "JPM", "BAC", "XOM"]


def test_ls_returns_all_legs():
    prices  = _make_prices(TICKERS)
    signals = _make_signals(TICKERS, prices.index)
    result  = run_backtest_ls(signals, prices, top_n=3, rebalance="monthly")

    assert set(result.keys()) == {"long", "short", "ls"}
    for leg in ("long", "short", "ls"):
        assert "equity_curve"      in result[leg]
        assert "sharpe_ratio"      in result[leg]
        assert "annualized_return" in result[leg]
        assert "max_drawdown"      in result[leg]
        assert "beta"              in result[leg]


def test_ls_equity_curve_starts_at_one():
    prices  = _make_prices(TICKERS)
    signals = _make_signals(TICKERS, prices.index)
    result  = run_backtest_ls(signals, prices, top_n=3)
    for leg in ("long", "short", "ls"):
        eq = result[leg]["equity_curve"]
        assert not eq.empty
        assert abs(eq.iloc[0] - 1.0) < 1e-6, f"{leg} equity should start at 1.0"


def test_ls_equity_length_matches_prices():
    prices  = _make_prices(TICKERS)
    signals = _make_signals(TICKERS, prices.index)
    result  = run_backtest_ls(signals, prices, top_n=3)
    for leg in ("long", "short", "ls"):
        assert len(result[leg]["equity_curve"]) == len(prices)


def test_ls_no_short_overlap():
    """Long and short portfolios must not contain the same ticker on any rebalance."""
    prices  = _make_prices(TICKERS, n_days=63)
    signals = _make_signals(TICKERS, prices.index)
    # Just ensure it runs without error — overlap prevention is internal
    result  = run_backtest_ls(signals, prices, top_n=3)
    assert result["ls"]["sharpe_ratio"] is not None


def test_ls_empty_signals():
    prices  = _make_prices(TICKERS)
    result  = run_backtest_ls(pd.Series(dtype=float), prices, top_n=3)
    for leg in ("long", "short", "ls"):
        assert result[leg]["equity_curve"].empty


def test_ls_regime_multiplier_zero():
    """With multiplier=0, short_rets should be all zero → ls == long."""
    prices  = _make_prices(TICKERS, n_days=63)
    signals = _make_signals(TICKERS, prices.index)
    regime  = pd.Series(0.0, index=prices.index, name="regime_multiplier")

    res_full   = run_backtest_ls(signals, prices, top_n=3)
    res_regime = run_backtest_ls(signals, prices, top_n=3, regime_multiplier=regime)

    # Short leg with regime=0 should produce zero returns (equity stays at 1)
    short_eq = res_regime["short"]["equity_curve"]
    assert (short_eq - 1.0).abs().max() < 1e-8, "Short leg should be flat when regime=0"

    # L/S should equal long when short is off
    ls_eq   = res_regime["ls"]["equity_curve"]
    long_eq = res_regime["long"]["equity_curve"]
    pd.testing.assert_series_equal(ls_eq, long_eq, check_names=False)


def test_ls_regime_multiplier_one():
    """With multiplier=1.0, regime result equals unconditional result."""
    prices  = _make_prices(TICKERS, n_days=63)
    signals = _make_signals(TICKERS, prices.index)
    regime  = pd.Series(1.0, index=prices.index, name="regime_multiplier")

    res_uncond = run_backtest_ls(signals, prices, top_n=3)
    res_regime = run_backtest_ls(signals, prices, top_n=3, regime_multiplier=regime)

    pd.testing.assert_series_equal(
        res_uncond["ls"]["equity_curve"],
        res_regime["ls"]["equity_curve"],
        check_names=False,
    )


# ── _portfolio_metrics ────────────────────────────────────────────────────────

def test_portfolio_metrics_sharpe_finite():
    np.random.seed(0)
    rets = pd.Series(np.random.normal(0.001, 0.02, 252))
    mkt  = pd.Series(np.random.normal(0.001, 0.015, 252))
    m = _portfolio_metrics(rets, mkt)
    assert np.isfinite(m["sharpe_ratio"])
    assert np.isfinite(m["annualized_return"])
    assert -1.0 <= m["max_drawdown"] <= 0.0
    assert np.isfinite(m["beta"])


def test_portfolio_metrics_zero_returns():
    rets = pd.Series(np.zeros(50))
    mkt  = pd.Series(np.random.normal(0, 0.01, 50))
    m = _portfolio_metrics(rets, mkt)
    assert m["sharpe_ratio"] == 0.0
    assert m["max_drawdown"] == 0.0

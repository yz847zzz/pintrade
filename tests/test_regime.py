"""
Unit tests for pintrade.backtest.regime.

All yfinance downloads are mocked so tests run offline.
"""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from pintrade.backtest.regime import compute_regime_multiplier, compute_regime_detail


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_ohlcv_df(dates, prices, tz=None):
    """Return a mock yfinance-style DataFrame with a Close column."""
    df = pd.DataFrame({"Close": prices}, index=pd.DatetimeIndex(dates))
    return df


def _mock_yf_download(vix_dates, vix_prices, spx_dates, spx_prices):
    """
    Returns a side_effect function for yf.download that returns different
    DataFrames depending on the first positional argument (ticker).
    """
    vix_df = _make_ohlcv_df(vix_dates, vix_prices)
    spx_df = _make_ohlcv_df(spx_dates, spx_prices)

    def _download(ticker, *args, **kwargs):
        if "VIX" in ticker:
            return vix_df
        return spx_df

    return _download


# ── Test data ─────────────────────────────────────────────────────────────────

# 400 trading days: enough warmup for 200-day MA and 252-day momentum
N = 400
DATES = pd.bdate_range("2022-01-03", periods=N)

# Trending SPX: starts at 4000, grows 0.05% per day (always above MA)
SPX_BULL = 4000 * np.cumprod(1 + np.full(N, 0.0005))

# VIX: stays at 15 (below threshold) the whole period
VIX_LOW = np.full(N, 15.0)


# ── compute_regime_multiplier ─────────────────────────────────────────────────

def test_multiplier_bull_market_is_zero():
    """Bull market (VIX<25, SPX>200MA, +12m mom) → all 3 indicators off → multiplier=0."""
    with patch("pintrade.backtest.regime.yf.download",
               side_effect=_mock_yf_download(DATES, VIX_LOW, DATES, SPX_BULL)):
        start = str(DATES[300].date())
        end   = str(DATES[-1].date())
        mult  = compute_regime_multiplier(start, end, vix_threshold=25.0,
                                          warmup_days=1)
        assert (mult == 0.0).all(), "Bull market → multiplier should be 0 everywhere"


def test_multiplier_bear_market_is_one():
    """
    All 3 bear indicators active → multiplier=1.0 everywhere.
    SPX trending down; VIX=30; 12m momentum negative.
    """
    spx_bear = 4000 * np.cumprod(1 + np.full(N, -0.001))  # declining
    vix_high = np.full(N, 30.0)

    with patch("pintrade.backtest.regime.yf.download",
               side_effect=_mock_yf_download(DATES, vix_high, DATES, spx_bear)):
        start = str(DATES[300].date())
        end   = str(DATES[-1].date())
        mult  = compute_regime_multiplier(start, end, vix_threshold=25.0,
                                          warmup_days=1)
        # After 252 days of decline, all 3 indicators fire → multiplier=1.0
        assert (mult == 1.0).any(), "Declining market should have some multiplier=1.0 days"


def test_multiplier_values_only_three():
    """Output must only contain values {0.0, 0.5, 1.0}."""
    with patch("pintrade.backtest.regime.yf.download",
               side_effect=_mock_yf_download(DATES, VIX_LOW, DATES, SPX_BULL)):
        start = str(DATES[300].date())
        end   = str(DATES[-1].date())
        mult  = compute_regime_multiplier(start, end, warmup_days=1)
        invalid = mult[~mult.isin([0.0, 0.5, 1.0])]
        assert invalid.empty, f"Unexpected multiplier values: {invalid.unique()}"


def test_multiplier_output_is_series():
    with patch("pintrade.backtest.regime.yf.download",
               side_effect=_mock_yf_download(DATES, VIX_LOW, DATES, SPX_BULL)):
        start = str(DATES[300].date())
        end   = str(DATES[-1].date())
        mult  = compute_regime_multiplier(start, end, warmup_days=1)
        assert isinstance(mult, pd.Series)
        assert mult.name == "regime_multiplier"


def test_multiplier_date_range():
    """Output index must be within [start, end]."""
    with patch("pintrade.backtest.regime.yf.download",
               side_effect=_mock_yf_download(DATES, VIX_LOW, DATES, SPX_BULL)):
        start = str(DATES[300].date())
        end   = str(DATES[-1].date())
        mult  = compute_regime_multiplier(start, end, warmup_days=1)
        assert mult.index.min() >= pd.Timestamp(start)
        assert mult.index.max() <= pd.Timestamp(end)


# ── compute_regime_detail ─────────────────────────────────────────────────────

def test_regime_detail_columns():
    with patch("pintrade.backtest.regime.yf.download",
               side_effect=_mock_yf_download(DATES, VIX_LOW, DATES, SPX_BULL)):
        start = str(DATES[300].date())
        end   = str(DATES[-1].date())
        detail = compute_regime_detail(start, end, warmup_days=1)

    required_cols = {"vix", "ma", "momentum", "bear_count", "multiplier",
                     "spx_close", "vix_level"}
    assert required_cols.issubset(set(detail.columns))


def test_regime_detail_bear_count_range():
    with patch("pintrade.backtest.regime.yf.download",
               side_effect=_mock_yf_download(DATES, VIX_LOW, DATES, SPX_BULL)):
        start = str(DATES[300].date())
        end   = str(DATES[-1].date())
        detail = compute_regime_detail(start, end, warmup_days=1)

    assert detail["bear_count"].between(0, 3).all()


def test_regime_multiplier_consistent_with_detail():
    """multiplier column in detail must match what compute_regime_multiplier returns."""
    with patch("pintrade.backtest.regime.yf.download",
               side_effect=_mock_yf_download(DATES, VIX_LOW, DATES, SPX_BULL)):
        start = str(DATES[300].date())
        end   = str(DATES[-1].date())
        detail = compute_regime_detail(start, end, warmup_days=1)

    # bear_count 0,1→0.0  2→0.5  3→1.0
    expected = detail["bear_count"].map({0: 0.0, 1: 0.0, 2: 0.5, 3: 1.0})
    pd.testing.assert_series_equal(
        detail["multiplier"].astype(float),
        expected.astype(float),
        check_names=False,
    )

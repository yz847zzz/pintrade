"""
Market regime detection for conditional short-leg sizing.

Three bear indicators (each worth 1 point):
  1. VIX  > threshold (default 25)       — fear / tail-risk elevated
  2. SPY  < 200-day rolling MA            — price below long-term trend
  3. SPY 12-month return < 0             — annual momentum negative

Bear score → short-leg multiplier applied per trading day:
  0 or 1  →  0.0   long-only, no shorts
  2       →  0.5   half-size short leg
  3       →  1.0   full L/S (unconditional)

Usage:
    from pintrade.backtest.regime import compute_regime_multiplier

    regime = compute_regime_multiplier('2019-01-01', '2024-12-31')
    # regime is a pd.Series  Date → {0.0, 0.5, 1.0}
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import yfinance as yf


def compute_regime_multiplier(
    start: str,
    end: str,
    vix_threshold: float = 25.0,
    ma_window: int = 200,
    momentum_window: int = 252,
    warmup_days: int = 350,
) -> pd.Series:
    """
    Compute daily short-leg multiplier {0.0, 0.5, 1.0} from 3 regime indicators.

    Indicators are computed at market close on each trading day — no lookahead.
    The 200-day MA and 12-month momentum only require history ending at date t.

    Parameters
    ----------
    start            : first date in output Series (ISO date string)
    end              : last date in output Series (ISO date string)
    vix_threshold    : VIX level above which indicator = bear
    ma_window        : rolling window for SPY MA (trading days)
    momentum_window  : lookback for 12-month momentum (trading days ≈ 252)
    warmup_days      : extra calendar days of history downloaded before start
                       (must cover max(ma_window, momentum_window) trading days)

    Returns
    -------
    pd.Series  Date → float {0.0, 0.5, 1.0}, indexed from start to end,
               forward-filled across weekends / holidays.
    """
    load_start = (
        pd.Timestamp(start) - pd.Timedelta(days=warmup_days)
    ).strftime('%Y-%m-%d')

    # ── Download VIX and S&P 500 ──────────────────────────────────────────────
    vix_raw = yf.download('^VIX',  start=load_start, end=end,
                          auto_adjust=True, progress=False, multi_level_index=False)
    spx_raw = yf.download('^GSPC', start=load_start, end=end,
                          auto_adjust=True, progress=False, multi_level_index=False)

    vix_close = vix_raw['Close'].squeeze().rename('vix')
    spx_close = spx_raw['Close'].squeeze().rename('spx')

    # ── Indicator 1: VIX level ────────────────────────────────────────────────
    ind_vix = (vix_close > vix_threshold).astype(float)

    # ── Indicator 2: SPY price vs 200-day MA ─────────────────────────────────
    spx_ma  = spx_close.rolling(ma_window, min_periods=ma_window // 2).mean()
    ind_ma  = (spx_close < spx_ma).astype(float)

    # ── Indicator 3: 12-month momentum ───────────────────────────────────────
    spx_mom = spx_close.pct_change(momentum_window)
    ind_mom = (spx_mom < 0).astype(float)

    # ── Combine on shared trading days ───────────────────────────────────────
    regime_df = pd.concat([ind_vix, ind_ma, ind_mom], axis=1)
    regime_df.columns = ['vix', 'ma', 'momentum']
    regime_df = regime_df.dropna()

    bear_count = regime_df.sum(axis=1).astype(int)

    multiplier_map = {0: 0.0, 1: 0.0, 2: 0.5, 3: 1.0}
    multiplier = bear_count.map(multiplier_map)

    # Trim to the requested date range; forward-fill over any gaps
    multiplier = multiplier.loc[start:]
    multiplier = multiplier.reindex(
        pd.date_range(start, end, freq='B'),  # business days
        method='ffill',
    ).dropna()

    return multiplier.rename('regime_multiplier')


def compute_regime_detail(
    start: str,
    end: str,
    vix_threshold: float = 25.0,
    ma_window: int = 200,
    momentum_window: int = 252,
    warmup_days: int = 350,
) -> pd.DataFrame:
    """
    Like compute_regime_multiplier but returns all components for diagnostics.

    Returns DataFrame with columns:
      vix, ma, momentum, bear_count, multiplier
    """
    load_start = (
        pd.Timestamp(start) - pd.Timedelta(days=warmup_days)
    ).strftime('%Y-%m-%d')

    vix_raw = yf.download('^VIX',  start=load_start, end=end,
                          auto_adjust=True, progress=False, multi_level_index=False)
    spx_raw = yf.download('^GSPC', start=load_start, end=end,
                          auto_adjust=True, progress=False, multi_level_index=False)

    vix_close = vix_raw['Close'].squeeze()
    spx_close = spx_raw['Close'].squeeze()

    ind_vix = (vix_close > vix_threshold).astype(float)
    spx_ma  = spx_close.rolling(ma_window, min_periods=ma_window // 2).mean()
    ind_ma  = (spx_close < spx_ma).astype(float)
    spx_mom = spx_close.pct_change(momentum_window)
    ind_mom = (spx_mom < 0).astype(float)

    df = pd.concat([ind_vix, ind_ma, ind_mom], axis=1).dropna()
    df.columns = ['vix', 'ma', 'momentum']
    df['bear_count']  = df.sum(axis=1).astype(int)
    df['multiplier']  = df['bear_count'].map({0: 0.0, 1: 0.0, 2: 0.5, 3: 1.0})
    df['spx_close']   = spx_close.reindex(df.index)
    df['vix_level']   = vix_close.reindex(df.index)

    return df.loc[start:]


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    detail = compute_regime_detail('2019-01-01', '2024-12-31')

    print("Regime summary 2019-2024:")
    print(detail['multiplier'].value_counts().sort_index())
    print(f"\nBear count distribution:\n{detail['bear_count'].value_counts().sort_index()}")
    print(f"\n% days with short active (mult > 0): "
          f"{(detail['multiplier'] > 0).mean()*100:.1f}%")

    # Plot
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    detail['vix_level'].plot(ax=axes[0], color='crimson')
    axes[0].axhline(25, color='grey', linestyle='--', lw=0.8)
    axes[0].set_title('VIX Level  (threshold = 25)')
    axes[0].set_ylabel('VIX')

    detail['spx_close'].plot(ax=axes[1], color='steelblue')
    detail['spx_close'].rolling(200, min_periods=1).mean().plot(
        ax=axes[1], color='orange', lw=0.9, linestyle='--')
    axes[1].set_title('S&P500 Close vs 200-day MA')
    axes[1].set_ylabel('Price')

    detail['bear_count'].plot(ax=axes[2], kind='area', color='salmon', alpha=0.6)
    axes[2].set_title('Bear Count (0-3)')
    axes[2].set_ylabel('Score')
    axes[2].set_yticks([0, 1, 2, 3])

    detail['multiplier'].plot(ax=axes[3], color='darkred', drawstyle='steps-post')
    axes[3].set_title('Short-Leg Multiplier  (0=off  0.5=half  1=full)')
    axes[3].set_ylabel('Multiplier')
    axes[3].set_yticks([0.0, 0.5, 1.0])

    plt.tight_layout()
    plt.savefig('regime_detail.png', dpi=120)
    plt.close()
    print("Regime plot saved -> regime_detail.png")

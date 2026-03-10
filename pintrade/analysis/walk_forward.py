"""
Rolling walk-forward backtest.

Default windows: 2-year training, 6-month OOS test, rolling forward 6 months.
Factors are computed on the full available slice per fold (no future leakage —
all factor calculations use only past prices via shift(1) / rolling windows).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.tseries.offsets import DateOffset

from pintrade.data.loader import load_ohlcv_data
from pintrade.features.factors import compute_factors, get_composite_score
from pintrade.backtest.engine import run_backtest


def _date_windows(start: pd.Timestamp, end: pd.Timestamp,
                  train_years: int = 2, test_months: int = 6):
    """Yield (train_start, train_end, test_start, test_end) tuples."""
    test_start = start + DateOffset(years=train_years)
    while True:
        test_end = test_start + DateOffset(months=test_months) - pd.Timedelta(days=1)
        if test_end > end:
            break
        train_start = test_start - DateOffset(years=train_years)
        train_end   = test_start - pd.Timedelta(days=1)
        yield pd.Timestamp(train_start), pd.Timestamp(train_end), \
              pd.Timestamp(test_start),  pd.Timestamp(test_end)
        test_start = test_start + DateOffset(months=test_months)


def run_walk_forward(
    tickers: list,
    start: str,
    end: str,
    train_years: int = 2,
    test_months: int = 6,
    top_n: int = 5,
    rebalance: str = 'monthly',
    include_pin: bool = False,
    momentum_buffer_days: int = 300,
) -> dict:
    """
    Rolling walk-forward backtest.

    For each fold:
    - Load prices from (first_train_start - momentum_buffer_days) to end once
    - Compute all factors on the full slice (expensive; done once for efficiency)
    - Slice signals and prices to the OOS test window
    - Run backtest on the test window and record OOS metrics

    Parameters
    ----------
    tickers               : list of ticker symbols
    start                 : overall period start (ISO date string)
    end                   : overall period end (ISO date string)
    train_years           : length of each training window in years
    test_months           : length of each OOS test window in months
    top_n                 : number of stocks to hold long in backtest
    rebalance             : 'monthly' or 'weekly'
    include_pin           : whether to compute the EKOP PIN factor (slow)
    momentum_buffer_days  : extra history prepended for momentum lookback warmup

    Returns
    -------
    dict with keys:
      'windows'  : DataFrame of per-window OOS metrics
      'summary'  : dict of aggregate OOS statistics
    """
    start_ts = pd.Timestamp(start)
    end_ts   = pd.Timestamp(end)

    windows = list(_date_windows(start_ts, end_ts, train_years, test_months))
    if not windows:
        raise ValueError(
            f"Date range {start} → {end} is too short for "
            f"train={train_years}yr / test={test_months}mo windows."
        )

    print(f"Walk-forward: {len(windows)} windows | "
          f"train={train_years}yr, test={test_months}mo")

    # Load full data once (with momentum warmup buffer)
    load_start = windows[0][0] - pd.Timedelta(days=momentum_buffer_days)
    print(f"Loading OHLCV data {load_start.date()} → {end_ts.date()}...")
    full_df = load_ohlcv_data(tickers, str(load_start.date()), end)

    print(f"Computing factors (include_pin={include_pin})...")
    full_factors = compute_factors(full_df, include_pin=include_pin)
    full_signals = get_composite_score(full_factors)

    window_results = []
    for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
        # Dates that fall inside the test window
        test_dates = full_df.index[
            (full_df.index >= test_start) & (full_df.index <= test_end)
        ]
        if len(test_dates) < 5:
            print(f"  Window {i+1}: skipped (only {len(test_dates)} trading days in test window)")
            continue

        test_prices  = full_df.loc[test_dates]
        signal_mask  = full_signals.index.get_level_values('Date').isin(test_dates)
        test_signals = full_signals[signal_mask]

        if test_signals.empty:
            print(f"  Window {i+1}: skipped (no signals in test window)")
            continue

        result   = run_backtest(test_signals, test_prices, top_n=top_n, rebalance=rebalance)
        oos_sr   = result['sharpe_ratio']
        oos_ret  = result['annualized_return']
        oos_mdd  = result['max_drawdown']

        window_results.append({
            'window':          i + 1,
            'train_start':     train_start.date(),
            'train_end':       train_end.date(),
            'test_start':      test_start.date(),
            'test_end':        test_end.date(),
            'oos_sharpe':      oos_sr,
            'oos_ann_return':  oos_ret,
            'oos_max_drawdown': oos_mdd,
        })

        print(f"  Window {i+1}: {test_start.date()} → {test_end.date()} | "
              f"Sharpe={oos_sr:+.3f}  AnnRet={oos_ret:+.3f}  MDD={oos_mdd:.3f}")

    if not window_results:
        print("No valid windows completed.")
        return {'windows': pd.DataFrame(), 'summary': {}}

    results_df = pd.DataFrame(window_results).set_index('window')

    summary = {
        'n_windows':             len(results_df),
        'mean_oos_sharpe':       results_df['oos_sharpe'].mean(),
        'median_oos_sharpe':     results_df['oos_sharpe'].median(),
        'pct_positive_sharpe':   (results_df['oos_sharpe'] > 0).mean(),
        'mean_oos_ann_return':   results_df['oos_ann_return'].mean(),
        'mean_oos_max_drawdown': results_df['oos_max_drawdown'].mean(),
    }

    return {'windows': results_df, 'summary': summary}


def plot_walk_forward(results: dict, save_path: str = 'walk_forward.png'):
    """
    Two-panel plot:
    1. OOS Sharpe per window (bar)
    2. OOS annualized return per window (bar)
    """
    df = results['windows']
    if df.empty:
        print("No results to plot.")
        return

    labels = [f"W{i}\n{row['test_start']}" for i, row in df.iterrows()]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(8, len(df) * 1.2), 8))

    colors_sr  = ['steelblue' if v >= 0 else 'salmon' for v in df['oos_sharpe']]
    colors_ret = ['steelblue' if v >= 0 else 'salmon' for v in df['oos_ann_return']]

    ax1.bar(labels, df['oos_sharpe'], color=colors_sr)
    ax1.axhline(0, color='black', linewidth=0.8)
    ax1.set_title('OOS Sharpe Ratio per Walk-Forward Window')
    ax1.set_ylabel('Sharpe Ratio')
    ax1.grid(axis='y')

    ax2.bar(labels, df['oos_ann_return'], color=colors_ret)
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.set_title('OOS Annualized Return per Walk-Forward Window')
    ax2.set_ylabel('Annualized Return')
    ax2.grid(axis='y')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Walk-forward plot saved → {save_path}")


if __name__ == '__main__':
    tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA',
               'META', 'NVDA', 'JPM', 'BAC', 'XOM']

    results = run_walk_forward(
        tickers=tickers,
        start='2018-01-01',
        end='2023-12-31',
        train_years=2,
        test_months=6,
        top_n=5,
        rebalance='monthly',
        include_pin=False,
    )

    print("\n=== Walk-Forward OOS Summary ===")
    for k, v in results['summary'].items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\nPer-window results:")
    print(results['windows'].to_string())

    plot_walk_forward(results, save_path='walk_forward.png')

from scipy.stats import spearmanr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def compute_ic(factor_df: pd.DataFrame,
               prices: pd.DataFrame,
               forward_days: int = 21) -> pd.DataFrame:
    """
    Compute daily Rank IC (Spearman) for each factor vs forward returns.

    Parameters
    ----------
    factor_df    : (Date, Ticker) MultiIndex → factor columns, cross-sectionally z-scored
    prices       : wide OHLCV, (Ticker, Price) MultiIndex columns, Date index
    forward_days : horizon for forward return calculation

    Returns
    -------
    DataFrame with Date index, one IC column per factor
    """
    # Extract Close prices → (Date × Ticker)
    close = prices.xs('Close', level='Price', axis=1)

    # Forward return at date t = price[t+n] / price[t] - 1
    fwd_ret = close.pct_change(forward_days).shift(-forward_days)

    ic_records = {}
    for factor in factor_df.columns:
        factor_wide = factor_df[factor].unstack('Ticker')  # (Date × Ticker)
        common_dates = factor_wide.index.intersection(fwd_ret.index)

        daily_ic = []
        for date in common_dates:
            f_vals = factor_wide.loc[date].dropna()
            r_vals = fwd_ret.loc[date].dropna()
            common_tickers = f_vals.index.intersection(r_vals.index)
            if len(common_tickers) < 3:
                daily_ic.append(np.nan)
                continue
            ic, _ = spearmanr(f_vals.loc[common_tickers], r_vals.loc[common_tickers])
            daily_ic.append(float(ic))

        ic_records[factor] = pd.Series(daily_ic, index=common_dates)

    return pd.DataFrame(ic_records)


def compute_ic_summary(ic_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns summary table: IC Mean, IC Std, ICIR, IC>0 Ratio, t-stat.
    """
    n = ic_df.count()
    return pd.DataFrame({
        'IC Mean':      ic_df.mean(),
        'IC Std':       ic_df.std(),
        'ICIR':         ic_df.mean() / ic_df.std(),
        'IC > 0 Ratio': (ic_df > 0).sum() / n,
        't-stat':       ic_df.mean() / (ic_df.std() / np.sqrt(n)),
    })


def plot_ic(ic_df: pd.DataFrame, save_path: str = 'ic_analysis.png'):
    """
    Three-panel plot:
    1. Daily IC time series per factor
    2. Cumulative IC (cumulative sum)
    3. ICIR bar chart (independent x-axis)
    """
    fig = plt.figure(figsize=(12, 14))
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2, sharex=ax1)
    ax3 = fig.add_subplot(3, 1, 3)

    ic_df.plot(ax=ax1)
    ax1.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax1.set_title(f'Factor Daily IC ({len(ic_df)} days)')
    ax1.set_ylabel('IC')
    ax1.grid(True)

    ic_df.cumsum().plot(ax=ax2)
    ax2.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax2.set_title('Cumulative IC')
    ax2.set_ylabel('Cumulative IC')
    ax2.grid(True)

    summary = compute_ic_summary(ic_df)
    summary['ICIR'].plot(kind='bar', ax=ax3)
    ax3.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax3.set_title('ICIR by Factor')
    ax3.set_ylabel('ICIR')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(axis='y')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"IC plot saved → {save_path}")


if __name__ == '__main__':
    from pintrade.data.loader import load_ohlcv_data
    from pintrade.features.factors import compute_factors

    tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA',
               'META', 'NVDA', 'JPM', 'BAC', 'XOM']

    # Extra history at the front for 252-day momentum warmup
    print("Loading OHLCV data...")
    df = load_ohlcv_data(tickers, '2019-01-01', '2023-12-31')

    print("Computing factors (include_pin=False)...")
    factor_df = compute_factors(df, include_pin=False)

    print("Computing IC...")
    ic_df = compute_ic(factor_df, df, forward_days=21)

    summary = compute_ic_summary(ic_df)
    print("\nIC Summary:")
    print(summary.to_string())

    plot_ic(ic_df, save_path='ic_analysis.png')

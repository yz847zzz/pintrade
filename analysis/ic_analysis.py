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


def compute_mi(factor_df: pd.DataFrame,
               prices: pd.DataFrame,
               forward_days: int = 21,
               n_neighbors: int = 5,
               random_state: int = 42) -> pd.Series:
    """
    Compute panel Mutual Information (MI) between each factor and forward returns.

    MI is estimated via the Kraskov k-NN estimator (sklearn implementation).
    All (Date, Ticker) observations are pooled into a single panel — this gives
    more samples than per-day cross-sectional estimation and is more reliable
    for universes of < ~100 tickers.

    Unlike IC (Spearman), MI captures nonlinear and non-monotonic dependencies.
    A factor with high MI but low |IC| has nonlinear alpha: the return
    relationship is curved, threshold-based, or regime-dependent rather than
    simply monotone.

    Parameters
    ----------
    factor_df    : (Date, Ticker) MultiIndex → factor columns
    prices       : wide OHLCV, (Ticker, Price) MultiIndex columns, Date index
    forward_days : horizon for forward return calculation
    n_neighbors  : k for Kraskov estimator (higher k = smoother, more bias)
    random_state : for reproducibility

    Returns
    -------
    Series: factor_name → MI value (in nats, non-negative)
    """
    from sklearn.feature_selection import mutual_info_regression

    close = prices.xs('Close', level='Price', axis=1)
    fwd_ret = close.pct_change(forward_days).shift(-forward_days)

    # Stack forward returns to (Date, Ticker) long format
    fwd_ret_long = fwd_ret.stack()
    fwd_ret_long.index.names = ['Date', 'Ticker']
    fwd_ret_long.name = '_fwd_ret'

    # Build panel: align all factors and forward return together
    # Only keep rows where forward return is non-NaN
    panel = factor_df.join(fwd_ret_long, how='inner').dropna(subset=['_fwd_ret'])

    factors = [c for c in factor_df.columns]
    y = panel['_fwd_ret'].values

    mi_values = {}
    for factor in factors:
        x = panel[factor].values
        mask = ~np.isnan(x)
        if mask.sum() < 50:
            mi_values[factor] = np.nan
            continue
        mi = mutual_info_regression(
            x[mask].reshape(-1, 1),
            y[mask],
            n_neighbors=n_neighbors,
            random_state=random_state,
        )[0]
        mi_values[factor] = float(mi)

    return pd.Series(mi_values, name='MI')


def compute_combined_summary(ic_df: pd.DataFrame,
                             mi: pd.Series,
                             nonlinear_threshold: float = 2.0) -> pd.DataFrame:
    """
    Merge IC summary with MI into a single comparison table.

    Adds a 'Nonlinearity' column defined as:

        Nonlinearity = MI / (|IC Mean| + 1e-4)

    High Nonlinearity means the factor carries information that Spearman rank
    correlation cannot detect — the return relationship is non-monotonic.

    A factor is flagged '* NONLINEAR' when:
        Nonlinearity >= nonlinear_threshold  AND  |ICIR| < 0.3

    Parameters
    ----------
    ic_df               : output of compute_ic()
    mi                  : output of compute_mi()
    nonlinear_threshold : Nonlinearity ratio above which the flag is raised

    Returns
    -------
    DataFrame sorted by MI descending, with columns:
        IC Mean | ICIR | t-stat | MI | Nonlinearity | Flag
    """
    ic_sum = compute_ic_summary(ic_df)

    combined = pd.DataFrame({
        'IC Mean':      ic_sum['IC Mean'],
        'ICIR':         ic_sum['ICIR'],
        't-stat':       ic_sum['t-stat'],
        'MI':           mi,
    })

    combined['Nonlinearity'] = (
        combined['MI'] / (combined['IC Mean'].abs() + 1e-4)
    )

    # Rank by MI and by |IC Mean| among factors that have MI values
    valid = combined.dropna(subset=['MI'])
    mi_rank  = valid['MI'].rank(ascending=False)
    ic_rank  = valid['IC Mean'].abs().rank(ascending=False)
    rank_gap = (ic_rank - mi_rank).reindex(combined.index)  # NaN for missing MI

    flag_nonlinear = (
        (combined['Nonlinearity'] >= nonlinear_threshold) &
        (combined['ICIR'].abs() < 0.3)
    )
    rank_gap_flag = rank_gap.fillna(0) >= 3

    combined['Flag'] = ''
    flagged = combined.index[flag_nonlinear | rank_gap_flag]
    # Only flag rows that actually have MI data
    flagged = flagged[flagged.isin(valid.index)]
    combined.loc[flagged, 'Flag'] = '* NONLINEAR'

    return combined.sort_values('MI', ascending=False, na_position='last')


def plot_ic_mi(ic_df: pd.DataFrame,
               mi: pd.Series,
               save_path: str = 'ic_mi_analysis.png'):
    """
    Four-panel plot:
    1. Daily IC time series per factor
    2. Cumulative IC
    3. ICIR bar chart
    4. MI bar chart — nonlinear factors highlighted in orange
    """
    ic_sum = compute_ic_summary(ic_df)
    combined = compute_combined_summary(ic_df, mi)

    nonlinear_factors = set(combined[combined['Flag'] != ''].index)

    fig = plt.figure(figsize=(14, 18))
    ax1 = fig.add_subplot(4, 1, 1)
    ax2 = fig.add_subplot(4, 1, 2, sharex=ax1)
    ax3 = fig.add_subplot(4, 1, 3)
    ax4 = fig.add_subplot(4, 1, 4)

    # Panel 1: daily IC
    ic_df.plot(ax=ax1)
    ax1.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax1.set_title(f'Factor Daily IC — Spearman ({len(ic_df)} days)')
    ax1.set_ylabel('IC')
    ax1.grid(True)

    # Panel 2: cumulative IC
    ic_df.cumsum().plot(ax=ax2)
    ax2.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax2.set_title('Cumulative IC')
    ax2.set_ylabel('Cumulative IC')
    ax2.grid(True)

    # Panel 3: ICIR bar (significant factors green, rest grey)
    icir = ic_sum['ICIR'].reindex(combined.index)
    colors3 = ['steelblue' if abs(v) >= 0.15 else 'lightgrey'
               for v in icir.fillna(0)]
    icir.plot(kind='bar', ax=ax3, color=colors3)
    ax3.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax3.axhline(0.15,  color='green', linewidth=0.6, linestyle=':')
    ax3.axhline(-0.15, color='green', linewidth=0.6, linestyle=':')
    ax3.set_title('ICIR by Factor  (green dotted = ±0.15 threshold)')
    ax3.set_ylabel('ICIR')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(axis='y')

    # Panel 4: MI bar — nonlinear factors highlighted
    mi_ordered = mi.reindex(combined.index)
    colors4 = ['darkorange' if f in nonlinear_factors else 'steelblue'
               for f in mi_ordered.index]
    mi_ordered.plot(kind='bar', ax=ax4, color=colors4)
    ax4.set_title('Mutual Information by Factor  (orange = nonlinear alpha flag)')
    ax4.set_ylabel('MI (nats)')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(axis='y')

    # Legend patch for MI panel
    from matplotlib.patches import Patch
    ax4.legend(handles=[
        Patch(color='darkorange', label='Nonlinear flag (MI >> IC)'),
        Patch(color='steelblue',  label='Normal'),
    ])

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"IC+MI plot saved → {save_path}")


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

    print("Computing MI...")
    mi = compute_mi(factor_df, df, forward_days=21)

    print("\nIC + MI Combined Summary:")
    combined = compute_combined_summary(ic_df, mi)
    print(combined.to_string())

    plot_ic_mi(ic_df, mi, save_path='ic_mi_analysis.png')

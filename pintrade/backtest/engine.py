import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pintrade.utils.metrics import sharpe_ratio, max_drawdown, annualized_return
from pintrade.data.loader import load_ohlcv_data
from pintrade.models.factor_model import FactorAlphaModel


def run_backtest(
    signals: pd.Series, # composite alpha score, (Date, Ticker) MultiIndex
    prices: pd.DataFrame, # OHLCV wide format from loader.py
    top_n: int = 5, # number of stocks to go long
    rebalance: str = 'monthly' # 'monthly' or 'weekly'
) -> dict:
    """
    Vectorized backtest:
    1. Rebalance on schedule — go long top_n stocks by signal score
    2. Equal weight portfolio
    3. Compute daily returns
    4. Output: equity curve, Sharpe ratio, max drawdown, annualized return
    5. Plot equity curve with matplotlib
    """
    if signals.empty or prices.empty:
        return {
            'equity_curve': pd.Series(dtype=float),
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'annualized_return': 0.0,
            'message': 'Empty signals or prices DataFrame.'
        }

    # Ensure prices has a proper MultiIndex for (Date, Ticker)
    if not isinstance(prices.columns, pd.MultiIndex):
        raise ValueError("Prices DataFrame must have a MultiIndex (Ticker, OHLCV_Type) for columns.")

    # Align signals and prices by date
    common_dates = pd.Index.intersection(signals.index.get_level_values('Date').unique(),
                                         prices.index.unique())
    if common_dates.empty:
        return {
            'equity_curve': pd.Series(dtype=float),
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'annualized_return': 0.0,
            'message': 'No common dates between signals and prices.'
        }

    signals = signals.loc[common_dates]
    prices = prices.loc[common_dates]

    # Convert prices to a daily return series for each ticker
    daily_returns_wide = prices.xs('Close', level=1, axis=1).pct_change()

    # Initialize equity curve
    equity_curve = pd.Series(1.0, index=daily_returns_wide.index)
    portfolio_returns = pd.Series(0.0, index=daily_returns_wide.index)

    last_rebalance_date = None

    for current_date in daily_returns_wide.index:
        if rebalance == 'monthly' and (last_rebalance_date is None or current_date.month != last_rebalance_date.month):
            rebalance_needed = True
        elif rebalance == 'weekly' and (last_rebalance_date is None or current_date.week != last_rebalance_date.week):
            rebalance_needed = True
        else:
            rebalance_needed = False

        if rebalance_needed:
            # Get signals for the rebalance date
            rebalance_signals = signals.xs(current_date, level='Date', drop_level=False)
            if rebalance_signals.empty:
                continue # No signals for this date, keep current portfolio or do nothing

            # Select top N stocks
            # Ensure sorting works even with NaNs by filling them for ranking
            ranked_signals = rebalance_signals.sort_values(ascending=False).head(top_n)

            # Identify tickers in the portfolio
            current_portfolio_tickers = ranked_signals.index.get_level_values('Ticker').tolist()
            last_rebalance_date = current_date

        if not current_portfolio_tickers:
            portfolio_returns.loc[current_date] = 0.0
        else:
            # Calculate average return for the chosen stocks for the current day
            daily_returns_portfolio = daily_returns_wide.loc[current_date, current_portfolio_tickers]
            # Handle potential NaNs in daily_returns_portfolio (e.g., if a stock didn't trade)
            daily_returns_portfolio = daily_returns_portfolio.dropna()

            if not daily_returns_portfolio.empty:
                portfolio_returns.loc[current_date] = daily_returns_portfolio.mean()
            else:
                portfolio_returns.loc[current_date] = 0.0 # No valid returns for selected stocks

    # Compute cumulative equity curve
    equity_curve = (1 + portfolio_returns).cumprod()

    # Calculate performance metrics
    sr = sharpe_ratio(portfolio_returns.dropna())
    mdd = max_drawdown(equity_curve)
    ann_ret = annualized_return(equity_curve)

    # Plotting
    plt.figure(figsize=(10, 6))
    equity_curve.plot(title='Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('Equity')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('backtest_equity_curve.png') # Save plot to file
    # plt.show() # Can be uncommented for interactive viewing

    return {
        'equity_curve': equity_curve,
        'sharpe_ratio': sr,
        'max_drawdown': mdd,
        'annualized_return': ann_ret
    }


# ── Long/Short backtest ───────────────────────────────────────────────────────

def _is_rebalance_day(date: pd.Timestamp,
                      last_date: pd.Timestamp | None,
                      rebalance: str) -> bool:
    if last_date is None:
        return True
    if rebalance == 'monthly':
        return date.month != last_date.month
    if rebalance == 'weekly':
        return date.isocalendar()[1] != last_date.isocalendar()[1]
    return False


def _portfolio_metrics(daily_rets: pd.Series,
                       market_rets: pd.Series) -> dict:
    """Compute Sharpe, Ann Return, Max DD, and market Beta for a return series."""
    rets   = daily_rets.replace(0, np.nan).dropna()
    equity = (1 + daily_rets.fillna(0)).cumprod()

    sr      = sharpe_ratio(rets)
    mdd     = max_drawdown(equity)
    ann_ret = annualized_return(equity)

    # Market Beta: slope of OLS regression portfolio ~ market
    aligned = pd.concat([daily_rets, market_rets], axis=1).dropna()
    aligned.columns = ['port', 'mkt']
    if len(aligned) > 2 and aligned['mkt'].var() > 0:
        beta = float(np.cov(aligned['port'], aligned['mkt'])[0, 1]
                     / aligned['mkt'].var())
    else:
        beta = float('nan')

    return {
        'equity_curve':      equity,
        'sharpe_ratio':      round(sr,      3),
        'annualized_return': round(ann_ret,  4),
        'max_drawdown':      round(mdd,      4),
        'beta':              round(beta,     3) if not np.isnan(beta) else float('nan'),
    }


def run_backtest_ls(
    signals: pd.Series,
    prices: pd.DataFrame,
    top_n: int = 5,
    rebalance: str = 'monthly',
    regime_multiplier: pd.Series | None = None,
) -> dict:
    """
    Long/Short backtest tracking three portfolios simultaneously.

    Long:  equal-weight top_n stocks by composite score (buy)
    Short: equal-weight bottom_n stocks by composite score (sell short)
    L/S:   self-financing combination — long_return + short_profit
           Dollar-neutral: $1 long funded by $1 short proceeds.
           Net market exposure ≈ 0 by construction.

    Convention:
      short_return  = -(actual return of shorted stocks)   [positive when they fall]
      ls_return     = long_return + short_return

    Parameters
    ----------
    regime_multiplier : optional daily Series (Date → {0.0, 0.5, 1.0}).
        Scales the short leg each day:
          0.0 → long-only that day (no shorts)
          0.5 → half-size short leg
          1.0 → full short leg (unconditional behaviour when None)
        When None, the short leg always runs at full size (1.0).
        Missing dates default to 0.0 (conservative: no short if no regime data).

    Returns
    -------
    dict with sub-dicts 'long', 'short', 'ls', each containing:
      equity_curve, sharpe_ratio, annualized_return, max_drawdown, beta
    """
    _empty_result = {
        'equity_curve':      pd.Series(dtype=float),
        'sharpe_ratio':      0.0,
        'annualized_return': 0.0,
        'max_drawdown':      0.0,
        'beta':              float('nan'),
    }

    if signals.empty or prices.empty:
        return {'long': _empty_result, 'short': _empty_result, 'ls': _empty_result}

    if not isinstance(prices.columns, pd.MultiIndex):
        raise ValueError("Prices must have (Ticker, Price) MultiIndex columns.")

    # Align to common dates
    sig_dates    = signals.index.get_level_values('Date').unique()
    common_dates = sig_dates.intersection(prices.index)
    if common_dates.empty:
        return {'long': _empty_result, 'short': _empty_result, 'ls': _empty_result}

    prices  = prices.loc[common_dates]
    signals = signals.loc[common_dates]

    close         = prices.xs('Close', level=1, axis=1)
    daily_ret_all = close.pct_change()

    # Equal-weight market return (benchmark for beta)
    market_rets = daily_ret_all.mean(axis=1)

    long_rets  = pd.Series(0.0, index=daily_ret_all.index)
    short_rets = pd.Series(0.0, index=daily_ret_all.index)

    # Pre-align regime multiplier to the price index (forward-fill, default 0)
    if regime_multiplier is not None:
        regime_aligned = (
            regime_multiplier
            .reindex(daily_ret_all.index, method='ffill')
            .fillna(0.0)
        )
    else:
        regime_aligned = None   # sentinel: full short every day

    long_tickers:  list[str] = []
    short_tickers: list[str] = []
    last_rebal: pd.Timestamp | None = None

    for date in daily_ret_all.index:
        if _is_rebalance_day(date, last_rebal, rebalance):
            try:
                day_sig = signals.xs(date, level='Date').dropna()
            except KeyError:
                day_sig = pd.Series(dtype=float)

            if len(day_sig) >= top_n * 2:          # need enough tickers for both sides
                ranked = day_sig.sort_values(ascending=False)
                new_longs  = ranked.index[:top_n].tolist()
                new_shorts = ranked.index[-top_n:].tolist()
                # Safety: ensure no overlap
                new_shorts = [t for t in new_shorts if t not in new_longs]
                if new_longs:
                    long_tickers  = new_longs
                if new_shorts:
                    short_tickers = new_shorts
                last_rebal = date
            elif len(day_sig) >= top_n:             # not enough for both sides
                ranked = day_sig.sort_values(ascending=False)
                long_tickers = ranked.index[:top_n].tolist()
                last_rebal = date

        # ── Long leg (always active) ──────────────────────────────────────────
        if long_tickers:
            avail = [t for t in long_tickers if t in daily_ret_all.columns]
            r = daily_ret_all.loc[date, avail].dropna()
            long_rets.loc[date] = r.mean() if not r.empty else 0.0

        # ── Short leg (scaled by regime multiplier) ───────────────────────────
        if short_tickers:
            scale = (float(regime_aligned.loc[date])
                     if regime_aligned is not None else 1.0)
            if scale > 0.0:
                avail = [t for t in short_tickers if t in daily_ret_all.columns]
                r = daily_ret_all.loc[date, avail].dropna()
                short_rets.loc[date] = (-r.mean() * scale) if not r.empty else 0.0

    # L/S: long P&L + short P&L (self-financing, 2× gross, 0× net exposure)
    ls_rets = long_rets + short_rets

    return {
        'long':  _portfolio_metrics(long_rets,  market_rets),
        'short': _portfolio_metrics(short_rets, market_rets),
        'ls':    _portfolio_metrics(ls_rets,    market_rets),
    }


def plot_ls_equity(
    ls_result: dict,
    title: str = 'Long / Short / L-S Equity Curves',
    save_path: str = 'ls_equity.png',
) -> None:
    """
    Plot equity curves of Long, Short, and L/S combined on one chart.

    Parameters
    ----------
    ls_result : output of run_backtest_ls()
    title     : chart title
    save_path : output PNG path
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    styles = {
        'long':  ('Long-only',  'steelblue',   '-'),
        'short': ('Short-only', 'darkorange',  '--'),
        'ls':    ('L/S Combined', 'forestgreen', '-'),
    }

    for key, (label, color, ls) in styles.items():
        eq = ls_result[key]['equity_curve']
        if eq.empty:
            continue
        sr  = ls_result[key]['sharpe_ratio']
        ar  = ls_result[key]['annualized_return']
        mdd = ls_result[key]['max_drawdown']
        b   = ls_result[key]['beta']
        full_label = (f"{label}  SR={sr:+.2f}  AnnRet={ar*100:+.1f}%  "
                      f"MDD={mdd*100:.1f}%  β={b:.2f}")
        eq.plot(ax=ax, label=full_label, color=color, linestyle=ls, linewidth=1.4)

    ax.axhline(1.0, color='black', linewidth=0.7, linestyle=':')
    ax.set_title(title)
    ax.set_ylabel('Portfolio Value (normalised to 1.0)')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"L/S equity plot saved → {save_path}")


if __name__ == '__main__':
    # Load data
    print("Loading OHLCV data...")
    ticker_universe = ['AAPL','MSFT','GOOG','AMZN','TSLA']
    start_date = '2022-01-01'
    end_date = '2023-12-31'
    df_ohlcv = load_ohlcv_data(ticker_universe, start_date, end_date)

    if df_ohlcv is None or df_ohlcv.empty:
        print("Error: OHLCV data could not be loaded or is empty.")
    else:
        print("Generating signals...")
        model = FactorAlphaModel()
        signals = model.generate_signals(df_ohlcv)

        if signals.empty:
            print("Error: Signals could not be generated or are empty.")
        else:
            print("Running backtest...")
            results = run_backtest(signals, df_ohlcv, top_n=3, rebalance='monthly')
            print("\nBacktest Results:")
            print(f"  Sharpe Ratio: {results['sharpe_ratio']:.4f}")
            print(f"  Max Drawdown: {results['max_drawdown']:.4f}")
            print(f"  Annualized Return: {results['annualized_return']:.4f}")

            if not results['equity_curve'].empty:
                print("Equity curve plot saved to backtest_equity_curve.png")

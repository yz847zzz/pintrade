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

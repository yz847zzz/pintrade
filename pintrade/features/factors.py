import pandas as pd
import numpy as np

def _calculate_rsi(series, window):
    """Calculates the Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def _cross_sectional_zscore_normalize(factor_df):
    """
    Normalizes factors using cross-sectional z-scoring (daily).
    Handles cases where std() is zero or all values are NaN for a given cross-section.
    """
    normalized_frames = []
    for date in factor_df.index.get_level_values('Date').unique():
        daily_factors = factor_df.loc[date] # Get cross-section for the current date (index is Ticker, columns are factors)
        
        # Apply z-score column-wise (across tickers for each factor)
        normalized_daily_factors = pd.DataFrame(index=daily_factors.index, columns=daily_factors.columns)
        for col in daily_factors.columns:
            series = daily_factors[col]
            if series.isnull().all():
                normalized_daily_factors[col] = np.nan
            else:
                std_val = series.std()
                if std_val == 0:
                    normalized_daily_factors[col] = 0.0
                else:
                    normalized_daily_factors[col] = (series - series.mean()) / std_val
        
        # Reconstruct MultiIndex for the normalized daily frame
        multi_idx = pd.MultiIndex.from_product([[date], daily_factors.index], names=['Date', 'Ticker'])
        normalized_daily_factors.index = multi_idx
        normalized_frames.append(normalized_daily_factors)

    if not normalized_frames:
        return pd.DataFrame(index=pd.MultiIndex.from_arrays([[],[]], names=['Date', 'Ticker']))

    normalized_df = pd.concat(normalized_frames)
    # After normalization, fill any remaining NaNs (e.g., from initial NaNs or all-NaN groups) with 0
    # Only fillna(0) AFTER cross-sectional normalization, if necessary. For now, let's rely on dropna at the compute_factors level.
    # normalized_df = normalized_df.fillna(0) 
    return normalized_df

def compute_factors(ohlcv_df: pd.DataFrame, include_pin: bool = True) -> pd.DataFrame:
    """
    Computes cross-sectional alpha factors from OHLCV data.
    Returns a DataFrame with factors, indexed by (Date, Ticker).
    
    Assumes ohlcv_df has a MultiIndex (Ticker, OHLCV_Type) for columns and a Date index for rows.
    """
    all_factors_by_ticker = []

    # Iterate through each ticker in the DataFrame
    for ticker in ohlcv_df.columns.get_level_values(0).unique():
        # Select data for the current ticker
        ticker_data = ohlcv_df[ticker]
        # ticker_data now has Date index and columns: Open, High, Low, Close, Volume

        ticker_factors_df = pd.DataFrame(index=ticker_data.index) # DataFrame for this ticker's factors
        
        # --- Momentum (1M, 3M, 12M returns, excluding most recent month) ---
        for period_days in [21, 63, 252]:
            past_price = ticker_data['Close'].shift(period_days)
            current_price_minus_1 = ticker_data['Close'].shift(1)
            momentum = (current_price_minus_1 / past_price) - 1
            ticker_factors_df[f'Momentum_{period_days}D'] = momentum

        # --- Mean-reversion ---
        # 5-day RSI
        ticker_factors_df['RSI_5D'] = _calculate_rsi(ticker_data['Close'], window=5)

        # 20-day z-score of closing price
        rolling_mean_20d = ticker_data['Close'].rolling(window=20).mean()
        rolling_std_20d = ticker_data['Close'].rolling(window=20).std()
        ticker_factors_df['Price_Zscore_20D'] = (ticker_data['Close'] - rolling_mean_20d) / rolling_std_20d

        # --- Volatility (20-day rolling standard deviation of daily returns) ---
        daily_returns = ticker_data['Close'].pct_change()
        ticker_factors_df['Volatility_20D'] = daily_returns.rolling(window=20).std()

        # --- Volume (20-day z-score of volume) ---
        rolling_mean_vol_20d = ticker_data['Volume'].rolling(window=20).mean()
        rolling_std_vol_20d = ticker_data['Volume'].rolling(window=20).std()
        ticker_factors_df['Volume_Zscore_20D'] = (ticker_data['Volume'] - rolling_mean_vol_20d) / rolling_std_vol_20d
        
        # Add a 'Ticker' column before appending to flatten and then re-MultiIndex later
        all_factors_by_ticker.append(ticker_factors_df.assign(Ticker=ticker))
    
    if not all_factors_by_ticker:
        return pd.DataFrame(index=pd.MultiIndex.from_arrays([[],[]], names=['Date', 'Ticker']))

    # Concatenate all ticker-specific factor DataFrames
    combined_factors = pd.concat(all_factors_by_ticker)
    
    # Set MultiIndex (Date, Ticker)
    combined_factors = combined_factors.set_index([combined_factors.index, 'Ticker']).sort_index()
    combined_factors.index.names = ['Date', 'Ticker']

    # Reinstating dropna after enough debugging
    combined_factors = combined_factors.dropna()

    # Apply cross-sectional z-scoring normalization
    factor_df = _cross_sectional_zscore_normalize(combined_factors)
    
    if include_pin:
        from pintrade.features.ekop_model import compute_ekop_factor
        pin_df = compute_ekop_factor(ohlcv_df, period='annual') # Pass ohlcv_df as it needs original data
        # pin_df has columns: PIN, event_label
        # join on (Date, Ticker) MultiIndex
        factor_df = factor_df.join(pin_df, how='left')
    
    return factor_df

def get_composite_score(factor_df: pd.DataFrame, weights: dict = None) -> pd.Series:
    """
    Combines all factors into a single composite alpha score.
    Uses equal weights by default, but accepts a custom weights dict.
    The output Series is indexed by (Date, Ticker).
    """
    if factor_df.empty:
        return pd.Series(dtype=float, index=pd.MultiIndex.from_arrays([[],[]], names=['Date', 'Ticker']))

    if weights is None:
        # Equal weighting
        num_factors = len(factor_df.columns)
        if num_factors == 0:
            return pd.Series(0, index=factor_df.index, dtype=float)
        composite_score = factor_df.mean(axis=1)
    else:
        # Custom weighting
        weighted_factors = pd.DataFrame(index=factor_df.index)
        for factor, weight in weights.items():
            if factor in factor_df.columns:
                weighted_factors[factor] = factor_df[factor] * weight
            else:
                print(f"Warning: Factor '{factor}' not found in DataFrame. Skipping.")
        
        if weighted_factors.empty:
             composite_score = pd.Series(0, index=factor_df.index, dtype=float)
        else:
            composite_score = weighted_factors.sum(axis=1) # Sum weighted factors

    return composite_score

if __name__ == "__main__":
    # Example Usage
    from pintrade.data.loader import load_ohlcv_data

    ticker_universe = ["AAPL", "MSFT", "GOOG"]
    start = "2022-01-01" # Adjusted start date to allow for 252-day momentum calculation
    end = "2024-01-01"

    ohlcv_data = load_ohlcv_data(ticker_universe, start, end)

    if ohlcv_data is not None and not ohlcv_data.empty:
        print("\n--- Generating Alpha Factors ---")
        alpha_factors_df = compute_factors(ohlcv_data)
        print("\nAlpha Factors Head (Normalized):")
        print(alpha_factors_df.head())
        print("\nAlpha Factors Info:")
        alpha_factors_df.info()

        print("\n--- Generating Composite Score (Equal Weighted) ---")
        composite_alpha_score = get_composite_score(alpha_factors_df)
        print("\nComposite Alpha Score Head:")
        print(composite_alpha_score.head())
        print("\nComposite Alpha Score Info:")
        composite_alpha_score.info()

        # Example with custom weights
        # custom_weights = {
        #     'Momentum_21D': 0.3,
        #     'RSI_5D': 0.2,
        #     'Price_Zscore_20D': 0.2,
        #     'Volatility_20D': -0.1, # Example: penalize high volatility
        #     'Volume_Zscore_20D': 0.2
        # }
        # custom_composite_score = get_composite_score(alpha_factors_df, weights=custom_weights)
        # print("\nCustom Weighted Composite Alpha Score Head:")
        # print(custom_composite_score.head())

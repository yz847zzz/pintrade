import yfinance as yf
import pandas as pd

def load_ohlcv_data(tickers, start_date, end_date):
    """
    Pulls OHLCV data via yfinance for a configurable ticker universe and date range.
    Returns a clean pandas DataFrame with a MultiIndex (Date, Ticker).
    """
    data = yf.download(tickers, start=start_date, end=end_date)
    
    if isinstance(tickers, str) or len(tickers) == 1:
        # For a single ticker, yfinance returns a DataFrame without a 'Ticker' level
        data = data.stack().to_frame().T
        data.index.name = 'Date'
        data.columns = pd.MultiIndex.from_product([data.columns, [tickers[0] if isinstance(tickers, list) else tickers]])
    
    # Clean column names (remove 'Adj Close' etc. if present, and flatten MultiIndex for easy access)
    # yfinance 0.2.x returns a MultiIndex for columns if multiple tickers are passed
    # Ensure consistent column naming: (OHLCV, Ticker)
    
    # If columns is a MultiIndex, flatten it.
    if isinstance(data.columns, pd.MultiIndex):
        # Rename 'Adj Close' to 'Close' for consistency if it's the primary close column
        if 'Adj Close' in data.columns.get_level_values(0):
            data = data.rename(columns={'Adj Close': 'Close'}, level=0)
        
        # Filter for standard OHLCV columns (Open, High, Low, Close, Volume)
        ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        data = data[[col for col in data.columns if col[0] in ohlcv_cols]]
        
        # Reorder columns to (Ticker, OHLCV) for easier cross-sectional operations later
        data = data.swaplevel(axis=1).sort_index(axis=1)
        
    data.index = pd.to_datetime(data.index)
    
    return data

if __name__ == "__main__":
    # Example Usage
    ticker_universe = ["AAPL", "MSFT", "GOOG"]
    start = "2023-01-01"
    end = "2024-01-01"

    ohlcv_data = load_ohlcv_data(ticker_universe, start, end)

    if ohlcv_data is not None:
        print("Downloaded OHLCV Data Head:")
        print(ohlcv_data.head())
        print("\nDownloaded OHLCV Data Info:")
        ohlcv_data.info()

import yfinance as yf
import pandas as pd

def load_ohlcv_data(tickers, start, end):
    """
    Download OHLCV data via yfinance.
    Returns wide DataFrame with MultiIndex columns (Ticker, Price).
    """
    if isinstance(tickers, str):
        tickers = [tickers]

    df = yf.download(tickers, start=start, end=end,
                     auto_adjust=True, progress=False)

    # Ensure consistent MultiIndex columns regardless of number of tickers
    if not isinstance(df.columns, pd.MultiIndex):
        # Single ticker — yfinance returns flat columns
        df.columns = pd.MultiIndex.from_product(
            [tickers, df.columns], names=['Ticker', 'Price']
        )
    else:
        # Multiple tickers — swap to (Ticker, Price) order if needed
        if df.columns.names[0] != 'Ticker':
            df = df.swaplevel(axis=1).sort_index(axis=1)
        df.columns.names = ['Ticker', 'Price']

    df.index.name = 'Date'
    return df

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

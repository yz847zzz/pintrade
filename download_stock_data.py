
import yfinance as yf
import pandas as pd

def download_stock_data(ticker, start_date, end_date):
    """Downloads historical stock data using yfinance."""
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if not data.empty:
            print(f"Successfully downloaded data for {ticker} from {start_date} to {end_date}.")
            print(data.head())
            return data
        else:
            print(f"No data found for {ticker} in the specified date range.")
            return None
    except Exception as e:
        print(f"Error downloading data for {ticker}: {e}")
        return None

if __name__ == "__main__":
    # Example usage
    stock_ticker = "AAPL"  # Apple Inc.
    start = "2023-01-01"
    end = "2024-01-01"

    aapl_data = download_stock_data(stock_ticker, start, end)

    if aapl_data is not None:
        # You can save the data to a CSV file
        # aapl_data.to_csv(f"{stock_ticker}_historical_data.csv")
        # print(f"Data saved to {stock_ticker}_historical_data.csv")
        pass

from pintrade.data.loader import load_ohlcv_data
from pintrade.models.factor_model import FactorAlphaModel
from pintrade.backtest.engine import run_backtest

TICKERS = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA']
START = '2022-01-01'
END = '2023-12-31'
TOP_N = 3
REBALANCE = 'monthly'

if __name__ == '__main__':
 print("=== PINtrade Pipeline ===")
 print("1. Loading data...")
 df = load_ohlcv_data(TICKERS, START, END)

 print("2. Generating signals...")
 model = FactorAlphaModel()
 signals = model.generate_signals(df)

 print("3. Running backtest...")
 results = run_backtest(signals, df, top_n=TOP_N, rebalance=REBALANCE)

 print("\n=== Results ===")
 for k, v in results.items():
  if isinstance(v, float):
   print(f" {k}: {v:.4f}")
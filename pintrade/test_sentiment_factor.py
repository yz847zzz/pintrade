"""
Test: sentiment factor integrated into compute_factors() and get_composite_score().
Run from: cd E:/emo/workspace && python pintrade/test_sentiment_factor.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from pintrade.data.loader import load_ohlcv_data
from pintrade.features.factors import compute_factors, get_composite_score
from pintrade.backtest.engine import run_backtest

TICKERS     = ["AAPL", "MSFT"]
WARMUP      = "2021-06-01"   # 1.5yr before analysis start (for Momentum_252D)
START       = "2023-01-01"   # analysis period (matches sentiment.db coverage)
END         = "2024-01-01"
SENT_DB     = Path(__file__).parent / "data" / "sentiment.db"

print("=" * 60)
print("1. Loading OHLCV data...")
print("=" * 60)
ohlcv = load_ohlcv_data(TICKERS, WARMUP, END)  # load from warmup for 252D momentum
print(f"   Shape: {ohlcv.shape}  |  Dates: {ohlcv.index[0].date()} → {ohlcv.index[-1].date()}")

print()
print("=" * 60)
print("2. Computing factors WITH sentiment...")
print("=" * 60)
factor_df = compute_factors(
    ohlcv,
    include_pin=False,          # skip PIN for speed
    include_sentiment=True,
    sentiment_db=SENT_DB,
)
print(f"   Factor shape: {factor_df.shape}")
print(f"   Columns: {list(factor_df.columns)}")

# Show sentiment coverage
for col in ['News_Sentiment', 'Filing_Sentiment']:
    if col in factor_df.columns:
        non_nan = factor_df[col].notna().sum()
        total   = len(factor_df)
        print(f"   {col}: {non_nan}/{total} non-NaN rows ({non_nan/total*100:.1f}% coverage)")

print()
print("=" * 60)
print("3. Sample sentiment scores (last 5 trading days)...")
print("=" * 60)
sent_cols = [c for c in ['News_Sentiment', 'Filing_Sentiment'] if c in factor_df.columns]
if sent_cols:
    sample = factor_df[sent_cols].dropna(how='all').tail(10)
    print(sample.to_string())

print()
print("=" * 60)
print("4. Composite score (includes sentiment with weight=+1)...")
print("=" * 60)
signals = get_composite_score(factor_df)
print(f"   Signal shape: {signals.shape}")
print(f"   Score range: {signals.min():.3f} to {signals.max():.3f}")
print()
print("   Top 5 signal dates+tickers:")
print(signals.nlargest(5).to_string())

print()
print("=" * 60)
print("5. Backtest (long top-1, monthly rebalance)...")
print("=" * 60)
results = run_backtest(signals, ohlcv, top_n=1, rebalance='monthly')
print(f"   Sharpe:   {results['sharpe_ratio']:.3f}")
print(f"   Ann Ret:  {results['annualized_return']*100:.2f}%")
print(f"   Max DD:   {results['max_drawdown']*100:.2f}%")

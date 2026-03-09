import pytest
import pandas as pd

def test_imports():
 from pintrade.data.loader import load_ohlcv_data
 from pintrade.features.factors import compute_factors, get_composite_score
 from pintrade.models.factor_model import FactorAlphaModel
 assert True

def test_loader():
 from pintrade.data.loader import load_ohlcv_data
 df = load_ohlcv_data(['AAPL'], '2023-01-01', '2023-03-01')
 assert df is not None
 assert len(df) > 0

def test_factors():
 from pintrade.data.loader import load_ohlcv_data
 from pintrade.features.factors import compute_factors
 df = load_ohlcv_data(['AAPL', 'MSFT'], '2021-01-01', '2023-06-01')
 factors = compute_factors(df, include_pin=False)
 assert isinstance(factors, pd.DataFrame)
 assert len(factors) > 0
 assert 'Momentum_21D' in factors.columns

def test_signals():
 from pintrade.data.loader import load_ohlcv_data
 from pintrade.models.factor_model import FactorAlphaModel
 df = load_ohlcv_data(['AAPL', 'MSFT'], '2021-01-01', '2023-06-01')
 model = FactorAlphaModel()
 model.fit(df)
 signals = model.generate_signals(df)
 assert isinstance(signals, pd.Series)
 assert len(signals) > 0
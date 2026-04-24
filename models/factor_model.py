from pintrade.models.base import BaseAlphaModel
import pandas as pd

class FactorAlphaModel(BaseAlphaModel):
 name = 'FactorModel'
 def fit(self, df: pd.DataFrame) -> None: pass # no training needed
 def generate_signals(self, df: pd.DataFrame) -> pd.Series:
  from pintrade.features.factors import compute_factors, get_composite_score
  factor_df = compute_factors(df)
  return get_composite_score(factor_df)
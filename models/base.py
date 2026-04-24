from abc import ABC, abstractmethod
import pandas as pd

class BaseAlphaModel(ABC):
 @property
 @abstractmethod
 def name(self) -> str:
  ...

 @abstractmethod
 def fit(self, df: pd.DataFrame) -> None:
  ...

 @abstractmethod
 def generate_signals(self, df: pd.DataFrame) -> pd.Series:
  """Returns composite alpha score Series with (Date, Ticker) MultiIndex"""
  ...
from typing import List
import pandas as pd
from .core.data_model import DataModel  # re-export for convenience

def list_numeric_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

def list_categorical_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]

__all__ = [
    "list_numeric_columns",
    "list_categorical_columns",
    "DataModel",
]

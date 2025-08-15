"""Pure transformation operations (stateless) + registry.

Each operation is a function(df, **kwargs) -> df.
Used for reproducible pipelines.
"""
from __future__ import annotations
from typing import Callable, Dict, Optional
import pandas as pd
import numpy as np
from .formula import safe_evaluate_formula, FormulaError

Registry: Dict[str, Callable] = {}

def register(name: str):
    def deco(fn: Callable):
        Registry[name] = fn
        return fn
    return deco

@register("sort")
def op_sort(df: pd.DataFrame, column: str, ascending: bool = True):
    return df.sort_values(column, ascending=ascending)

@register("filter_query")
def op_filter_query(df: pd.DataFrame, expression: str):
    try:
        return df.query(expression)
    except Exception:
        return df

@register("moving_average")
def op_moving_average(
    df: pd.DataFrame,
    column: str,
    window: int,
    out_col: Optional[str] = None,
):
    out_col = out_col or f"{column}_ma{window}"
    df[out_col] = df[column].rolling(window, center=True, min_periods=1).mean()
    return df

@register("formula")
def op_formula(df: pd.DataFrame, formula: str, out_col: str):
    try:
        result = safe_evaluate_formula(df, formula)
        df[out_col] = result
    except FormulaError:
        # Silently ignore invalid formulas (could log)
        return df
    except Exception:
        return df
    return df

@register("baseline_asls")
def op_baseline_asls(
    df: pd.DataFrame,
    column: str,
    lam: float = 1e5,
    p: float = 0.01,
    niter: int = 10,
    out_col: Optional[str] = None,
):
    y = df[column].values.astype(float)
    L = len(y)
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    D2 = sp.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
    w = np.ones(L)
    for _ in range(niter):
        W = sp.diags(w, 0, shape=(L, L))
        Z = W + lam * (D2 @ D2.T)
        z = spla.spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    baseline = z
    corr = y - baseline
    if out_col is None:
        out_col = f"{column}_baseline"
    df[out_col] = baseline
    df[f"{column}_corr"] = corr
    return df

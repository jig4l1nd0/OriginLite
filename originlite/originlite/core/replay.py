"""Operation replay utilities.

Reconstruct a DataFrame by applying a recorded operation sequence
produced by ``DataModel.log`` + manual logging in the app.
"""
from __future__ import annotations

from typing import List, Dict, Any
import pandas as pd
from . import operations as ops


def replay_operations(df: pd.DataFrame, operation_log: List[Dict[str, Any]]) -> pd.DataFrame:
    """Apply the logged operations to a fresh copy of *df*.

    Supports ops: sort, filter, moving_average, formula, baseline_asls.
    Unknown ops are skipped gracefully.
    """
    out = df.copy()
    for rec in operation_log:
        op = rec.get("op")
        params = dict(rec.get("params", {}))
        try:
            if op == "sort":
                out = ops.op_sort(out, column=params["column"], ascending=params.get("ascending", True))
            elif op == "filter":
                col = params.get("column")
                expr = params.get("expr")
                if col and expr:
                    out = ops.op_filter_query(out, f"`{col}` {expr}")
            elif op == "moving_average":
                if "out" in params:
                    params["out_col"] = params.pop("out")
                out = ops.op_moving_average(out, column=params["column"], window=params["window"], out_col=params.get("out_col"))
            elif op == "formula":
                formula = params.get("expr")
                out_col = params.get("out")
                if formula and out_col:
                    out = ops.op_formula(out, formula=formula, out_col=out_col)
            elif op == "baseline_asls":
                out = ops.op_baseline_asls(out, column=params["column"], lam=params.get("lam", 1e5), p=params.get("p", 0.01))
        except Exception:
            continue
    return out

__all__ = ["replay_operations"]

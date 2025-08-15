"""Core data model abstraction.

The DataModel wraps a pandas DataFrame adding:
 - Column metadata (units, label, comments)
 - Operation log (for reproducibility)
 - Convenience methods for applying registered operations
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
import pandas as pd
from . import operations

OperationRecord = Dict[str, Any]


@dataclass
class ColumnMeta:
    name: str
    label: Optional[str] = None
    unit: Optional[str] = None
    comment: Optional[str] = None


@dataclass
class DataModel:
    df: pd.DataFrame
    columns_meta: Dict[str, ColumnMeta] = field(default_factory=dict)
    operations: List[OperationRecord] = field(default_factory=list)

    def log(self, op: str, **params):
        rec: OperationRecord = {
            "op": op,
            "params": params,
            "timestamp": datetime.utcnow().isoformat(),
            "rows": int(len(self.df)),
            "cols": int(len(self.df.columns)),
        }
        self.operations.append(rec)

    # --- Column metadata ---
    def set_column_meta(self, col: str, **meta):
        if col not in self.columns_meta:
            self.columns_meta[col] = ColumnMeta(name=col)
        cm = self.columns_meta[col]
        for k, v in meta.items():
            setattr(cm, k, v)
        self.log("set_column_meta", column=col, meta=meta)

    # --- Operations wrappers ---
    def apply(
        self,
        func: Callable[[pd.DataFrame], pd.DataFrame],
        op_name: str,
        **params,
    ):
        before_shape = self.df.shape
        new_df = func(self.df.copy())
        self.df = new_df
        self.log(
            op_name,
            before_rows=before_shape[0],
            after_rows=new_df.shape[0],
            **params,
        )
        return self

    def add_column(self, name: str, series: pd.Series, **meta):
        self.df[name] = series
        if meta:
            self.set_column_meta(name, **meta)
        self.log("add_column", column=name)
        return self

    # Convenience method to apply registered op by name and log
    def apply_operation(self, name: str, **kwargs):
        if name in operations.Registry:
            fn = operations.Registry[name]
            before_shape = self.df.shape
            self.df = fn(self.df, **kwargs)
            # Normalize certain param names for replay compatibility
            log_params = dict(kwargs)
            if name == 'formula':
                # formula uses formula/out_col -> store as expr/out
                if 'formula' in log_params:
                    log_params['expr'] = log_params.pop('formula')
                if 'out_col' in log_params:
                    log_params['out'] = log_params.pop('out_col')
            if name == 'moving_average':
                if 'out_col' in log_params:
                    log_params['out'] = log_params.pop('out_col')
            self.log(
                name,
                before_rows=before_shape[0],
                after_rows=self.df.shape[0],
                **log_params,
            )
        return self

    # --- Serialization helpers ---
    def to_dict(self) -> Dict[str, Any]:
        return {
            "data": self.df.to_dict(orient="list"),
            "columns_meta": {k: vars(v) for k, v in self.columns_meta.items()},
            "operations": self.operations,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DataModel":
        dm = cls(pd.DataFrame(d["data"]))
        dm.columns_meta = {
            k: ColumnMeta(**v) for k, v in d.get("columns_meta", {}).items()
        }
        dm.operations = d.get("operations", [])
        return dm

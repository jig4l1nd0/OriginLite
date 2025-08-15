"""Secure formula evaluation utilities.

Validates and evaluates user formulas against a whitelist of tokens.
"""
from __future__ import annotations
import re
from typing import Dict, Iterable, Set
import numpy as np
from numexpr import evaluate as ne_eval
import pandas as pd

ALLOWED_FUNCTIONS: Set[str] = {
    # numexpr supported math functions (subset)
    'sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan',
    'sinh', 'cosh', 'tanh', 'exp', 'log', 'log10', 'sqrt', 'abs',
    'where', 'pow', 'maximum', 'minimum'
}
# Operators allowed implicitly by regex validation:
# + - * / ** // % < > <= >= == != & | ^ ~ ( )
IDENTIFIER_RE = re.compile(r"[A-Za-z_]\w*")

class FormulaError(ValueError):
    """Raised when a user formula contains disallowed identifiers or chars."""
    pass


def extract_identifiers(expr: str) -> Set[str]:
    return set(IDENTIFIER_RE.findall(expr))


def validate_formula(expr: str, columns: Iterable[str]):
    ids = extract_identifiers(expr)
    col_set = set(columns)
    disallowed = []
    for ident in ids:
        if ident in col_set:
            continue
        if ident in ALLOWED_FUNCTIONS:
            continue
        # Block names that look like dunder / builtins / modules
        if ident.startswith('__'):
            disallowed.append(ident)
            continue
        # Anything not in columns or allowed funcs is disallowed
        if ident not in col_set:
            disallowed.append(ident)
    if disallowed:
        bad = ', '.join(sorted(disallowed))
        raise FormulaError(f"Disallowed identifiers: {bad}")
    # Basic character whitelist: digits, letters, underscore, parentheses,
    # operators, dot, comma, spaces
    if not re.fullmatch(r"[0-9A-Za-z_\s\.\+\-\*/%<>=!&|,^~()]*", expr):
        raise FormulaError("Expression contains invalid characters")


def safe_evaluate_formula(df: pd.DataFrame, expr: str) -> np.ndarray:
    validate_formula(expr, df.columns)
    local_dict: Dict[str, np.ndarray] = {c: df[c].values for c in df.columns}
    # numexpr will only have access to arrays and allowed functions implicitly
    return ne_eval(expr, local_dict)


__all__ = [
    'safe_evaluate_formula',
    'validate_formula',
    'FormulaError',
    'ALLOWED_FUNCTIONS',
]

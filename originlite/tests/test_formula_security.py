import pandas as pd
import pytest

from originlite.core.formula import (
    validate_formula,
    FormulaError,
    safe_evaluate_formula,
)


def test_validate_formula_allows_basic_math():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [2, 4, 6]})
    validate_formula("x + y", df.columns)
    out = safe_evaluate_formula(df, "x + y")
    assert list(out) == [3, 6, 9]


@pytest.mark.parametrize("expr", [
    "__import__",  # dunder
    "os",          # not a column or allowed function
    "x + unknown",  # unknown identifier
])
def test_validate_formula_blocks_identifiers(expr):
    df = pd.DataFrame({"x": [1, 2]})
    with pytest.raises(FormulaError):
        validate_formula(expr, df.columns)


def test_validate_formula_blocks_bad_chars():
    df = pd.DataFrame({"x": [1, 2]})
    with pytest.raises(FormulaError):
        validate_formula("x + y; import os", df.columns)


def test_safe_evaluate_only_columns_available():
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(FormulaError):
        safe_evaluate_formula(df, "b + 1")

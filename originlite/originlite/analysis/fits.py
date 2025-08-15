"""Fitting routines: linear, polynomial, exponential,
power law, generic non-linear."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


@dataclass
class FitResult:
    model: str
    params: Dict[str, float]
    y_fit: pd.Series
    r2: float
    stderr: float
    ci95: Optional[Dict[str, float]] = None
    equation: Optional[str] = None

 
def _r2(y, yhat):
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot != 0 else 0.0


def _stderr(y, yhat, p):
    n = len(y)
    dof = max(n - p, 1)
    return float(np.sqrt(np.sum((y - yhat) ** 2) / dof))


def _ci95(cov):
    if cov is None:
        return None
    return {
        f"p{i}": 1.96 * float(np.sqrt(cov[i, i]))
        for i in range(cov.shape[0])
    }

 
def fit_linear(x: pd.Series, y: pd.Series) -> FitResult:
    coeffs = np.polyfit(x, y, deg=1)
    yhat = np.polyval(coeffs, x)
    slope = float(coeffs[0])
    intercept = float(coeffs[1])
    eq = f"y = {slope:.4g}x + {intercept:.4g}"
    return FitResult(
        "linear",
        {"slope": slope, "intercept": intercept},
        pd.Series(yhat, index=y.index),
        _r2(y, yhat),
        _stderr(y, yhat, 2),
        equation=eq,
    )

 
def fit_poly(x: pd.Series, y: pd.Series, deg: int = 2) -> FitResult:
    coeffs = np.polyfit(x, y, deg=deg)
    yhat = np.polyval(coeffs, x)
    params = {f"c{i}": float(c) for i, c in enumerate(coeffs[::-1])}
    # Build readable equation from highest degree descending
    terms = []
    for power, coef in enumerate(params.values()):
        if abs(coef) < 1e-14:
            continue
        if power == 0:
            terms.append(f"{coef:.4g}")
        elif power == 1:
            terms.append(f"{coef:.4g}x")
        else:
            terms.append(f"{coef:.4g}x^{power}")
    eq = "y = " + " + ".join(terms) if terms else "y = 0"
    return FitResult(
        f"poly_{deg}",
        params,
        pd.Series(yhat, index=y.index),
        _r2(y, yhat),
        _stderr(y, yhat, deg + 1),
        equation=eq,
    )

 
def fit_exponential(x: pd.Series, y: pd.Series) -> Optional[FitResult]:
    y_pos = y[y > 0]
    x_pos = x[y > 0]
    if len(y_pos) < 2:
        return None
    b, ln_a = np.polyfit(x_pos, np.log(y_pos), 1)
    a = np.exp(ln_a)
    yhat = a * np.exp(b * x)
    eq = f"y = {a:.4g} e^{b:.4g}x"
    return FitResult(
        "exponential",
        {"a": float(a), "b": float(b)},
        pd.Series(yhat, index=y.index),
        _r2(y, yhat),
        _stderr(y, yhat, 2),
        equation=eq,
    )

 
def fit_powerlaw(x: pd.Series, y: pd.Series) -> Optional[FitResult]:
    mask = (x > 0) & (y > 0)
    if mask.sum() < 2:
        return None
    bx, ln_a = np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)
    a = np.exp(ln_a)
    yhat = a * np.power(x, bx)
    eq = f"y = {a:.4g} x^{bx:.4g}"
    return FitResult(
        "powerlaw",
        {"a": float(a), "b": float(bx)},
        pd.Series(yhat, index=y.index),
        _r2(y, yhat),
        _stderr(y, yhat, 2),
        equation=eq,
    )

# --- Adsorption isotherm models ---


def fit_langmuir(x: pd.Series, y: pd.Series) -> Optional[FitResult]:
    """Langmuir isotherm: q = (qmax * K * C) / (1 + K * C)

    Parameters:
      x: concentration C (must be >=0)
      y: adsorbed amount q (must be >=0)
    Returns FitResult with params qmax, K.
    """
    x_arr = pd.to_numeric(x, errors='coerce')
    y_arr = pd.to_numeric(y, errors='coerce')
    mask = (x_arr >= 0) & (y_arr >= 0) & x_arr.notna() & y_arr.notna()
    if mask.sum() < 3:
        return None
    
    def model(c, qmax, K):
        return (qmax * K * c) / (1.0 + K * c)
    try:
        # heuristic initials: qmax ~ max(y), K from low-conc slope
        qmax0 = float(np.nanmax(y_arr[mask])) if mask.any() else 1.0
        # Avoid division by zero; use lower 40% of concentration range
        small_mask = (x_arr[mask] > 0) & (
            x_arr[mask] <= np.nanpercentile(x_arr[mask], 40)
        )
        if small_mask.sum() >= 2:
            slope = np.polyfit(
                x_arr[mask][small_mask], y_arr[mask][small_mask], 1
            )[0]
            K0 = max(slope / qmax0, 1e-6)
        else:
            K0 = 0.1
        popt, pcov = curve_fit(
            model,
            x_arr[mask].values,
            y_arr[mask].values,
            p0=[qmax0, K0],
            maxfev=10000,
        )
    except Exception:
        return None
    yhat = model(x_arr.values, *popt)
    params = {"qmax": float(popt[0]), "K": float(popt[1])}
    eq = (
        "q = (" f"{params['qmax']:.4g} * {params['K']:.4g} * C" " ) / (1 + "
        f"{params['K']:.4g} * C)"
    )
    return FitResult(
        "langmuir",
        params,
        pd.Series(yhat, index=y.index),
        _r2(y_arr, yhat),
        _stderr(y_arr, yhat, len(popt)),
        equation=eq,
    )


def fit_freundlich(x: pd.Series, y: pd.Series) -> Optional[FitResult]:
    """Freundlich isotherm: q = Kf * C^(1/n)

    Linear form: log q = log Kf + (1/n) log C
    Returns params Kf, n.
    """
    x_arr = pd.to_numeric(x, errors='coerce')
    y_arr = pd.to_numeric(y, errors='coerce')
    mask = (x_arr > 0) & (y_arr > 0) & x_arr.notna() & y_arr.notna()
    if mask.sum() < 2:
        return None
    lx = np.log(x_arr[mask])
    ly = np.log(y_arr[mask])
    try:
        slope, intercept = np.polyfit(lx, ly, 1)
    except Exception:
        return None
    n = 1.0 / slope if slope != 0 else np.inf
    Kf = np.exp(intercept)
    yhat = Kf * np.power(x_arr, 1.0 / n)
    params = {"Kf": float(Kf), "n": float(n)}
    eq = (
        f"q = {params['Kf']:.4g} * C^(1/{params['n']:.4g})"
    )
    return FitResult(
        "freundlich",
        params,
        pd.Series(yhat, index=y.index),
        _r2(y_arr, yhat),
        _stderr(y_arr, yhat, 2),
        equation=eq,
    )


def fit_redlich_peterson(x: pd.Series, y: pd.Series) -> Optional[FitResult]:
    """Redlich-Peterson isotherm: q = (A * C) / (1 + B * C^g)

    Parameters: A, B, g (0<g<=1). We constrain g into (0,1] via bounds.
    Returns params A, B, g.
    """
    x_arr = pd.to_numeric(x, errors='coerce')
    y_arr = pd.to_numeric(y, errors='coerce')
    mask = (x_arr >= 0) & (y_arr >= 0) & x_arr.notna() & y_arr.notna()
    if mask.sum() < 3:
        return None
    
    def model(c, A, B, g):
        return (A * c) / (1.0 + B * np.power(c, g))
    try:
        A0 = (
            float(
                np.nanmax(y_arr[mask]) / (np.nanmax(x_arr[mask]) + 1e-9)
            )
            if mask.any()
            else 1.0
        )
        B0 = 0.01
        g0 = 1.0
        popt, pcov = curve_fit(
            model,
            x_arr[mask].values,
            y_arr[mask].values,
            p0=[A0, B0, g0],
            bounds=([0, 0, 0.1], [np.inf, np.inf, 1.0]),
            maxfev=20000,
        )
    except Exception:
        return None
    yhat = model(x_arr.values, *popt)
    params = {"A": float(popt[0]), "B": float(popt[1]), "g": float(popt[2])}
    eq = (
        "q = (" f"{params['A']:.4g} * C) / (1 + "
        f"{params['B']:.4g} * C^{params['g']:.4g})"
    )
    return FitResult(
        "redlich-peterson",
        params,
        pd.Series(yhat, index=y.index),
        _r2(y_arr, yhat),
        _stderr(y_arr, yhat, len(popt)),
        equation=eq,
    )


def fit_generic(
    x: pd.Series,
    y: pd.Series,
    expr: str,
    param_names: str,
    p0: Optional[list] = None,
) -> Optional[FitResult]:
    names = [n.strip() for n in param_names.split(',') if n.strip()]
    if not names:
        return None
    
    def model_func(xarr, *pars):
        loc = {"x": xarr}
        for n, v in zip(names, pars):
            loc[n] = v
        return eval(
            expr,
            {
                "__builtins__": {
                    "abs": abs,
                    "exp": np.exp,
                    "log": np.log,
                    "sqrt": np.sqrt,
                    "sin": np.sin,
                    "cos": np.cos,
                    "pi": np.pi,
                }
            },
            loc,
        )
    try:
        popt, pcov = curve_fit(
            model_func, x.values, y.values, p0=p0, maxfev=10000
        )
    except Exception:
        return None
    yhat = model_func(x.values, *popt)
    params = {n: float(v) for n, v in zip(names, popt)}
    return FitResult(
        f"custom: {expr}",
        params,
        pd.Series(yhat, index=y.index),
        _r2(y, yhat),
        _stderr(y, yhat, len(popt)),
    )

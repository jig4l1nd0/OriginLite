"""Legacy fitting API.

This module now delegates to ``originlite.analysis.fits`` to avoid code duplication.
It is kept for backward compatibility; prefer importing from ``originlite.analysis.fits``.
"""
from __future__ import annotations

from originlite.analysis.fits import (  # noqa: F401
    FitResult,
    fit_linear,
    fit_poly,
    fit_exponential,
    fit_powerlaw,
)

__all__ = [
    "FitResult",
    "fit_linear",
    "fit_poly",
    "fit_exponential",
    "fit_powerlaw",
]


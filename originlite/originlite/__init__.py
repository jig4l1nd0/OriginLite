"""OriginLite package root.

Exposes high-level API surface for convenience.
"""
from .charts import make_chart  # noqa: F401
from .fitting import (  # legacy path
	fit_linear as legacy_fit_linear,
	fit_poly as legacy_fit_poly,
	fit_exponential as legacy_fit_exponential,
	fit_powerlaw as legacy_fit_powerlaw,
)  # noqa: F401

# New architecture re-exports
from .analysis import fits as analysis_fits  # noqa: F401
from .core import DataModel  # noqa: F401
from .project import save_project, load_project  # noqa: F401

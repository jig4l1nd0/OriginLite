import pandas as pd
import numpy as np
from originlite.fitting import fit_linear as legacy_fit_linear
from originlite.analysis.fits import fit_linear, fit_poly, fit_exponential, fit_powerlaw

def test_linear_fit_matches_line():
    x = pd.Series(np.linspace(0, 10, 50))
    y = 3 * x + 5
    res = fit_linear(x, y)
    assert abs(res.params['slope'] - 3) < 1e-6
    assert abs(res.params['intercept'] - 5) < 1e-6
    assert res.r2 > 0.999999

def test_exponential_positive_only():
    x = pd.Series(np.linspace(0, 4, 40))
    y = pd.Series(np.exp(0.5 * x) * 2)
    res = fit_exponential(x, y)
    assert res is not None
    assert res.r2 > 0.99

def test_powerlaw():
    x = pd.Series(np.linspace(1, 10, 50))
    y = 4 * (x ** 2.5)
    res = fit_powerlaw(x, y)
    assert res is not None
    assert abs(res.params['b'] - 2.5) < 0.05


def test_legacy_linear_still_works():
    x = pd.Series([0,1,2,3])
    y = pd.Series([1,3,5,7])
    res = legacy_fit_linear(x,y)
    assert res.r2 > 0.99

"""Signal processing utilities: FFT, peak detection.

Savitzky-Golay smoothing was removed to simplify the UI and feature set.
"""
from __future__ import annotations
from typing import Dict, Any
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

def spectrum_fft(series: pd.Series, sample_spacing: float = 1.0) -> pd.DataFrame:
    """Return single-sided FFT magnitude spectrum for a real-valued series."""
    y = series.values
    n = len(y)
    fft_vals = np.fft.rfft(y - np.mean(y))
    freqs = np.fft.rfftfreq(n, d=sample_spacing)
    mag = np.abs(fft_vals)
    return pd.DataFrame({"freq": freqs, "amplitude": mag})

 
def detect_peaks(
    series: pd.Series, prominence: float = 0.0, height: float = 0.0
) -> Dict[str, Any]:
    """Detect peaks.

    Returns indices, x/y values, plus selected peak properties.
    """
    peaks, props = find_peaks(
        series.values,
        prominence=prominence if prominence > 0 else None,
        height=height if height > 0 else None,
    )
    result = {
        "indices": peaks.tolist(),
        "x": series.index[peaks].tolist(),
        "y": series.iloc[peaks].tolist(),
    }
    for k, v in props.items():
        if hasattr(v, "tolist"):
            result[k] = v.tolist()
    return result

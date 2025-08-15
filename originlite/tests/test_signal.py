import pandas as pd
import numpy as np
from originlite.analysis.signal import spectrum_fft, detect_peaks
from originlite.core.operations import op_baseline_asls
import pandas as pd


def test_fft_basic_properties():
    t = np.linspace(0, 1, 256, endpoint=False)
    s = pd.Series(np.sin(2*np.pi*50*t) + 0.5*np.sin(2*np.pi*80*t))
    spec = spectrum_fft(s, sample_spacing=t[1]-t[0])
    assert {'freq', 'amplitude'} <= set(spec.columns)


def test_detect_peaks_finds():
    s = pd.Series([0, 1, 5, 1, 0, 2, 6, 2, 0])
    res = detect_peaks(s, prominence=1)
    assert len(res['indices']) >= 2


def test_baseline_asls_adds_columns():
    x = np.linspace(0, 100, 201)
    # baseline + peaks
    y = 0.01 * x + np.exp(-(x-40)**2/50) + np.exp(-(x-70)**2/80)
    df = pd.DataFrame({'y': y})
    df2 = op_baseline_asls(df.copy(), column='y', lam=1e5, p=0.01)
    assert 'y_baseline' in df2.columns
    assert 'y_corr' in df2.columns

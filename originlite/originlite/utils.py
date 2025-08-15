import pandas as pd
import io
import re
from xml.etree import ElementTree as ET

def to_numeric_safe(series: pd.Series):
    return pd.to_numeric(series, errors="coerce")

def moving_average(s: pd.Series, window: int = 5):
    if window <= 1:
        return s
    return s.rolling(window, min_periods=1, center=True).mean()


def df_from_upload(uploaded_file):
    if uploaded_file is None:
        return None
    name = uploaded_file.name.lower()
    # Reset pointer (Streamlit UploadedFile persists across reruns)
    if hasattr(uploaded_file, 'seek'):
        try:
            uploaded_file.seek(0)
        except Exception:  # pragma: no cover - defensive
            pass
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    if name.endswith(".xrdml"):
        try:
            raw_bytes = (
                uploaded_file.getvalue()
                if hasattr(uploaded_file, 'getvalue')
                else uploaded_file.read()
            )
            if isinstance(raw_bytes, bytes):
                xml_txt = raw_bytes.decode('utf-8', errors='ignore')
            else:
                xml_txt = str(raw_bytes)
            root = ET.fromstring(xml_txt)
            # Find 2Theta positions node
            pos_candidates = root.findall('.//{*}positions')
            positions_node = None
            for p in pos_candidates:
                axis = p.attrib.get('axis', '').lower()
                if '2theta' in axis or 'gonio' in axis:
                    positions_node = p
                    break
            intensities_node = root.find('.//{*}intensities')
            if positions_node is None or intensities_node is None:
                return None
            # Extract intensities
            ints_text = intensities_node.text or ''
            ys = [float(v) for v in ints_text.split() if v.strip()]
            # Extract start/end/step via child tags or attributes

            def _get_child(tag):
                el = positions_node.find(f'.//{{*}}{tag}')
                return float(el.text) if el is not None and el.text else None
            start = _get_child('startPosition')
            end = _get_child('endPosition')
            step = _get_child('stepSize')
            if start is None:
                try:
                    start = float(positions_node.attrib.get('startPosition'))
                except Exception:
                    start = None
            if end is None:
                try:
                    end = float(positions_node.attrib.get('endPosition'))
                except Exception:
                    end = None
            if step is None:
                try:
                    step = float(positions_node.attrib.get('stepSize'))
                except Exception:
                    step = None
            xs = []
            n = len(ys)
            if start is not None and end is not None and n:
                if step is None and n > 1:
                    step = (end - start) / (n - 1)
                if step is not None:
                    xs = [start + i * step for i in range(n)]
            if (not xs) and positions_node.text:
                try:
                    xs = [float(v) for v in positions_node.text.split()]
                except Exception:
                    xs = []
            m = min(len(xs), len(ys))
            if m == 0:
                return None
            df = pd.DataFrame({
                'two_theta_deg': xs[:m],
                'intensity_counts': ys[:m],
            })
            return df
        except Exception:
            return None
    if name.endswith('.txt'):
        try:
            raw_bytes = (
                uploaded_file.getvalue()
                if hasattr(uploaded_file, 'getvalue')
                else uploaded_file.read()
            )
            if isinstance(raw_bytes, bytes):
                content = raw_bytes.decode('utf-8', errors='ignore')
            else:
                content = str(raw_bytes)
            lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
            data_lines = []
            for ln in lines:
                # Skip metadata/header lines enclosed in quotes or with letters
                if ln.startswith('"'):
                    continue
                # Typical header with column names
                if 'Wavelength' in ln and 'R%' in ln:
                    continue
                # Keep lines that look like two numeric columns (tab or space)
                if re.match(r'^[-+]?\d', ln):  # starts numeric
                    data_lines.append(ln)
            if not data_lines:
                return None
            buf = io.StringIO('\n'.join(data_lines))
            df_txt = pd.read_csv(
                buf,
                sep='\t',
                engine='python',
                header=None,
                comment='#',
            )
            # If only one column (no tabs) fallback to generic whitespace split
            if df_txt.shape[1] == 1:
                buf.seek(0)
                df_txt = pd.read_csv(
                    buf, delim_whitespace=True, header=None, comment='#'
                )
            if df_txt.shape[1] >= 2:
                df_txt = df_txt.iloc[:, :2]
                df_txt.columns = ['wavelength_nm', 'reflectance_pct']
                # Coerce numeric
                for c in df_txt.columns:
                    df_txt[c] = pd.to_numeric(df_txt[c], errors='coerce')
                df_txt = df_txt.dropna(how='any')
            # Filter out empty
            if df_txt.empty:
                return None
            return df_txt.reset_index(drop=True)
        except Exception:
            return None
    # Try CSV fallback
    try:
        return pd.read_csv(uploaded_file)
    except Exception:
        return None

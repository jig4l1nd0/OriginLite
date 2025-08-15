import sys
from pathlib import Path
import pytest


def test_import_app():
    try:
        import streamlit  # noqa: F401
    except ImportError:
        pytest.skip("Streamlit not fully importable in test environment")
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    # Only compile (syntax check) without executing runtime code
    # (avoids st.stop in helper during non-interactive test)
    app_path = root / 'originlite' / 'app.py'
    source = app_path.read_text(encoding='utf-8')
    compile(source, str(app_path), 'exec')

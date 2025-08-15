import sys
from pathlib import Path

# Add both the project directory and its parent to sys.path to resolve the nested package
TEST_FILE = Path(__file__).resolve()
PROJECT_DIR = TEST_FILE.parents[1]  # directory containing 'originlite' package dir
OUTER_DIR = PROJECT_DIR.parent
for p in (PROJECT_DIR, OUTER_DIR):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

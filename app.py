"""Top-level WSGI entrypoint for Render.

Render is starting Gunicorn with `app:app` from the repository root, while the
real Flask application lives in `backend/app.py`. This bootstrap loads the
backend module and exposes its Flask application object as `app`.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent / "backend"
BACKEND_APP_PATH = BACKEND_DIR / "app.py"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

_spec = importlib.util.spec_from_file_location("backend_app", BACKEND_APP_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Unable to load Flask app from {BACKEND_APP_PATH}")

_backend_app = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_backend_app)

app = _backend_app.app

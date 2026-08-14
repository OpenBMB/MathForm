"""Global configuration constants and runtime-mutable state."""

from os import environ
from pathlib import Path
from typing import Optional

_SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_DATASET_PATHS = [
    str(_SCRIPT_DIR / "benchmarks" / "formalmath_lite.jsonl"),
    str(_SCRIPT_DIR / "benchmarks" / "combibench.jsonl"),
    str(_SCRIPT_DIR / "benchmarks" / "proverbench.jsonl"),
    str(_SCRIPT_DIR / "benchmarks" / "fate_m.jsonl"),
    str(_SCRIPT_DIR / "benchmarks" / "fate_h.jsonl"),
    str(_SCRIPT_DIR / "benchmarks" / "fate_x.jsonl"),
]

DEFAULT_MAX_WORKERS = 64
SAVE_INTERVAL = 500

# Kimina Lean Server (compilation check). The server is assumed to be
# already running; set its address via environment variables.
LEAN_SERVER_HOST = environ.get("LEAN_SERVER_HOST", "127.0.0.1")
LEAN_SERVER_PORT = int(environ.get("LEAN_SERVER_PORT", "8000"))
LEAN_SERVER_TIMEOUT = 120
LEAN_SERVER_MAX_WORKERS = 24

# Remote OpenAI-compatible API backend used by both inference and judge.
# API_KEY is read from an environment variable (see main.py) instead of a
# command-line argument to avoid leaking it into process arguments.
API_BASE_URL: Optional[str] = None
API_KEY: Optional[str] = None
API_MODEL: Optional[str] = None

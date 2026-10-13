"""Put src/ on sys.path so `pytest tests/` works with no PYTHONPATH step."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

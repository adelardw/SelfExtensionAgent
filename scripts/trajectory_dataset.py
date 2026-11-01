"""Compatibility launcher. Prefer python -m src.agent_train export."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.agent_train.cli import export_main

if __name__ == '__main__':
    raise SystemExit(export_main())

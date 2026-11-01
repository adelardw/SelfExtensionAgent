"""Compatibility launcher. Prefer python -m src.agent_train --config ..."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.agent_train.cli import main

if __name__ == '__main__':
    raise SystemExit(main())

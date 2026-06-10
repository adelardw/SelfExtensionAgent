import sys
from pathlib import Path

from dotenv import load_dotenv

# Тесты запускаются из корня репо (config.yml и src/ резолвятся относительно).
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

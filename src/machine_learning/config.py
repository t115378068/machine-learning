"""Project paths and local configuration."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def _project_path(env_name: str, default: str) -> Path:
    """Resolve configured paths relative to the project root."""
    value = Path(os.getenv(env_name, default)).expanduser()
    return value if value.is_absolute() else PROJECT_ROOT / value


DATA_DIR = _project_path("DATA_DIR", "data")
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = _project_path("MODEL_DIR", "models")
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

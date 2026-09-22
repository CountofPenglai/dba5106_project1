"""Repository locations, independent of the process working directory."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = PROJECT_ROOT / "100_Portfolios_10x10_Daily.csv"

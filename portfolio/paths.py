"""Repository locations, independent of the process working directory."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = PROJECT_ROOT / "100_Portfolios_10x10_Daily.csv"

# Published releases used to produce the report tables.
ORIGINAL_SHA256 = '11399a830c7b2a192d652b39e866a74c05d7380dd6edf1573ce1c40f9d271c23'
TEST_SHA256 = '222a1b790835db7e2c37920ae4326acb4e866f6bf8d5f0c0fa9bd88e8b68b683'

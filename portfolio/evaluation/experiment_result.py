"""Common experiment result, independent of the chosen decision method."""
from dataclasses import dataclass

@dataclass
class ExperimentResult:
    frames: dict
    metadata: dict
    frozen: dict

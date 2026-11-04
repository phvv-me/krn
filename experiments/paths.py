"""Local evidence storage, excluded from source control and host sync."""

from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "evidence" / "results"

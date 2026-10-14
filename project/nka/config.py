"""Paths and constants shared by every module."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent        # the Project directory
DATA_DIR = ROOT / "data"
CORPUS_DIR = DATA_DIR / "corpus"
EVAL_FILE = DATA_DIR / "eval_queries.json"
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

MODEL_NAME = "BAAI/bge-small-en-v1.5"

# role -> sensitivity levels it may read (security trimming). Unknown roles are denied.
ROLES = {
    "intern": ["public"],
    "employee": ["public", "internal"],
    "hr_partner": ["public", "internal", "confidential"],
}

# Refusal threshold on the best chunk's cosine similarity (as in Day 3 lab 3A).
MIN_SIM = 0.60

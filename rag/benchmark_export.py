from pathlib import Path

RESULTS_DIR = Path("results")

def ensure_results_dir() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
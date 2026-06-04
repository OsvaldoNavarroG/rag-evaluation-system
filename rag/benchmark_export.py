from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path("results")


def ensure_results_dir() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)


def benchmark_timestamp() -> str:
    """
    Returns a timestamp suitable for filenames

    Example:
    2026-06-04_15-42-31
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def benchmark_basename() -> str:
    """
    Example:
    benchmark_2026-06-04_15-42-31
    """
    return f"benchmark_{benchmark_timestamp()}"

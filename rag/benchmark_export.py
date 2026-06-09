import json
from datetime import datetime
from pathlib import Path
from typing import Any

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


def export_json(data: Any) -> Path:
    """
    Export benchmark results to a timestamped JSON file.

    Returns:
        Path to created file.
    """
    ensure_results_dir()

    output_file = RESULTS_DIR / f"{benchmark_basename()}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return output_file

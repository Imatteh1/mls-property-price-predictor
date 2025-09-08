from __future__ import annotations

import shutil
from pathlib import Path

from src.model import train_project


def bootstrap_raw_data(source_csv: Path, raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    destination = raw_dir / source_csv.name
    if not destination.exists():
        shutil.copy2(source_csv, destination)
    return destination


def run_pipeline(source_csv: Path) -> dict[str, object]:
    raw_csv = bootstrap_raw_data(source_csv, Path("data/raw"))
    report = train_project(
        csv_path=raw_csv,
        model_dir=Path("models"),
        report_path=Path("reports/model_evaluation.json"),
    )
    return report


if __name__ == "__main__":
    output = run_pipeline(Path("property_v2.csv"))
    print(output)

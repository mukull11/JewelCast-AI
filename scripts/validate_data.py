"""Validate raw datasets and processed artefacts for every available data mode."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.data.schema import SchemaValidationError, validate_daily_frame
from src.utils.config import load_config
from src.utils.paths import resolve_path


def validate_mode(mode: str, processed_dir: Path) -> list[str]:
    errors: list[str] = []
    daily_path = processed_dir / mode / "daily_sales.parquet"
    if not daily_path.exists():
        return [f"[{mode}] missing {daily_path}"]
    frame = pd.read_parquet(daily_path)
    try:
        summary = validate_daily_frame(frame, mode=mode, require_inventory=(mode == "synthetic"))
    except SchemaValidationError as exc:
        return [f"[{mode}] schema validation failed: {exc}"]

    if mode in {"uci", "m5"} and summary["null_inventory_rows"] != len(frame):
        errors.append(f"[{mode}] available_stock should be entirely null but is not")
    if mode == "synthetic" and summary["null_inventory_rows"] == len(frame):
        errors.append("[synthetic] available_stock is entirely null")

    for artifact in ("sku_master.parquet", "quality_report.json", "provenance.json", "dataset_summary.json"):
        if not (processed_dir / mode / artifact).exists():
            errors.append(f"[{mode}] missing artifact {artifact}")

    print(
        f"[{mode}] rows={summary['rows']:,} skus={summary['skus']:,} "
        f"dates={summary['date_min']}..{summary['date_max']} "
        f"null_inventory={summary['null_inventory_rows']:,}"
    )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate JewelCast AI datasets.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if any check fails.")
    args = parser.parse_args()

    config = load_config()
    raw_dir = resolve_path(config["paths"]["data_raw"])
    processed_dir = resolve_path(config["paths"]["data_processed"])

    errors: list[str] = []
    uci = raw_dir / config["data"]["uci"]["excel_filename"]
    print(f"[raw] UCI present: {uci.exists()} ({uci})")
    for name in ("sales_train_validation.csv", "calendar.csv", "sell_prices.csv"):
        path = raw_dir / name
        print(f"[raw] M5 {name} present: {path.exists()}")

    available = [mode for mode in ("synthetic", "uci", "m5") if (processed_dir / mode / "daily_sales.parquet").exists()]
    if not available:
        errors.append("No processed modes found. Run: python scripts/build_datasets.py --all")
    for mode in available:
        errors.extend(validate_mode(mode, processed_dir))

    if errors:
        print("\nValidation issues:")
        for error in errors:
            print(f"  - {error}")
        return 1 if args.strict else 0

    print("\nAll available datasets passed validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

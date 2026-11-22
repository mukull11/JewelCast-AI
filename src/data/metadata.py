"""Dataset metadata: sizes, SHA-256 hashes, row counts, schemas and provenance."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pyarrow.parquet as pq
from openpyxl import load_workbook

from src.utils.io import sha256_file, write_json
from src.utils.paths import PROJECT_ROOT, resolve_path

ARCHIVE_NAMES = ("online+retail+ii.zip", "m5-forecasting-accuracy.zip")


def _csv_metadata(path: Path) -> dict:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        rows = sum(1 for _ in reader)
    return {
        "type": "csv",
        "rows": rows,
        "columns": header,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _xlsx_metadata(path: Path) -> dict:
    workbook = load_workbook(path, read_only=True, data_only=True)
    sheets: list[dict[str, Any]] = []
    try:
        for worksheet in workbook.worksheets:
            header = next(worksheet.iter_rows(min_row=1, max_row=1, values_only=True), ())
            max_row = worksheet.max_row or 0
            sheets.append(
                {
                    "name": worksheet.title,
                    "rows": int(max(0, max_row - 1)),
                    "columns": [str(value) for value in header],
                }
            )
    finally:
        workbook.close()
    return {
        "type": "xlsx",
        "sheets": sheets,
        "total_rows": int(sum(sheet["rows"] for sheet in sheets)),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _parquet_metadata(path: Path) -> dict:
    schema = pq.read_schema(path)
    return {
        "type": "parquet",
        "rows": int(pq.ParquetFile(path).metadata.num_rows),
        "columns": [{"name": field.name, "type": str(field.type)} for field in schema],
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _raw_files(raw_dir: Path) -> dict:
    files: dict[str, Any] = {}
    for path in sorted(raw_dir.iterdir()):
        if not path.is_file() or path.name == ".gitkeep":
            continue
        suffix = path.suffix.lower()
        if suffix == ".csv":
            files[path.name] = _csv_metadata(path)
        elif suffix in (".xlsx", ".xlsm"):
            files[path.name] = _xlsx_metadata(path)
        else:
            files[path.name] = {
                "type": suffix.lstrip(".") or "unknown",
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    return files


def _archives() -> dict:
    archives: dict[str, Any] = {}
    for name in ARCHIVE_NAMES:
        path = PROJECT_ROOT / name
        if path.exists():
            archives[name] = {
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "preserved": True,
            }
    return archives


def _processed_modes(processed_dir: Path) -> dict:
    modes: dict[str, Any] = {}
    if not processed_dir.exists():
        return modes
    for mode_dir in sorted(processed_dir.iterdir()):
        if not mode_dir.is_dir():
            continue
        summary_path = mode_dir / "dataset_summary.json"
        if not summary_path.exists():
            continue
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        quality = None
        quality_path = mode_dir / "quality_report.json"
        if quality_path.exists():
            with quality_path.open("r", encoding="utf-8") as handle:
                quality = json.load(handle)
        artifacts: dict[str, Any] = {}
        for path in sorted(mode_dir.iterdir()):
            if path.is_file() and path.suffix == ".parquet":
                artifacts[path.name] = _parquet_metadata(path)
        modes[mode_dir.name] = {
            "summary": summary,
            "quality": {
                "rows": quality.get("rows"),
                "skus": quality.get("skus"),
                "date_min": quality.get("date_min"),
                "date_max": quality.get("date_max"),
                "zero_sales_share": quality.get("totals", {}).get("zero_sales_share"),
                "null_inventory_rows": quality.get("null_inventory_rows"),
                "limitations": quality.get("limitations"),
            }
            if quality
            else None,
            "artifacts": artifacts,
        }
    return modes


def _sanitize_paths(value: Any) -> Any:
    """Replace absolute repository paths with portable relative paths."""
    if isinstance(value, str):
        root = str(PROJECT_ROOT)
        if value.startswith(root):
            return value[len(root):].lstrip("\\/").replace("\\", "/")
        return value
    if isinstance(value, dict):
        return {key: _sanitize_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_paths(item) for item in value]
    return value


def build_dataset_metadata(config: Mapping[str, Any]) -> dict:
    raw_dir = resolve_path(config["paths"]["data_raw"])
    processed_dir = resolve_path(config["paths"]["data_processed"])
    synthetic_dir = resolve_path(config["paths"]["data_synthetic"])

    synthetic_config = None
    synthetic_config_path = synthetic_dir / "synthetic_config.json"
    if synthetic_config_path.exists():
        with synthetic_config_path.open("r", encoding="utf-8") as handle:
            synthetic_config = json.load(handle)

    metadata = {
        "project": config.get("project", {}).get("name", "JewelCast AI"),
        "version": config.get("project", {}).get("version"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_archives": _archives(),
        "raw_files": _raw_files(raw_dir),
        "processed_modes": _processed_modes(processed_dir),
        "synthetic_config": synthetic_config,
        "provenance": {
            "uci": "UCI Online Retail II - UK online-retail transactions in GBP. Not Indian retail data and not jewellery-only.",
            "m5": "M5 Forecasting Accuracy (Walmart) - a separate benchmark, not jewellery data and not Voylla data.",
            "synthetic": "Fully simulated Indian fashion-jewellery business dataset. Not real company data.",
        },
        "limitations": {
            "uci": [
                "No inventory records: lost demand cannot be identified.",
                "Jewellery categories are keyword-derived and validated by SKU count.",
                "Historical 2009-2011 UK transactions; not current Indian market data.",
            ],
            "m5": [
                "Only a configurable item-store subset is processed.",
                "No inventory records: lost demand cannot be identified.",
                "US calendar events are stored separately from festival_flag.",
            ],
            "synthetic": [
                "All demand, promotions, inventory and purchase orders are simulated.",
                "Festival dates and effects are documented assumptions.",
            ],
        },
    }
    return _sanitize_paths(metadata)


def write_dataset_metadata(config: Mapping[str, Any], path: str | Path = "data/dataset_metadata.json") -> Path:
    metadata = build_dataset_metadata(config)
    return write_json(metadata, path)

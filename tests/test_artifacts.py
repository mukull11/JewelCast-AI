"""Integration checks for built processed artefacts."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.data.schema import validate_daily_frame
from src.utils.paths import resolve_path

MODES = ("synthetic", "uci", "m5")


@pytest.mark.parametrize("mode", MODES)
def test_processed_mode_artifacts(mode):
    directory = resolve_path(f"data/processed/{mode}")
    daily_path = directory / "daily_sales.parquet"
    if not daily_path.exists():
        pytest.skip(f"{mode} not built")

    frame = pd.read_parquet(daily_path)
    summary = validate_daily_frame(frame, mode=mode, require_inventory=(mode == "synthetic"))
    assert summary["rows"] == len(frame)
    assert summary["skus"] > 0
    assert set(frame["data_source"]) == {mode}

    for name in ("sku_master.parquet", "quality_report.json", "quality_report.md", "provenance.json", "dataset_summary.json"):
        assert (directory / name).exists(), f"{mode}: missing {name}"

    stored = json.loads((directory / "dataset_summary.json").read_text(encoding="utf-8"))
    assert stored["rows"] == len(frame)
    assert stored["skus"] == summary["skus"]

    if mode in {"uci", "m5"}:
        assert frame["available_stock"].isna().all()
    else:
        assert frame["available_stock"].notna().all()
        assert (directory / "inventory_snapshots.parquet").exists()
        assert (directory / "purchase_orders.parquet").exists()


def test_dataset_metadata_exists():
    path = resolve_path("data/dataset_metadata.json")
    if not path.exists():
        pytest.skip("metadata not built")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    assert "synthetic" in metadata["processed_modes"]
    assert "uci" in metadata["processed_modes"]
    assert "m5" in metadata["processed_modes"]

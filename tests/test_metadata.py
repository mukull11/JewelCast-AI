"""Tests for dataset metadata generation and hashes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.utils.io import sha256_file
from src.utils.paths import PROJECT_ROOT, resolve_path

METADATA_PATH = PROJECT_ROOT / "data" / "dataset_metadata.json"


@pytest.mark.skipif(not METADATA_PATH.exists(), reason="dataset_metadata.json not built yet")
def test_metadata_structure_and_hashes():
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    assert metadata["project"] == "JewelCast AI"
    assert metadata["source_archives"]
    assert metadata["raw_files"]
    assert metadata["processed_modes"]

    for name, info in metadata["source_archives"].items():
        assert info["size_bytes"] > 0
        assert len(info["sha256"]) == 64
        assert (PROJECT_ROOT / name).exists()

    calendar = metadata["raw_files"]["calendar.csv"]
    assert calendar["rows"] > 0
    assert calendar["columns"][0] == "date"
    assert calendar["sha256"] == sha256_file(resolve_path("data/raw/calendar.csv"))


@pytest.mark.skipif(not METADATA_PATH.exists(), reason="dataset_metadata.json not built yet")
def test_metadata_provenance_and_sheet_names():
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    assert "Not Indian" in metadata["provenance"]["uci"]
    assert "not jewellery" in metadata["provenance"]["m5"].lower()
    assert "simulated" in metadata["provenance"]["synthetic"].lower()

    sheets = metadata["raw_files"]["online_retail_II.xlsx"]["sheets"]
    names = {sheet["name"] for sheet in sheets}
    assert names == {"Year 2009-2010", "Year 2010-2011"}
    assert sum(sheet["rows"] for sheet in sheets) == 1_067_371


@pytest.mark.skipif(not METADATA_PATH.exists(), reason="dataset_metadata.json not built yet")
def test_metadata_processed_modes_match_summaries():
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    for mode, info in metadata["processed_modes"].items():
        summary_path = resolve_path(f"data/processed/{mode}/dataset_summary.json")
        assert summary_path.exists()
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        assert info["summary"]["rows"] == summary["rows"]
        assert info["summary"]["skus"] == summary["skus"]
        assert info["artifacts"]["daily_sales.parquet"]["rows"] == summary["rows"]

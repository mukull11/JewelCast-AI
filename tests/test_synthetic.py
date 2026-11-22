"""Tests for the synthetic jewellery generator."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.schema import REQUIRED_COLUMNS, validate_daily_frame
from src.data.synthetic import generate_synthetic_dataset


def test_schema_and_provenance(synthetic_result):
    daily = synthetic_result.daily_sales
    for column in REQUIRED_COLUMNS:
        assert column in daily.columns
    summary = validate_daily_frame(daily, mode="synthetic", require_inventory=True)
    assert summary["rows"] > 0
    assert set(daily["data_source"]) == {"synthetic"}
    assert (daily["units_sold"] >= 0).all()
    assert (daily["revenue"] >= 0).all()
    assert daily["available_stock"].notna().all()
    assert (daily["available_stock"] >= 0).all()
    assert set(daily["stockout_flag"].unique()).issubset({0, 1})


def test_products_and_inventory(synthetic_result):
    products = synthetic_result.products
    assert products["sku"].is_unique
    assert len(products) == 24
    assert products["category"].nunique() == 8
    assert (products["initial_inventory"] > 0).all()
    assert (products["supplier_lead_time_days"] >= 1).all()

    snapshots = synthetic_result.inventory_snapshots
    assert not snapshots.empty
    assert (snapshots["available_stock"] >= 0).all()
    assert (snapshots["incoming_po_qty"] >= 0).all()
    assert (snapshots["committed_qty"] >= 0).all()

    orders = synthetic_result.purchase_orders
    assert not orders.empty
    assert (orders["quantity"] > 0).all()
    assert set(orders["status"]).issubset({"received", "open"})


def test_cold_start_and_festivals(synthetic_result):
    assert int(synthetic_result.products["is_cold_start"].sum()) > 0
    daily = synthetic_result.daily_sales
    assert int(daily["festival_flag"].sum()) > 0
    festival_names = set(daily.loc[daily["festival_flag"] == 1, "festival_name"].dropna())
    assert festival_names
    # The fixture window is January-April 2024, so early-year festivals apply.
    assert "Makar Sankranti" in festival_names or "Valentine's Day" in festival_names


def test_reproducibility(config, synthetic_overrides):
    first = generate_synthetic_dataset(config, output_dir=None, overrides=synthetic_overrides)
    second = generate_synthetic_dataset(config, output_dir=None, overrides=synthetic_overrides)
    pd.testing.assert_frame_equal(
        first.daily_sales.sort_values(["sku", "date"]).reset_index(drop=True),
        second.daily_sales.sort_values(["sku", "date"]).reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        first.products.sort_values("sku").reset_index(drop=True),
        second.products.sort_values("sku").reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        first.purchase_orders.sort_values("po_id").reset_index(drop=True),
        second.purchase_orders.sort_values("po_id").reset_index(drop=True),
    )


def test_save_artifacts(config, synthetic_overrides, tmp_path: Path):
    output = tmp_path / "synthetic"
    result = generate_synthetic_dataset(config, output_dir=output, overrides=synthetic_overrides)
    for name in (
        "products.parquet",
        "daily_sales.parquet",
        "inventory_snapshots.parquet",
        "purchase_orders.parquet",
        "festivals.parquet",
        "festival_calendar.csv",
        "synthetic_config.json",
    ):
        assert (output / name).exists(), name
    assert result.settings["seed"] == 123

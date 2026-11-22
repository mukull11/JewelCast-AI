"""Tests for the M5 loader."""

from __future__ import annotations

from pathlib import Path

from src.data.m5_loader import load_m5
from src.preprocessing.clean import clean_daily


def test_load_small_m5(config, m5_dir: Path):
    result = load_m5(
        config,
        raw_dir=m5_dir,
        write=False,
        subset_overrides={"n_items": 2, "n_stores": 2, "max_days": 20},
    )
    audit = result["audit"]
    daily = result["daily_sales"]

    assert audit["selected_series"] == 4
    assert audit["selected_items"] == 2
    assert audit["selected_stores"] == 2
    assert audit["days"] == 20
    assert audit["skus"] == 4
    assert audit["missing_price_rows"] == 0
    assert "Walmart" in audit["provenance"]

    assert len(daily) == 4 * 20
    assert set(daily["data_source"]) == {"m5"}
    assert daily["sku"].str.contains("_").all()
    assert daily["available_stock"].isna().all()
    assert daily["discount_percentage"].isna().all()
    assert daily["selling_price"].notna().all()
    assert daily["category"].nunique() == 2
    assert daily["event_name"].notna().any()


def test_m5_units_and_prices_join(config, m5_dir: Path):
    result = load_m5(
        config,
        raw_dir=m5_dir,
        write=False,
        subset_overrides={"n_items": 2, "n_stores": 2, "max_days": 20},
    )
    daily = result["daily_sales"]
    item1_store1 = daily.loc[daily["sku"] == "ITEM1_STORE1"].sort_values("date")
    assert item1_store1["units_sold"].sum() == sum(index % 4 for index in range(1, 21))
    expected_prices = [11.0] * 7 + [12.0] * 7 + [13.0] * 6
    assert item1_store1["selling_price"].round(4).tolist() == expected_prices
    expected_revenue = (item1_store1["units_sold"] * item1_store1["selling_price"]).sum()
    assert abs(item1_store1["revenue"].sum() - expected_revenue) < 0.01


def test_m5_clean_keeps_inventory_null(config, m5_dir: Path):
    result = load_m5(
        config,
        raw_dir=m5_dir,
        write=False,
        subset_overrides={"n_items": 2, "n_stores": 2, "max_days": 20},
    )
    cleaned, summary = clean_daily(result["daily_sales"], mode="m5", require_inventory=False)
    assert summary["rows"] == len(cleaned)
    assert cleaned["available_stock"].isna().all()
    assert set(cleaned["data_source"]) == {"m5"}


def test_m5_writes_processed_artifacts(config, m5_dir: Path):
    result = load_m5(
        config,
        raw_dir=m5_dir,
        write=True,
        subset_overrides={"n_items": 2, "n_stores": 2, "max_days": 20},
    )
    assert result["audit"]["daily_rows"] == 80
    output = Path(config["paths"]["data_processed"]) / "m5"
    for name in ("daily_sales.parquet", "sku_master.parquet", "audit.json", "provenance.json"):
        assert (output / name).exists(), name

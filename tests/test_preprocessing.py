"""Tests for cleaning, date completion, schema validation and null-inventory semantics."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.schema import SchemaValidationError, validate_daily_frame
from src.preprocessing.clean import clean_daily, fill_missing_dates


def _row(date, sku, units=1, price=10.0, stock=None, source="synthetic"):
    return {
        "date": pd.Timestamp(date),
        "sku": sku,
        "product_name": f"Product {sku}",
        "category": "Earrings",
        "units_sold": units,
        "selling_price": price,
        "discount_percentage": 0.0,
        "revenue": units * price,
        "festival_flag": 0,
        "festival_name": None,
        "promotion_flag": 0,
        "available_stock": stock,
        "stockout_flag": 0 if stock is not None else pd.NA,
        "data_source": source,
    }


def test_fill_missing_dates_and_forward_fill_price():
    frame = pd.DataFrame(
        [
            _row("2024-01-01", "A", units=2, price=10.0, stock=None, source="uci"),
            _row("2024-01-03", "A", units=1, price=12.0, stock=None, source="uci"),
            _row("2024-01-02", "B", units=4, price=20.0, stock=None, source="uci"),
            _row("2024-01-04", "B", units=0, price=20.0, stock=None, source="uci"),
        ]
    )
    filled, stats = fill_missing_dates(frame)
    assert stats["output_rows"] == 6
    assert stats["imputed_rows"] == 2
    assert (filled["date_imputed"] == 1).sum() == 2
    gap = filled.loc[(filled["sku"] == "A") & (filled["date"] == pd.Timestamp("2024-01-02"))].iloc[0]
    assert gap["units_sold"] == 0
    assert gap["selling_price"] == 10.0
    assert pd.isna(gap["available_stock"])


def test_duplicate_sku_date_is_aggregated_not_deleted():
    frame = pd.DataFrame(
        [
            _row("2024-01-01", "A", units=1, price=10.0),
            _row("2024-01-01", "A", units=2, price=10.0),
        ]
    )
    cleaned, summary = clean_daily(frame, mode="synthetic", require_inventory=False)
    assert summary["duplicates_aggregated"] == 1
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["units_sold"] == 3
    assert cleaned.iloc[0]["revenue"] == 30.0


def test_negative_units_rejected():
    frame = pd.DataFrame([_row("2024-01-01", "A", units=-1, price=10.0)])
    with pytest.raises(SchemaValidationError):
        validate_daily_frame(frame, mode="synthetic")


def test_unknown_data_source_rejected():
    frame = pd.DataFrame([_row("2024-01-01", "A")])
    frame.loc[0, "data_source"] = "unknown"
    with pytest.raises(SchemaValidationError):
        validate_daily_frame(frame)


def test_missing_required_column_rejected():
    frame = pd.DataFrame([_row("2024-01-01", "A")]).drop(columns=["revenue"])
    with pytest.raises(SchemaValidationError):
        validate_daily_frame(frame)


def test_null_inventory_is_not_coerced_to_zero():
    frame = pd.DataFrame(
        [
            _row("2024-01-01", "A", stock=None, source="uci"),
            _row("2024-01-02", "A", stock=None, source="uci"),
        ]
    )
    cleaned, _ = clean_daily(frame, mode="uci", require_inventory=False)
    assert cleaned["available_stock"].isna().all()
    assert not (cleaned["available_stock"] == 0).any()


def test_require_inventory_raises_when_all_null():
    frame = pd.DataFrame([_row("2024-01-01", "A", stock=None, source="synthetic")])
    with pytest.raises(SchemaValidationError):
        validate_daily_frame(frame, mode="synthetic", require_inventory=True)

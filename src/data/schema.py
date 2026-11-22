"""Unified daily dataset schema and validation."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

REQUIRED_COLUMNS: tuple[str, ...] = (
    "date",
    "sku",
    "product_name",
    "category",
    "units_sold",
    "selling_price",
    "discount_percentage",
    "revenue",
    "festival_flag",
    "promotion_flag",
    "available_stock",
    "data_source",
)

OPTIONAL_COLUMNS: tuple[str, ...] = (
    "festival_name",
    "stockout_flag",
    "event_name",
    "date_imputed",
)

UNIFIED_COLUMNS: tuple[str, ...] = REQUIRED_COLUMNS + OPTIONAL_COLUMNS

DATA_SOURCES = ("synthetic", "uci", "m5")


class SchemaValidationError(ValueError):
    """Raised when a dataset does not satisfy the unified daily schema."""


def empty_unified_frame() -> pd.DataFrame:
    return pd.DataFrame({column: pd.Series(dtype="object") for column in UNIFIED_COLUMNS})


def _require_columns(frame: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise SchemaValidationError(f"Missing required columns: {missing}")


def validate_daily_frame(
    frame: pd.DataFrame,
    mode: str | None = None,
    require_inventory: bool = False,
) -> dict:
    """Validate the unified schema and return a structured summary.

    Raises ``SchemaValidationError`` for structural problems. Soft issues (for
    example null inventory in real datasets) are reported, not silently fixed.
    """
    _require_columns(frame)
    if frame.empty:
        raise SchemaValidationError("Dataset is empty")

    summary: dict = {"rows": int(len(frame)), "columns": list(frame.columns)}

    dates = pd.to_datetime(frame["date"], errors="coerce")
    invalid_dates = int(dates.isna().sum())
    if invalid_dates:
        raise SchemaValidationError(f"{invalid_dates} rows have invalid dates")

    if frame["sku"].isna().any():
        raise SchemaValidationError("SKU contains null values")

    for column in ("units_sold", "revenue"):
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any():
            raise SchemaValidationError(f"{column} contains non-numeric values")
        if (values < 0).any():
            raise SchemaValidationError(f"{column} contains negative values")

    for column in ("selling_price", "discount_percentage", "available_stock"):
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            non_null = values.dropna()
            if (non_null < 0).any():
                raise SchemaValidationError(f"{column} contains negative values")

    if "discount_percentage" in frame.columns:
        discounts = pd.to_numeric(frame["discount_percentage"], errors="coerce").dropna()
        if (discounts > 100).any():
            raise SchemaValidationError("discount_percentage exceeds 100")

    sources = set(frame["data_source"].dropna().astype(str).unique())
    unknown = sources.difference(DATA_SOURCES)
    if unknown:
        raise SchemaValidationError(f"Unknown data_source values: {sorted(unknown)}")
    if mode is not None and sources and sources != {mode}:
        raise SchemaValidationError(f"Expected data_source={mode!r}, found {sorted(sources)}")

    null_inventory = int(pd.to_numeric(frame["available_stock"], errors="coerce").isna().sum())
    if require_inventory and null_inventory == len(frame):
        raise SchemaValidationError("available_stock is entirely null but inventory is required")

    summary.update(
        {
            "date_min": dates.min(),
            "date_max": dates.max(),
            "skus": int(frame["sku"].nunique()),
            "categories": int(frame["category"].nunique(dropna=True)),
            "null_inventory_rows": null_inventory,
            "null_price_rows": int(pd.to_numeric(frame["selling_price"], errors="coerce").isna().sum()),
            "data_sources": sorted(sources),
        }
    )
    return summary


def coerce_unified_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply efficient, consistent dtypes to the unified frame."""
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    for column in ("sku", "product_name", "category", "data_source", "festival_name", "event_name"):
        if column in out.columns:
            out[column] = out[column].astype("string")
    out["units_sold"] = pd.to_numeric(out["units_sold"], errors="coerce").fillna(0).astype("int32")
    out["revenue"] = pd.to_numeric(out["revenue"], errors="coerce").fillna(0.0).astype("float64").round(2)
    for column in ("selling_price", "discount_percentage", "available_stock"):
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce").astype("float64")
    out["festival_flag"] = pd.to_numeric(out["festival_flag"], errors="coerce").fillna(0).astype("int8")
    out["promotion_flag"] = pd.to_numeric(out["promotion_flag"], errors="coerce").fillna(0).astype("int8")
    if "stockout_flag" in out.columns:
        out["stockout_flag"] = pd.to_numeric(out["stockout_flag"], errors="coerce").astype("Int8")
    return out


def ensure_optional_columns(frame: pd.DataFrame, columns: Iterable[str] = OPTIONAL_COLUMNS) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = pd.NA
    return out

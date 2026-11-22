"""Unified cleaning: duplicate aggregation, date completion and dtype coercion."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.data.festivals import festival_flag_for_dates
from src.data.schema import coerce_unified_dtypes, ensure_optional_columns, validate_daily_frame


def _aggregate_duplicate_sku_dates(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    duplicated = int(frame.duplicated(subset=["sku", "date"]).sum())
    if not duplicated:
        return frame, 0
    grouped = (
        frame.groupby(["sku", "date"], as_index=False)
        .agg(
            product_name=("product_name", "first"),
            category=("category", "first"),
            units_sold=("units_sold", "sum"),
            revenue=("revenue", "sum"),
            selling_price=("selling_price", "mean"),
            discount_percentage=("discount_percentage", "mean"),
            festival_flag=("festival_flag", "max"),
            festival_name=("festival_name", "first"),
            event_name=("event_name", "first"),
            promotion_flag=("promotion_flag", "max"),
            available_stock=("available_stock", "first"),
            stockout_flag=("stockout_flag", "max"),
            data_source=("data_source", "first"),
        )
        .reset_index()
    )
    grouped["selling_price"] = np.where(
        grouped["units_sold"] > 0,
        grouped["revenue"] / grouped["units_sold"],
        grouped["selling_price"],
    )
    return grouped, duplicated


def fill_missing_dates(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Reindex each SKU onto a continuous daily calendar between its first and last sale.

    Missing observations are recorded with ``date_imputed=1``. Prices and static
    attributes are forward-filled within SKU; inventory is forward-filled only when
    it was observed (it remains null for datasets without inventory). Calendar
    labels (``festival_name`` and ``event_name``) are intentionally not
    forward-filled because they describe only the date on which the event occurs.
    """
    working = frame.copy()
    working["date"] = pd.to_datetime(working["date"]).dt.normalize()

    bounds = working.groupby("sku", observed=True)["date"].agg(["min", "max"])
    starts = bounds["min"].to_numpy()
    ends = bounds["max"].to_numpy()
    lengths = (ends - starts).astype("timedelta64[D]").astype(int) + 1
    sku_repeat = np.repeat(bounds.index.to_numpy(), lengths)
    date_values = np.concatenate(
        [pd.date_range(start, end, freq="D").to_numpy() for start, end in zip(starts, ends)]
    )
    skeleton = pd.DataFrame({"sku": sku_repeat, "date": date_values})
    skeleton["sku"] = skeleton["sku"].astype(working["sku"].dtype)

    merged = skeleton.merge(working, on=["sku", "date"], how="left", indicator=True)
    merged["date_imputed"] = (merged["_merge"] == "left_only").astype("int8")
    merged = merged.drop(columns=["_merge"])

    merged = merged.sort_values(["sku", "date"]).reset_index(drop=True)
    for column in ("product_name", "category", "selling_price", "available_stock"):
        if column in merged.columns:
            merged[column] = merged.groupby("sku", observed=True)[column].ffill()

    for column in ("product_name", "category"):
        if column in merged.columns:
            merged[column] = merged.groupby("sku", observed=True)[column].bfill()

    merged["units_sold"] = pd.to_numeric(merged["units_sold"], errors="coerce").fillna(0)
    merged["revenue"] = pd.to_numeric(merged["revenue"], errors="coerce").fillna(0.0)
    merged["festival_flag"] = pd.to_numeric(merged["festival_flag"], errors="coerce").fillna(0)
    merged["promotion_flag"] = pd.to_numeric(merged["promotion_flag"], errors="coerce").fillna(0)
    merged["data_source"] = merged["data_source"].fillna(working["data_source"].iloc[0])

    stats = {
        "input_rows": int(len(working)),
        "output_rows": int(len(merged)),
        "imputed_rows": int(merged["date_imputed"].sum()),
        "imputed_share": round(float(merged["date_imputed"].mean()), 4),
    }
    return merged, stats


def clean_daily(
    frame: pd.DataFrame,
    mode: str,
    require_inventory: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Full cleaning pass used by every data mode."""
    working = ensure_optional_columns(frame)
    working = working.dropna(subset=["sku", "date"]).copy()
    working, duplicates_aggregated = _aggregate_duplicate_sku_dates(working)
    working, date_stats = fill_missing_dates(working)

    if mode == "uci":
        working["festival_flag"] = festival_flag_for_dates(working["date"])
        working["festival_name"] = np.where(working["festival_flag"] == 1, "UK Bank Holiday", None)

    working = coerce_unified_dtypes(working)
    working = working.sort_values(["sku", "date"]).reset_index(drop=True)

    summary = validate_daily_frame(working, mode=mode, require_inventory=require_inventory)
    summary.update(
        {
            "duplicates_aggregated": duplicates_aggregated,
            **date_stats,
        }
    )
    return working, summary

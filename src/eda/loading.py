"""Parquet loaders for the EDA pipeline.

Only columns needed for analysis are read, and optional SKU subsets are applied
immediately. The processed SKU-day Parquet files are the single input source; EDA
does not regenerate or mutate the datasets.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
import pyarrow.parquet as pq

from src.utils.paths import resolve_path

DAILY_COLUMNS = (
    "date",
    "sku",
    "product_name",
    "category",
    "units_sold",
    "selling_price",
    "discount_percentage",
    "revenue",
    "festival_flag",
    "festival_name",
    "promotion_flag",
    "available_stock",
    "stockout_flag",
    "event_name",
    "date_imputed",
    "data_source",
)

OPTIONAL_FILES = {
    "synthetic": ("inventory_snapshots.parquet", "purchase_orders.parquet"),
    "uci": ("transactions_clean.parquet",),
    "m5": (),
}


def processed_directory(config: Mapping[str, Any], processed_dir: str | Path | None = None) -> Path:
    return resolve_path(processed_dir or config["paths"]["data_processed"])


def mode_directory(config: Mapping[str, Any], mode: str, processed_dir: str | Path | None = None) -> Path:
    return processed_directory(config, processed_dir) / mode


def load_daily(
    mode: str,
    config: Mapping[str, Any],
    processed_dir: str | Path | None = None,
    sku_subset: Iterable[str] | None = None,
    sku_limit: int | None = None,
    exclude_non_products: bool = True,
    return_meta: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
    """Read the unified SKU-day Parquet for one mode with a bounded column set.

    For UCI, explicit non-product codes configured in ``config.yaml`` (postage,
    manual, bad debt, fees and similar) are excluded by default. They remain in
    the processed dataset for audit, but they are not retail products and would
    distort SKU, price and ABC metrics. Set ``exclude_non_products=False`` to
    analyse every retained row.
    """
    path = mode_directory(config, mode, processed_dir) / "daily_sales.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {path}. Run python scripts/build_datasets.py --modes {mode}"
        )

    schema_names = set(pq.read_schema(path).names)
    columns = [column for column in DAILY_COLUMNS if column in schema_names]
    frame = pd.read_parquet(path, columns=columns)
    meta: dict[str, Any] = {"rows_loaded": int(len(frame))}

    sources = set(frame["data_source"].dropna().astype(str).unique()) if "data_source" in frame else {mode}
    if sources and sources != {mode}:
        raise ValueError(f"Expected data_source={mode!r}, found {sorted(sources)}")

    if mode == "uci" and exclude_non_products:
        codes = {
            str(code).upper()
            for code in (config.get("data", {}).get("uci", {}) or {}).get("non_product_codes", []) or []
        }
        if codes:
            mask = frame["sku"].astype(str).str.upper().isin(codes)
            meta["excluded_non_product_rows"] = int(mask.sum())
            meta["excluded_non_product_skus"] = int(frame.loc[mask, "sku"].nunique())
            frame = frame.loc[~mask].copy()
            meta["rows_after_non_product_exclusion"] = int(len(frame))

    if sku_subset:
        requested = {str(sku) for sku in sku_subset}
        frame = frame.loc[frame["sku"].astype(str).isin(requested)].copy()
        if frame.empty:
            raise ValueError("The requested SKU subset does not match any rows in the dataset")

    if sku_limit is not None and not sku_subset:
        limit = int(sku_limit)
        if limit <= 0:
            raise ValueError("sku_limit must be a positive integer")
        totals = frame.groupby("sku", observed=True)["units_sold"].sum().nlargest(limit)
        frame = frame.loc[frame["sku"].isin(totals.index)].copy()

    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame = frame.sort_values(["sku", "date"], kind="mergesort").reset_index(drop=True)
    meta["rows_analysed"] = int(len(frame))
    meta["skus_analysed"] = int(frame["sku"].nunique())
    if return_meta:
        return frame, meta
    return frame


def load_sku_master(
    mode: str,
    config: Mapping[str, Any],
    processed_dir: str | Path | None = None,
) -> pd.DataFrame:
    path = mode_directory(config, mode, processed_dir) / "sku_master.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def load_inventory_snapshots(
    config: Mapping[str, Any],
    processed_dir: str | Path | None = None,
) -> pd.DataFrame:
    path = mode_directory(config, "synthetic", processed_dir) / "inventory_snapshots.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def load_purchase_orders(
    config: Mapping[str, Any],
    processed_dir: str | Path | None = None,
) -> pd.DataFrame:
    path = mode_directory(config, "synthetic", processed_dir) / "purchase_orders.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def load_provenance(
    mode: str,
    config: Mapping[str, Any],
    processed_dir: str | Path | None = None,
) -> dict[str, Any]:
    path = mode_directory(config, mode, processed_dir) / "provenance.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_dataset_summary(
    mode: str,
    config: Mapping[str, Any],
    processed_dir: str | Path | None = None,
) -> dict[str, Any]:
    path = mode_directory(config, mode, processed_dir) / "dataset_summary.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)

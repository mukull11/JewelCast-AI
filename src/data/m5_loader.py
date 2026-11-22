"""M5 Forecasting Accuracy loader.

Source: https://www.kaggle.com/competitions/m5-forecasting-accuracy/data

M5 is a Walmart retail benchmark. It is **not** jewellery data and it is **not**
Voylla data. This loader deliberately selects a small, configurable subset of
item-store series before melting the wide daily matrix, keeping memory bounded on
a 16 GB CPU-only machine.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.utils.io import save_parquet, write_json
from src.utils.paths import resolve_path

ID_COLUMNS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]


def _read_calendar(path: Path) -> pd.DataFrame:
    calendar = pd.read_csv(
        path,
        dtype={"d": "string", "wm_yr_wk": "int32", "event_name_1": "string", "event_type_1": "string",
               "event_name_2": "string", "event_type_2": "string"},
    )
    calendar["date"] = pd.to_datetime(calendar["date"])
    return calendar


def _select_series(sales: pd.DataFrame, d_columns: list[str], subset: Mapping[str, Any]) -> pd.DataFrame:
    n_items = int(subset.get("n_items", 30))
    n_stores = int(subset.get("n_stores", 3))
    configured_items = subset.get("item_ids")
    configured_stores = subset.get("store_ids")

    sales = sales.copy()
    sales["total_units"] = sales[d_columns].sum(axis=1)

    if configured_items:
        items = [str(item) for item in configured_items]
    elif bool(subset.get("stratify_by_category", True)):
        item_totals = (
            sales.groupby(["cat_id", "item_id"], observed=True)["total_units"]
            .sum()
            .reset_index()
        )
        categories = item_totals["cat_id"].unique().tolist()
        per_category = max(1, n_items // max(len(categories), 1))
        picked: list[str] = []
        for category in categories:
            picked.extend(
                item_totals.loc[item_totals["cat_id"] == category]
                .nlargest(per_category, "total_units")["item_id"]
                .astype(str)
                .tolist()
            )
        if len(picked) < n_items:
            remaining = (
                item_totals.loc[~item_totals["item_id"].isin(picked)]
                .nlargest(n_items - len(picked), "total_units")["item_id"]
                .astype(str)
                .tolist()
            )
            picked.extend(remaining)
        items = list(dict.fromkeys(picked))[:n_items]
    else:
        items = (
            sales.groupby("item_id", observed=True)["total_units"]
            .sum()
            .sort_values(ascending=False)
            .head(n_items)
            .index.astype(str)
            .tolist()
        )

    if configured_stores:
        stores = [str(store) for store in configured_stores]
    elif bool(subset.get("stratify_by_state", True)):
        store_totals = (
            sales.groupby(["state_id", "store_id"], observed=True)["total_units"]
            .sum()
            .reset_index()
        )
        states = store_totals["state_id"].unique().tolist()
        picked_stores: list[str] = []
        for state in states:
            picked_stores.extend(
                store_totals.loc[store_totals["state_id"] == state]
                .nlargest(1, "total_units")["store_id"]
                .astype(str)
                .tolist()
            )
        if len(picked_stores) < n_stores:
            remaining = (
                store_totals.loc[~store_totals["store_id"].isin(picked_stores)]
                .nlargest(n_stores - len(picked_stores), "total_units")["store_id"]
                .astype(str)
                .tolist()
            )
            picked_stores.extend(remaining)
        stores = list(dict.fromkeys(picked_stores))[:n_stores]
    else:
        stores = (
            sales.groupby("store_id", observed=True)["total_units"]
            .sum()
            .sort_values(ascending=False)
            .head(n_stores)
            .index.astype(str)
            .tolist()
        )

    selected = sales.loc[sales["item_id"].isin(items) & sales["store_id"].isin(stores)].copy()
    selected = selected.drop(columns=["total_units"])
    return selected


def _read_prices_for_subset(
    path: Path,
    items: set[str],
    stores: set[str],
    chunk_size: int = 1_000_000,
) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    reader = pd.read_csv(
        path,
        usecols=["store_id", "item_id", "wm_yr_wk", "sell_price"],
        dtype={"store_id": "string", "item_id": "string", "wm_yr_wk": "int32", "sell_price": "float32"},
        chunksize=chunk_size,
    )
    for chunk in reader:
        mask = chunk["item_id"].isin(items) & chunk["store_id"].isin(stores)
        if mask.any():
            chunks.append(chunk.loc[mask])
    if not chunks:
        return pd.DataFrame(columns=["store_id", "item_id", "wm_yr_wk", "sell_price"])
    return pd.concat(chunks, ignore_index=True)


def load_m5(
    config: Mapping[str, Any],
    raw_dir: str | Path | None = None,
    processed_dir: str | Path | None = None,
    write: bool = True,
    subset_overrides: Mapping[str, Any] | None = None,
) -> dict:
    """Load a bounded M5 subset and return unified daily + SKU master frames."""
    m5 = config.get("data", {}).get("m5", {}) or {}
    subset = dict(m5.get("subset", {}) or {})
    if subset_overrides:
        subset.update(dict(subset_overrides))

    raw = resolve_path(raw_dir or config["paths"]["data_raw"])
    sales_path = raw / m5.get("sales_file", "sales_train_validation.csv")
    calendar_path = raw / m5.get("calendar_file", "calendar.csv")
    prices_path = raw / m5.get("prices_file", "sell_prices.csv")
    for path in (sales_path, calendar_path, prices_path):
        if not path.exists():
            raise FileNotFoundError(f"M5 file not found: {path}. Place the M5 CSV files in data/raw/.")

    header = pd.read_csv(sales_path, nrows=0)
    d_columns = [column for column in header.columns if column.startswith("d_")]
    if not d_columns:
        raise ValueError("M5 sales file has no d_* day columns")

    sales = pd.read_csv(
        sales_path,
        usecols=ID_COLUMNS + d_columns,
        dtype={**{column: "string" for column in ID_COLUMNS}, **{column: "int16" for column in d_columns}},
    )
    raw_rows, raw_columns = sales.shape

    selected = _select_series(sales, d_columns, subset)
    selected_items = set(selected["item_id"].astype(str).unique())
    selected_stores = set(selected["store_id"].astype(str).unique())

    long = selected.melt(
        id_vars=ID_COLUMNS,
        value_vars=d_columns,
        var_name="d",
        value_name="units_sold",
    )
    long["units_sold"] = pd.to_numeric(long["units_sold"], errors="coerce").fillna(0).astype("int32")
    long["item_id"] = long["item_id"].astype("string")
    long["store_id"] = long["store_id"].astype("string")

    calendar = _read_calendar(calendar_path)
    calendar_slim = calendar[
        ["d", "date", "wm_yr_wk", "event_name_1", "event_type_1", "event_name_2", "event_type_2"]
    ]
    long = long.merge(calendar_slim, on="d", how="left")
    long["date"] = pd.to_datetime(long["date"], errors="coerce")
    long = long.dropna(subset=["date"])

    max_days = int(subset.get("max_days", 730))
    max_date = long["date"].max()
    cutoff = max_date - pd.Timedelta(days=max_days - 1)
    long = long.loc[long["date"] >= cutoff].copy()

    prices = _read_prices_for_subset(prices_path, selected_items, selected_stores)
    long = long.merge(
        prices,
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left",
    )

    long["sku"] = long["item_id"].astype(str) + "_" + long["store_id"].astype(str)
    long = long.sort_values(["sku", "date"]).reset_index(drop=True)
    long["selling_price"] = long.groupby("sku", observed=True)["sell_price"].ffill()
    missing_price = int(long["selling_price"].isna().sum())
    long["revenue"] = (long["units_sold"].astype("float64") * long["selling_price"].fillna(0.0)).round(2)

    long["product_name"] = long["item_id"].astype(str)
    long["category"] = long["cat_id"].astype(str)
    long["discount_percentage"] = np.nan
    long["festival_flag"] = 0
    long["festival_name"] = None
    long["event_name"] = long["event_name_1"].where(long["event_name_1"].notna(), long["event_name_2"])
    long["promotion_flag"] = 0
    long["available_stock"] = np.nan
    long["stockout_flag"] = pd.NA
    long["data_source"] = "m5"

    daily = long[
        [
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
            "event_name",
            "promotion_flag",
            "available_stock",
            "stockout_flag",
            "data_source",
        ]
    ].copy()

    sku_master = (
        long.groupby(["sku", "item_id", "store_id", "dept_id", "cat_id", "state_id"], observed=True)
        .agg(
            total_units=("units_sold", "sum"),
            total_revenue=("revenue", "sum"),
            avg_price=("selling_price", "mean"),
            first_date=("date", "min"),
            last_date=("date", "max"),
        )
        .reset_index()
    )
    sku_master["product_name"] = sku_master["item_id"].astype(str)
    sku_master["category"] = sku_master["cat_id"].astype(str)
    sku_master["data_source"] = "m5"

    audit = {
        "mode": "m5",
        "source_file": str(sales_path),
        "provenance": "Walmart M5 Forecasting Accuracy benchmark. Not jewellery data and not Voylla data.",
        "raw_rows": int(raw_rows),
        "raw_columns": int(raw_columns),
        "day_columns": len(d_columns),
        "selected_series": int(len(selected)),
        "selected_items": int(len(selected_items)),
        "selected_stores": int(len(selected_stores)),
        "selected_item_ids": sorted(selected_items),
        "selected_store_ids": sorted(selected_stores),
        "daily_rows": int(len(daily)),
        "date_min": daily["date"].min(),
        "date_max": daily["date"].max(),
        "days": int(daily["date"].nunique()),
        "skus": int(daily["sku"].nunique()),
        "missing_price_rows": missing_price,
        "zero_sales_rows": int((daily["units_sold"] == 0).sum()),
        "event_days": int(daily["event_name"].notna().sum()),
        "subset": {
            "n_items": int(subset.get("n_items", 30)),
            "n_stores": int(subset.get("n_stores", 3)),
            "max_days": max_days,
        },
        "inventory_note": "M5 has no store inventory records. available_stock is null; lost demand cannot be identified.",
    }

    result = {"daily_sales": daily, "sku_master": sku_master, "audit": audit}

    if write:
        output = resolve_path(processed_dir or (resolve_path(config["paths"]["data_processed"]) / "m5"))
        output.mkdir(parents=True, exist_ok=True)
        save_parquet(daily, output / "daily_sales.parquet")
        save_parquet(sku_master, output / "sku_master.parquet")
        write_json(audit, output / "audit.json")
        write_json(
            {
                "mode": "m5",
                "source": "M5 Forecasting Accuracy (Walmart)",
                "url": "https://www.kaggle.com/competitions/m5-forecasting-accuracy/data",
                "provenance": audit["provenance"],
                "subset": audit["subset"],
                "inventory_note": audit["inventory_note"],
                "event_note": "event_name holds US calendar events from M5; it is not an Indian festival flag.",
            },
            output / "provenance.json",
        )

    return result

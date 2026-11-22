"""UCI Online Retail II loader.

Source: https://archive.ics.uci.edu/dataset/502/online+retail+ii

This is a UK online-retail transaction dataset (GBP). It is **not** Indian retail
data and it is **not** a jewellery-only dataset. Jewellery categories are derived
with transparent, documented keyword rules and the resulting SKU counts are
validated before the subset is used.

Suspicious records (cancellations, returns, duplicates, invalid prices) are never
silently deleted: each row receives an ``exclusion_reason`` and the cleaned
transaction-level table is retained for audit.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.data.festivals import festival_flag_for_dates
from src.utils.io import save_parquet, write_json
from src.utils.paths import resolve_path

CATEGORY_ORDER = (
    "Jewellery Sets",
    "Earrings",
    "Necklaces",
    "Rings",
    "Bracelets",
    "Bangles",
    "Anklets",
    "Pendants",
    "Other Jewellery",
)

NON_PRODUCT_HINTS = ("postage", "bank", "manual", "discount", "adjust", "amazon", "fee", "carriage")


def _normalise_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip().lower()


def classify_jewellery(
    description: Any,
    rules: Mapping[str, list[str]],
    exclusions: list[str] | None = None,
) -> str | None:
    """Return a jewellery category for a description, or ``None``.

    Rules are applied with word boundaries, after checking exclusion phrases such
    as ``jewellery box`` or ``key ring``. Longer/more specific categories are
    checked before generic ones.
    """
    text = _normalise_text(description)
    if not text:
        return None

    for phrase in exclusions or []:
        if _normalise_text(phrase) and _normalise_text(phrase) in text:
            return None

    for category in CATEGORY_ORDER:
        for keyword in rules.get(category, []) or []:
            keyword_text = _normalise_text(keyword)
            if not keyword_text:
                continue
            if re.search(rf"\b{re.escape(keyword_text)}(?:s|es)?\b", text):
                return category
    return None


def _is_non_product(sku: str, description: Any, config: Mapping[str, Any]) -> bool:
    uci = config.get("data", {}).get("uci", {}) or {}
    codes = {str(code).upper() for code in uci.get("non_product_codes", []) or []}
    if str(sku).upper() in codes:
        return True
    text = _normalise_text(description)
    return any(hint in text for hint in NON_PRODUCT_HINTS)


def _read_sheets(path: Path, sheets: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for sheet in sheets:
        frame = pd.read_excel(
            path,
            sheet_name=sheet,
            engine="openpyxl",
            dtype={
                "InvoiceNo": "string",
                "Invoice": "string",
                "StockCode": "string",
                "Description": "string",
                "CustomerID": "string",
                "Customer ID": "string",
                "Country": "string",
            },
        )
        frame = frame.rename(
            columns={
                "InvoiceNo": "invoice_no",
                "Invoice": "invoice_no",
                "StockCode": "sku",
                "Description": "description",
                "Quantity": "quantity",
                "InvoiceDate": "invoice_date",
                "UnitPrice": "unit_price",
                "Price": "unit_price",
                "CustomerID": "customer_id",
                "Customer ID": "customer_id",
                "Country": "country",
            }
        )
        expected = [
            "invoice_no",
            "sku",
            "description",
            "quantity",
            "invoice_date",
            "unit_price",
            "customer_id",
            "country",
        ]
        missing = [column for column in expected if column not in frame.columns]
        if missing:
            raise ValueError(f"Sheet {sheet!r} is missing columns: {missing}")
        frame = frame[expected]
        frame["sheet_year"] = sheet
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    return combined


def load_uci_transactions(
    config: Mapping[str, Any],
    path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Read both sheets and annotate every row with audit flags."""
    uci = config.get("data", {}).get("uci", {}) or {}
    source = resolve_path(path or (resolve_path(config["paths"]["data_raw"]) / uci.get("excel_filename", "online_retail_II.xlsx")))
    if not source.exists():
        raise FileNotFoundError(
            f"UCI file not found at {source}. Place online_retail_II.xlsx in data/raw/."
        )

    sheets = list(uci.get("sheets") or ["Year 2009-2010", "Year 2010-2011"])
    frame = _read_sheets(source, sheets)

    frame["invoice_date"] = pd.to_datetime(frame["invoice_date"], errors="coerce")
    frame["quantity"] = pd.to_numeric(frame["quantity"], errors="coerce")
    frame["unit_price"] = pd.to_numeric(frame["unit_price"], errors="coerce")
    frame["sku"] = frame["sku"].astype("string").str.strip()
    frame["invoice_no"] = frame["invoice_no"].astype("string").str.strip()
    frame["description"] = frame["description"].astype("string").str.strip()

    frame["is_cancellation"] = frame["invoice_no"].str.upper().str.startswith("C", na=False)
    frame["is_return"] = frame["quantity"].fillna(0) < 0
    frame["is_nonpositive_price"] = frame["unit_price"].fillna(0) <= 0
    frame["is_missing_sku"] = frame["sku"].isna() | (frame["sku"].str.len() == 0)
    frame["is_missing_date"] = frame["invoice_date"].isna()
    frame["is_missing_description"] = frame["description"].isna() | (frame["description"].str.len() == 0)

    duplicate_mask = frame.duplicated(
        subset=["invoice_no", "sku", "description", "quantity", "invoice_date", "unit_price", "customer_id", "country"],
        keep="first",
    )
    frame["is_duplicate"] = duplicate_mask

    frame["is_non_product"] = [
        _is_non_product(sku, description, config)
        for sku, description in zip(frame["sku"].fillna(""), frame["description"])
    ]

    conditions = [
        frame["is_missing_sku"],
        frame["is_missing_date"],
        frame["is_cancellation"],
        frame["is_return"],
        frame["is_nonpositive_price"],
        frame["is_duplicate"],
    ]
    reasons = [
        "missing_sku",
        "invalid_date",
        "cancellation",
        "negative_quantity_return",
        "nonpositive_unit_price",
        "duplicate_transaction",
    ]
    frame["exclusion_reason"] = np.select(conditions, reasons, default=None)
    frame["is_valid_sale"] = frame["exclusion_reason"].isna()

    frame["revenue"] = frame["quantity"].fillna(0) * frame["unit_price"].fillna(0)

    audit = {
        "mode": "uci",
        "source_file": str(source),
        "sheets": sheets,
        "currency": uci.get("currency", "GBP"),
        "provenance": "UCI Online Retail II - UK online retail transactions in GBP. Not Indian data and not jewellery-only.",
        "raw_rows": int(len(frame)),
        "duplicate_rows": int(frame["is_duplicate"].sum()),
        "missing_description_rows": int(frame["is_missing_description"].sum()),
        "missing_customer_id_rows": int(frame["customer_id"].isna().sum()),
        "cancellation_rows": int(frame["is_cancellation"].sum()),
        "return_rows_negative_quantity": int(frame["is_return"].sum()),
        "nonpositive_price_rows": int(frame["is_nonpositive_price"].sum()),
        "missing_sku_rows": int(frame["is_missing_sku"].sum()),
        "invalid_date_rows": int(frame["is_missing_date"].sum()),
        "non_product_rows": int(frame["is_non_product"].sum()),
        "excluded_rows": int((~frame["is_valid_sale"]).sum()),
        "included_rows": int(frame["is_valid_sale"].sum()),
        "exclusion_counts": {
            reason: int((frame["exclusion_reason"] == reason).sum()) for reason in reasons
        },
        "date_min": frame["invoice_date"].min(),
        "date_max": frame["invoice_date"].max(),
    }
    return frame, audit


def build_uci_daily(
    transactions: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Aggregate valid sales to SKU-day level and build the SKU master."""
    uci = config.get("data", {}).get("uci", {}) or {}
    rules = uci.get("jewellery_rules", {}) or {}
    exclusions = list(uci.get("jewellery_exclusions", []) or [])

    valid = transactions.loc[transactions["is_valid_sale"]].copy()
    if valid.empty:
        raise ValueError("No valid UCI sales rows after exclusions")

    valid["date"] = valid["invoice_date"].dt.normalize()
    valid["description_clean"] = valid["description"].fillna("UNKNOWN")

    unique_descriptions = valid["description_clean"].dropna().unique()
    description_to_category = {
        description: classify_jewellery(description, rules, exclusions)
        for description in unique_descriptions
    }
    valid["derived_category"] = valid["description_clean"].map(description_to_category)

    def _mode_or_default(series: pd.Series, default: str) -> str:
        cleaned = series.dropna()
        if cleaned.empty:
            return default
        modes = cleaned.mode()
        return str(modes.iloc[0]) if not modes.empty else default

    product_names = valid.groupby("sku", observed=True)["description_clean"].agg(
        lambda series: _mode_or_default(series, "UNKNOWN")
    )
    categories = valid.groupby("sku", observed=True)["derived_category"].agg(
        lambda series: _mode_or_default(series, "Other")
    )
    sku_meta = valid.groupby("sku", observed=True).agg(
        total_units=("quantity", "sum"),
        total_revenue=("revenue", "sum"),
        first_sale_date=("date", "min"),
        last_sale_date=("date", "max"),
        avg_unit_price=("unit_price", "mean"),
    )
    sku_meta["product_name"] = product_names
    sku_meta["category"] = categories
    sku_meta["is_jewellery"] = ~sku_meta["category"].isin(["Other"])
    sku_meta["is_product"] = [
        not _is_non_product(sku, name, config)
        for sku, name in zip(sku_meta.index, sku_meta["product_name"])
    ]
    sku_meta["data_source"] = "uci"
    sku_meta = sku_meta.reset_index()

    grouped = valid.groupby(["date", "sku"], observed=True).agg(
        units_sold=("quantity", "sum"),
        revenue=("revenue", "sum"),
    ).reset_index()
    grouped = grouped.merge(
        sku_meta[["sku", "product_name", "category"]],
        on="sku",
        how="left",
    )
    grouped["selling_price"] = np.where(
        grouped["units_sold"] > 0,
        grouped["revenue"] / grouped["units_sold"],
        np.nan,
    )
    grouped["discount_percentage"] = np.nan
    grouped["festival_flag"] = festival_flag_for_dates(grouped["date"])
    grouped["festival_name"] = np.where(
        grouped["festival_flag"] == 1, "UK Bank Holiday", None
    )
    grouped["promotion_flag"] = 0
    grouped["available_stock"] = np.nan
    grouped["stockout_flag"] = pd.NA
    grouped["data_source"] = "uci"

    daily = grouped[
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
            "promotion_flag",
            "available_stock",
            "stockout_flag",
            "data_source",
        ]
    ].copy()

    jewellery_skus = int(sku_meta["is_jewellery"].sum())
    product_skus = int(sku_meta["is_product"].sum())
    audit = {
        "valid_sales_rows": int(len(valid)),
        "daily_rows": int(len(daily)),
        "skus": int(sku_meta["sku"].nunique()),
        "product_skus": product_skus,
        "non_product_skus": int(len(sku_meta) - product_skus),
        "jewellery_skus": jewellery_skus,
        "jewellery_share_of_skus": round(jewellery_skus / max(len(sku_meta), 1), 4),
        "category_sku_counts": sku_meta["category"].value_counts().to_dict(),
        "jewellery_validation": (
            "VALID: jewellery subset has at least 20 SKUs"
            if jewellery_skus >= 20
            else "WARNING: fewer than 20 jewellery SKUs; subset is too small for a jewellery-specific claim"
        ),
        "date_min": daily["date"].min(),
        "date_max": daily["date"].max(),
        "note": "Full UCI daily sales include non-jewellery categories. Filter category != 'Other' for the jewellery view.",
    }
    return daily, sku_meta, audit


def load_uci(
    config: Mapping[str, Any],
    path: str | Path | None = None,
    processed_dir: str | Path | None = None,
    write: bool = True,
) -> dict:
    """Full UCI pipeline: read, audit, aggregate and optionally persist."""
    transactions, read_audit = load_uci_transactions(config, path=path)
    daily, sku_master, build_audit = build_uci_daily(transactions, config)
    audit = {**read_audit, **build_audit}

    result = {
        "daily_sales": daily,
        "sku_master": sku_master,
        "transactions": transactions,
        "audit": audit,
    }

    if write:
        output = resolve_path(processed_dir or (resolve_path(config["paths"]["data_processed"]) / "uci"))
        output.mkdir(parents=True, exist_ok=True)
        save_parquet(daily, output / "daily_sales.parquet")
        save_parquet(sku_master, output / "sku_master.parquet")
        save_parquet(transactions, output / "transactions_clean.parquet")
        write_json(audit, output / "audit.json")
        write_json(
            {
                "mode": "uci",
                "source": "UCI Online Retail II",
                "url": "https://archive.ics.uci.edu/dataset/502/online+retail+ii",
                "currency": "GBP",
                "provenance": audit["provenance"],
                "jewellery_classification": "Keyword rules on product descriptions; see config.yaml data.uci.jewellery_rules.",
                "inventory_note": "UCI has no inventory records. available_stock is null; lost demand cannot be identified.",
            },
            output / "provenance.json",
        )

    return result

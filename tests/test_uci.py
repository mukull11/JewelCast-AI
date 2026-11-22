"""Tests for the UCI Online Retail II loader."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.uci_loader import classify_jewellery, load_uci
from src.preprocessing.clean import clean_daily


RULES = {
    "Jewellery Sets": ["jewellery set", "necklace and earring"],
    "Earrings": ["earring"],
    "Necklaces": ["necklace"],
    "Rings": ["ring"],
    "Bracelets": ["bracelet"],
    "Bangles": ["bangle"],
    "Anklets": ["anklet"],
    "Pendants": ["pendant"],
    "Other Jewellery": ["brooch", "charm", "jewellery"],
}
EXCLUSIONS = ["key ring", "jewellery box"]


def test_classify_jewellery_rules():
    assert classify_jewellery("SILVER EARRINGS", RULES, EXCLUSIONS) == "Earrings"
    assert classify_jewellery("GOLD NECKLACE", RULES, EXCLUSIONS) == "Necklaces"
    assert classify_jewellery("DIAMOND RING", RULES, EXCLUSIONS) == "Rings"
    assert classify_jewellery("KUNDAN BANGLE", RULES, EXCLUSIONS) == "Bangles"
    assert classify_jewellery("NECKLACE AND EARRING SET", RULES, EXCLUSIONS) == "Jewellery Sets"
    assert classify_jewellery("KEY RING", RULES, EXCLUSIONS) is None
    assert classify_jewellery("JEWELLERY BOX", RULES, EXCLUSIONS) is None
    assert classify_jewellery("POSTAGE", RULES, EXCLUSIONS) is None


def test_load_small_uci(config, uci_workbook: Path):
    result = load_uci(config, path=uci_workbook, write=False)
    audit = result["audit"]
    daily = result["daily_sales"]

    assert audit["raw_rows"] == 13
    assert audit["cancellation_rows"] == 1
    assert audit["return_rows_negative_quantity"] == 1
    assert audit["duplicate_rows"] == 1
    assert audit["missing_description_rows"] == 1
    assert audit["non_product_rows"] >= 1
    assert audit["jewellery_skus"] >= 5
    assert "WARNING" in audit["jewellery_validation"]

    assert set(daily["data_source"]) == {"uci"}
    assert daily["available_stock"].isna().all()
    assert daily["discount_percentage"].isna().all()
    assert daily["revenue"].ge(0).all()

    cancelled = daily.loc[daily["sku"] == "JEW01"]
    assert cancelled["units_sold"].sum() == 2  # cancellation excluded

    duplicate = daily.loc[daily["sku"] == "JEW03"]
    assert duplicate["units_sold"].sum() == 4  # duplicate excluded

    key_ring = result["sku_master"].set_index("sku").loc["JEW05"]
    assert not bool(key_ring["is_jewellery"])


def test_uci_clean_fills_dates_and_keeps_inventory_null(config, uci_workbook: Path):
    result = load_uci(config, path=uci_workbook, write=False)
    cleaned, summary = clean_daily(result["daily_sales"], mode="uci", require_inventory=False)
    assert summary["imputed_rows"] > 0
    assert (cleaned["date_imputed"] == 1).any()
    assert cleaned["available_stock"].isna().all()
    assert set(cleaned["data_source"]) == {"uci"}
    # UK bank holiday on 2010-12-27 should be flagged on imputed/observed rows
    holiday = cleaned.loc[cleaned["date"] == pd.Timestamp("2010-12-27")]
    assert not holiday.empty
    assert (holiday["festival_flag"] == 1).all()


def test_uci_writes_processed_artifacts(config, uci_workbook: Path):
    result = load_uci(config, path=uci_workbook, write=True)
    assert result["audit"]["included_rows"] > 0
    output = Path(config["paths"]["data_processed"]) / "uci"
    for name in (
        "daily_sales.parquet",
        "sku_master.parquet",
        "transactions_clean.parquet",
        "audit.json",
        "provenance.json",
    ):
        assert (output / name).exists(), name
    transactions = pd.read_parquet(output / "transactions_clean.parquet")
    assert "exclusion_reason" in transactions.columns
    assert "is_valid_sale" in transactions.columns

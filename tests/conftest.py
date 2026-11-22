"""Shared pytest fixtures for JewelCast AI."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.synthetic import generate_synthetic_dataset
from src.utils.config import load_config


@pytest.fixture()
def config(tmp_path: Path) -> dict:
    return load_config(
        overrides={
            "paths": {
                "data_raw": str(tmp_path / "raw"),
                "data_processed": str(tmp_path / "processed"),
                "data_synthetic": str(tmp_path / "synthetic"),
                "data_external": str(tmp_path / "external"),
                "models": str(tmp_path / "models"),
                "reports": str(tmp_path / "reports"),
                "logs": str(tmp_path / "logs"),
            }
        }
    )


@pytest.fixture()
def synthetic_overrides() -> dict:
    return {
        "n_skus": 24,
        "start_date": "2024-01-01",
        "end_date": "2024-04-30",
        "seed": 123,
        "cold_start_fraction": 0.25,
    }


@pytest.fixture()
def synthetic_result(config, synthetic_overrides):
    return generate_synthetic_dataset(config, output_dir=None, overrides=synthetic_overrides)


@pytest.fixture()
def uci_workbook(tmp_path: Path) -> Path:
    path = tmp_path / "online_retail_II.xlsx"
    columns = [
        "Invoice",
        "StockCode",
        "Description",
        "Quantity",
        "InvoiceDate",
        "Price",
        "Customer ID",
        "Country",
    ]
    first_sheet = pd.DataFrame(
        [
            ["536365", "85123A", "WHITE HANGING HEART T-LIGHT HOLDER", 6, "2010-12-01 08:26:00", 2.55, "17850", "United Kingdom"],
            ["536365", "JEW01", "GOLD NECKLACE", 2, "2010-12-01 08:26:00", 25.0, "17850", "United Kingdom"],
            ["536365", "JEW02", "SILVER EARRINGS", 3, "2010-12-01 08:26:00", 12.5, "17850", "United Kingdom"],
            ["C536366", "JEW01", "GOLD NECKLACE", -1, "2010-12-01 09:00:00", 25.0, "17850", "United Kingdom"],
            ["536367", "JEW03", "KUNDAN BANGLE", 4, "2010-12-02 10:00:00", 18.0, "17851", "United Kingdom"],
            ["536374", "JEW08", "GOLD ANKLET", 1, "2010-12-01 10:00:00", 15.0, "17857", "United Kingdom"],
        ],
        columns=columns,
    )
    second_sheet = pd.DataFrame(
        [
            ["536367", "JEW03", "KUNDAN BANGLE", 4, "2010-12-02 10:00:00", 18.0, "17851", "United Kingdom"],
            ["536368", "POST", "POSTAGE", 1, "2010-12-02 11:00:00", 10.0, "17852", "United Kingdom"],
            ["536369", "JEW04", None, 2, "2010-12-03 12:00:00", 0.0, "17853", "United Kingdom"],
            ["536370", "JEW05", "KEY RING", 2, "2010-12-03 13:00:00", 5.0, "17854", "United Kingdom"],
            ["536371", "JEW06", "DIAMOND RING", 1, "2010-12-27 14:00:00", 45.0, "17855", "United Kingdom"],
            ["536372", "JEW07", "PEARL BRACELET", 2, "2010-12-27 15:00:00", 30.0, "17856", "United Kingdom"],
            ["536373", "JEW08", "GOLD ANKLET", 1, "2010-12-05 10:00:00", 15.0, "17857", "United Kingdom"],
        ],
        columns=columns,
    )
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        first_sheet.to_excel(writer, sheet_name="Year 2009-2010", index=False)
        second_sheet.to_excel(writer, sheet_name="Year 2010-2011", index=False)
    return path


@pytest.fixture()
def m5_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "m5"
    directory.mkdir()
    days = pd.date_range("2020-01-01", periods=20, freq="D")
    calendar = pd.DataFrame(
        {
            "date": days.strftime("%Y-%m-%d"),
            "wm_yr_wk": [1] * 7 + [2] * 7 + [3] * 6,
            "weekday": days.day_name(),
            "wday": days.dayofweek + 1,
            "month": days.month,
            "year": days.year,
            "d": [f"d_{index}" for index in range(1, 21)],
            "event_name_1": [None] * 19 + ["NewYear"],
            "event_type_1": [None] * 19 + ["Holiday"],
            "event_name_2": [None] * 20,
            "event_type_2": [None] * 20,
            "snap_CA": 0,
            "snap_TX": 0,
            "snap_WI": 0,
        }
    )
    calendar.to_csv(directory / "calendar.csv", index=False)

    sales_rows = []
    for item in ("ITEM1", "ITEM2"):
        for store in ("STORE1", "STORE2"):
            row = {
                "id": f"{item}_{store}_validation",
                "item_id": item,
                "dept_id": "DEPT1",
                "cat_id": "CAT1" if item == "ITEM1" else "CAT2",
                "store_id": store,
                "state_id": "CA" if store == "STORE1" else "TX",
            }
            for index in range(1, 21):
                row[f"d_{index}"] = (index % 4) if item == "ITEM1" else (index % 3)
            sales_rows.append(row)
    pd.DataFrame(sales_rows).to_csv(directory / "sales_train_validation.csv", index=False)

    price_rows = []
    for item in ("ITEM1", "ITEM2"):
        for store in ("STORE1", "STORE2"):
            for week in (1, 2, 3):
                price_rows.append(
                    {
                        "store_id": store,
                        "item_id": item,
                        "wm_yr_wk": week,
                        "sell_price": 10.0 + week + (1 if item == "ITEM2" else 0),
                    }
                )
    pd.DataFrame(price_rows).to_csv(directory / "sell_prices.csv", index=False)
    return directory

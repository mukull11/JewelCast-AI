"""Tests for inventory position, purchase orders and inventory-state alignment."""

from __future__ import annotations

import pandas as pd

from src.inventory.data_builder import (
    InventoryData,
    build_inventory_state,
    clean_purchase_orders,
    incoming_purchase_order_summary,
    pending_purchase_orders,
)
from src.inventory.inventory_position import incoming_stock, inventory_position


def _purchase_orders() -> pd.DataFrame:
    rows = [
        ("PO1", "A", "2024-01-01", "2024-01-10", 10, "received", 9),
        ("PO1", "A", "2024-01-01", "2024-01-10", 10, "received", 9),
        ("PO2", "A", "2024-01-02", "2024-01-06", 5, "open", 4),
        ("PO3", "A", "2024-01-01", "2024-01-03", 7, "received", 2),
        ("PO4", "A", "2024-01-01", "2024-01-20", 100, "cancelled", 19),
        ("PO5", "B", "2024-01-04", "2024-01-30", 20, "open", 26),
        ("PO6", "A", "2024-01-01", "2024-01-04", 3, "open", 3),
        ("PO7", "A", "2024-01-06", "2024-01-15", 9, "open", 9),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "po_id",
            "sku",
            "order_date",
            "expected_arrival_date",
            "quantity",
            "status",
            "supplier_lead_time_days",
        ],
    )


def test_inventory_position_formula() -> None:
    assert inventory_position(20, 5, 3) == 22
    assert inventory_position(0, 0, 4) == 0
    assert inventory_position(10, 2) == 12


def test_purchase_orders_are_deduplicated_and_not_double_counted() -> None:
    orders, meta = clean_purchase_orders(_purchase_orders())
    assert meta["duplicate_po_rows_dropped"] == 1
    assert meta["cancelled_purchase_orders_dropped"] == 1
    pending = pending_purchase_orders(orders, "2024-01-05")
    assert "PO4" not in set(pending["po_id"])
    assert pending.loc[pending["po_id"] == "PO1", "quantity"].sum() == 10
    assert incoming_stock(orders, "2024-01-05", "A") == 18  # PO1 + PO2 + overdue PO6
    assert incoming_stock(orders, "2024-01-05", "B") == 20
    # PO7 is ordered in the future and must not be counted
    assert "PO7" not in set(pending["po_id"])


def test_incoming_summary_due_within_lead_time() -> None:
    orders, _ = clean_purchase_orders(_purchase_orders())
    summary = incoming_purchase_order_summary(orders, "2024-01-05", pd.Series({"A": 7, "B": 30}))
    row_a = summary.set_index("sku").loc["A"]
    assert row_a["incoming_stock"] == 18
    assert row_a["incoming_due_within_lead_time"] == 18
    assert row_a["incoming_overdue_qty"] == 3
    row_b = summary.set_index("sku").loc["B"]
    assert row_b["incoming_stock"] == 20
    assert row_b["incoming_due_within_lead_time"] == 20
    assert row_b["incoming_po_count"] == 1


def test_inventory_state_uses_start_of_day_stock_and_flags_missing() -> None:
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    daily = pd.DataFrame(
        {
            "date": list(dates) * 2,
            "sku": ["A"] * len(dates) + ["B"] * len(dates),
            "product_name": ["A"] * len(dates) + ["B"] * len(dates),
            "category": ["Earrings"] * len(dates) + ["Rings"] * len(dates),
            "units_sold": [1] * (2 * len(dates)),
            "available_stock": [20] * len(dates) + [5] * 5 + [None] * 5,
            "stockout_flag": [0] * (2 * len(dates)),
            "data_source": ["synthetic"] * (2 * len(dates)),
        }
    )
    master = pd.DataFrame(
        {
            "sku": ["A", "B"],
            "category": ["Earrings", "Rings"],
            "launch_date": ["2024-01-01", "2024-01-01"],
            "supplier_lead_time_days": [7, None],
        }
    )
    data = InventoryData(
        mode="synthetic",
        daily=daily,
        sku_master=master,
        purchase_orders=pd.DataFrame(),
        snapshots=pd.DataFrame(),
    )
    state = build_inventory_state(data, "2024-01-08", default_lead_time_days=14)
    by_sku = state.set_index("sku")
    assert by_sku.loc["A", "current_stock"] == 20
    assert by_sku.loc["A", "lead_time_days"] == 7
    assert not by_sku.loc["A", "lead_time_defaulted"]
    assert by_sku.loc["B", "current_stock"] == 5
    assert by_sku.loc["B", "lead_time_days"] == 14
    assert by_sku.loc["B", "lead_time_defaulted"]
    assert by_sku.loc["B", "stock_stale"]
    assert "stock_from_earlier_date" in by_sku.loc["B", "state_flags"]

def test_clean_daily_rejects_missing_inventory_and_unsupported_modes() -> None:
    import pytest

    from src.inventory.data_builder import _clean_daily, load_inventory_data

    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "sku": ["A", "A"],
            "product_name": ["A", "A"],
            "category": ["Earrings", "Earrings"],
            "units_sold": [1, 1],
            "available_stock": [None, None],
            "stockout_flag": [0, 0],
            "data_source": ["uci", "uci"],
        }
    )
    with pytest.raises(ValueError, match="available_stock is entirely null"):
        _clean_daily(frame, "uci")
    with pytest.raises(ValueError, match="supported only for the synthetic dataset"):
        load_inventory_data("m5", {})


def test_inventory_state_flags_missing_stock_without_silent_zero_claim() -> None:
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    daily = pd.DataFrame(
        {
            "date": list(dates) * 2,
            "sku": ["A"] * len(dates) + ["C"] * len(dates),
            "product_name": ["A"] * len(dates) + ["C"] * len(dates),
            "category": ["Earrings"] * len(dates) + ["Rings"] * len(dates),
            "units_sold": [1] * (2 * len(dates)),
            "available_stock": [20] * len(dates) + [None] * len(dates),
            "stockout_flag": [0] * (2 * len(dates)),
            "data_source": ["synthetic"] * (2 * len(dates)),
        }
    )
    master = pd.DataFrame(
        {
            "sku": ["A", "C"],
            "category": ["Earrings", "Rings"],
            "launch_date": ["2024-01-01", "2024-01-01"],
            "supplier_lead_time_days": [7, 7],
        }
    )
    data = InventoryData(
        mode="synthetic",
        daily=daily,
        sku_master=master,
        purchase_orders=pd.DataFrame(),
        snapshots=pd.DataFrame(),
    )
    state = build_inventory_state(data, "2024-01-05").set_index("sku")
    assert state.loc["C", "stock_missing"]
    assert state.loc["C", "current_stock"] == 0.0
    assert "stock_missing" in state.loc["C", "state_flags"]
    assert not state.loc["A", "stock_missing"]

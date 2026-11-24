"""Integration tests for static inventory recommendations."""

from __future__ import annotations

import pandas as pd

from src.inventory.data_builder import InventoryData
from src.inventory.engine import RECOMMENDATION_COLUMNS, build_recommendations


def _tiny_inventory() -> tuple[InventoryData, pd.Timestamp, pd.DataFrame]:
    decision = pd.Timestamp("2024-04-01")
    history = pd.date_range(end=decision, periods=60, freq="D")
    rows = []
    for index, date in enumerate(history):
        rows.append(
            {
                "date": date,
                "sku": "A",
                "product_name": "A",
                "category": "Earrings",
                "units_sold": 1 + (index % 3),
                "available_stock": 100,
                "stockout_flag": 0,
                "data_source": "synthetic",
            }
        )
        rows.append(
            {
                "date": date,
                "sku": "B",
                "product_name": "B",
                "category": "Rings",
                "units_sold": 0,
                "available_stock": 1000,
                "stockout_flag": 0,
                "data_source": "synthetic",
            }
        )
    daily = pd.DataFrame(rows)
    master = pd.DataFrame(
        {
            "sku": ["A", "B"],
            "category": ["Earrings", "Rings"],
            "launch_date": [history[0], history[0]],
            "supplier_lead_time_days": [5, None],
        }
    )
    data = InventoryData(
        mode="synthetic",
        daily=daily,
        sku_master=master,
        purchase_orders=pd.DataFrame(),
        snapshots=pd.DataFrame(),
    )
    dates = pd.date_range(decision + pd.Timedelta(days=1), periods=10, freq="D")
    forecast = pd.DataFrame(
        {
            "sku": ["A"] * len(dates),
            "date": list(dates),
            "predicted_units": [4.0] * len(dates),
            "forecast_source": ["test"] * len(dates),
        }
    )
    return data, decision, forecast


def test_recommendation_schema_and_core_logic() -> None:
    data, decision, forecast = _tiny_inventory()
    result = build_recommendations(data, decision, forecast, service_level=0.95)
    for column in RECOMMENDATION_COLUMNS:
        assert column in result.columns
    assert set(result["sku"]) == {"A", "B"}

    row_a = result.set_index("sku").loc["A"]
    assert row_a["lead_time_days"] == 5
    assert row_a["lead_time_demand"] == 20
    assert row_a["safety_stock"] > 0
    assert row_a["reorder_point"] > 20
    assert row_a["inventory_position"] == 100
    assert row_a["reorder_status"] == "Healthy"
    assert row_a["recommended_order_qty"] == 0

    row_b = result.set_index("sku").loc["B"]
    assert row_b["lead_time_days"] == 14
    assert "lead_time_defaulted" in row_b["flags"]
    assert row_b["overstock_flag"]
    assert row_b["recommended_order_qty"] == 0
    assert "slow_moving_demand" in row_b["flags"]


def test_missing_forecast_is_flagged_and_does_not_crash() -> None:
    data, decision, forecast = _tiny_inventory()
    only_a = forecast.loc[forecast["sku"] == "A"]
    result = build_recommendations(data, decision, only_a, service_level=0.95)
    row_b = result.set_index("sku").loc["B"]
    assert "forecast_missing_fallback" in row_b["flags"]
    assert pd.notna(row_b["recommended_order_qty"])


def test_lead_time_and_demand_scenarios_change_recommendations() -> None:
    data, decision, forecast = _tiny_inventory()
    low = build_recommendations(
        data,
        decision,
        forecast,
        service_level=0.95,
        demand_multiplier=1.0,
        lead_time_extra_days=0,
    ).set_index("sku").loc["A"]
    high = build_recommendations(
        data,
        decision,
        forecast,
        service_level=0.99,
        demand_multiplier=2.0,
        lead_time_extra_days=10,
    ).set_index("sku").loc["A"]
    assert high["safety_stock"] > low["safety_stock"]
    assert high["lead_time_demand"] > low["lead_time_demand"]
    assert high["reorder_point"] > low["reorder_point"]


def test_committed_stock_is_not_used_by_default() -> None:
    data, decision, forecast = _tiny_inventory()
    snapshots = pd.DataFrame(
        {
            "snapshot_date": [decision],
            "sku": ["A"],
            "committed_qty": [50],
        }
    )
    data = InventoryData(
        mode=data.mode,
        daily=data.daily,
        sku_master=data.sku_master,
        purchase_orders=data.purchase_orders,
        snapshots=snapshots,
    )
    default = build_recommendations(data, decision, forecast).set_index("sku").loc["A"]
    assert default["committed_stock"] == 0
    assert not default["committed_stock_used"]
    opted_in = (
        build_recommendations(data, decision, forecast, use_committed_stock=True)
        .set_index("sku")
        .loc["A"]
    )
    assert opted_in["committed_stock_used"]
    assert opted_in["committed_stock"] == 50

def test_variable_supplier_lead_times_use_per_sku_horizon() -> None:
    data, decision, forecast = _tiny_inventory()
    master = data.sku_master.copy()
    master.loc[master["sku"] == "B", "supplier_lead_time_days"] = 20
    data = InventoryData(
        mode=data.mode,
        daily=data.daily,
        sku_master=master,
        purchase_orders=data.purchase_orders,
        snapshots=data.snapshots,
    )
    dates = pd.date_range(decision + pd.Timedelta(days=1), periods=25, freq="D")
    longer = pd.DataFrame(
        {
            "sku": ["A"] * len(dates) + ["B"] * len(dates),
            "date": list(dates) * 2,
            "predicted_units": [2.0] * len(dates) + [1.0] * len(dates),
            "forecast_source": ["test"] * (2 * len(dates)),
        }
    )
    result = build_recommendations(data, decision, longer).set_index("sku")
    assert result.loc["A", "lead_time_days"] == 5
    assert result.loc["B", "lead_time_days"] == 20
    assert result.loc["A", "lead_time_demand"] == 10.0
    assert result.loc["B", "lead_time_demand"] == 20.0
    assert result.loc["B", "protection_period_days"] == 27.0

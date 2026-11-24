"""Tests for scenarios and report schemas."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.inventory.data_builder import InventoryData
from src.inventory.engine import build_recommendations
from src.inventory.reporting import (
    write_evaluation_outputs,
    write_recommendation_outputs,
    write_scenario_outputs,
    write_simulation_outputs,
)
from src.inventory.scenarios import run_scenarios, scenario_deltas
from src.inventory.simulation import SimulationResult


def _inventory() -> tuple[InventoryData, pd.Timestamp, pd.DataFrame]:
    decision = pd.Timestamp("2024-04-01")
    dates = pd.date_range(end=decision, periods=45, freq="D")
    rows = []
    for index, date in enumerate(dates):
        for sku, category, sales, stock in (
            ("A", "Earrings", 1 + index % 4, 30),
            ("B", "Rings", 0, 500),
        ):
            rows.append(
                {
                    "date": date,
                    "sku": sku,
                    "category": category,
                    "units_sold": sales,
                    "available_stock": stock,
                    "stockout_flag": 0,
                    "data_source": "synthetic",
                }
            )
    daily = pd.DataFrame(rows)
    master = pd.DataFrame(
        {
            "sku": ["A", "B"],
            "category": ["Earrings", "Rings"],
            "launch_date": [dates[0], dates[0]],
            "supplier_lead_time_days": [7, 14],
        }
    )
    data = InventoryData(
        mode="synthetic",
        daily=daily,
        sku_master=master,
        purchase_orders=pd.DataFrame(),
        snapshots=pd.DataFrame(),
    )
    future = pd.date_range(decision + pd.Timedelta(days=1), periods=10, freq="D")
    forecast = pd.DataFrame(
        {
            "sku": ["A"] * len(future) + ["B"] * len(future),
            "date": list(future) * 2,
            "predicted_units": [4.0] * len(future) + [0.0] * len(future),
            "forecast_source": ["test"] * (2 * len(future)),
        }
    )
    return data, decision, forecast


def test_scenarios_are_labelled_and_change_assumptions_only() -> None:
    data, decision, forecast = _inventory()
    scenarios = [
        {"name": "baseline", "demand_multiplier": 1.0, "service_level": 0.95},
        {"name": "demand_up_50", "demand_multiplier": 1.5, "service_level": 0.95},
        {"name": "lead_time_plus_7", "lead_time_extra_days": 7, "service_level": 0.95},
        {"name": "service_99", "service_level": 0.99},
    ]
    summary, details = run_scenarios(data, decision, forecast, {}, scenarios=scenarios)
    assert list(summary["scenario"]) == [item["name"] for item in scenarios]
    assert set(details["scenario"]) == {item["name"] for item in scenarios}
    assert details["scenario_demand_multiplier"].notna().all()
    assert (summary["skus"] == 2).all()

    deltas = scenario_deltas(summary)
    assert "recommended_order_units_delta_vs_baseline" in deltas.columns
    baseline = summary.set_index("scenario").loc["baseline"]
    demand_up = summary.set_index("scenario").loc["demand_up_50"]
    service_99 = summary.set_index("scenario").loc["service_99"]
    assert demand_up["total_lead_time_demand"] >= baseline["total_lead_time_demand"]
    assert service_99["total_safety_stock"] > baseline["total_safety_stock"]


def test_duplicate_scenario_names_raise() -> None:
    data, decision, forecast = _inventory()
    with pytest.raises(ValueError):
        run_scenarios(
            data,
            decision,
            forecast,
            {},
            scenarios=[{"name": "x"}, {"name": "x"}],
        )


def test_recommendation_report_schema_and_scenario_section(tmp_path: Path) -> None:
    data, decision, forecast = _inventory()
    recommendations = build_recommendations(data, decision, forecast)
    summary, details = run_scenarios(
        data,
        decision,
        forecast,
        {},
        scenarios=[{"name": "baseline"}, {"name": "demand_up", "demand_multiplier": 2.0}],
    )
    paths = write_recommendation_outputs(
        "synthetic",
        decision,
        recommendations,
        tmp_path,
        metadata={"source": "test"},
        scenario_summary=scenario_deltas(summary),
    )
    for key in ("recommendations", "recommendations_csv", "summary", "report"):
        assert paths[key].exists()
    report = paths["report"].read_text(encoding="utf-8")
    assert "Scenario analysis" in report
    assert "demand_up" in report
    assert "Not real company" in report
    payload = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert payload["metrics"]["skus"] == 2
    assert payload["scenario_names"] == ["baseline", "demand_up"]
    assert payload["assumptions"]["source"] == "test"
    assert payload["metrics"]["holding_cost"] is None

    scenario_paths = write_scenario_outputs(summary, details, tmp_path / "scenarios")
    assert scenario_paths["summary_csv"].exists()
    assert scenario_paths["details"].exists()
    scenario_report = scenario_paths["report"].read_text(encoding="utf-8")
    assert "not causal" in scenario_report
    assert "No cost" in scenario_report


def test_simulation_and_evaluation_outputs_with_empty_orders(tmp_path: Path) -> None:
    dates = pd.date_range("2024-01-01", periods=2, freq="D")
    daily = pd.DataFrame(
        {
            "policy": ["forecast_driven"] * 2,
            "dataset": ["synthetic"] * 2,
            "date": dates,
            "sku": ["A"] * 2,
            "category": ["Earrings"] * 2,
            "opening_on_hand": [10.0, 8.0],
            "received_units": [0.0, 0.0],
            "demand_proxy_units": [2.0, 2.0],
            "fulfilled_units": [2.0, 2.0],
            "lost_units": [0.0, 0.0],
            "closing_on_hand": [8.0, 6.0],
            "incoming_open_units": [0.0, 0.0],
            "inventory_position_after_sales": [8.0, 6.0],
            "stockout_flag": [0, 0],
        }
    )
    result = SimulationResult(
        policy="forecast_driven",
        daily=daily,
        decisions=pd.DataFrame(),
        orders=pd.DataFrame(),
        metadata={
            "skus": 1,
            "start_date": "2024-01-01",
            "end_date": "2024-01-02",
            "decision_frequency_days": 7,
            "forecast_source": "test",
        },
    )
    comparison = pd.DataFrame(
        [
            {
                "policy": "forecast_driven",
                "skus": 1,
                "sku_days": 2,
                "demand_proxy_units": 4.0,
                "fulfilled_units": 4.0,
                "lost_sales_units": 0.0,
                "fill_rate": 1.0,
                "service_level_sku_day": 1.0,
                "stockout_frequency": 0.0,
                "average_inventory_units": 7.0,
                "inventory_turnover": 0.5,
                "days_of_inventory": 3.5,
                "excess_inventory_units": 0.0,
                "replenishment_count": 0,
                "replenishment_units": 0.0,
                "holding_cost": None,
                "dataset": "synthetic",
            }
        ]
    )
    per_sku = pd.DataFrame(
        [
            {
                "policy": "forecast_driven",
                "sku": "A",
                "category": "Earrings",
                "fill_rate": 1.0,
                "average_inventory_units": 7.0,
                "replenishment_count": 0,
            }
        ]
    )
    paths = write_simulation_outputs(
        {"forecast_driven": result},
        comparison,
        per_sku,
        tmp_path / "simulation",
    )
    assert paths["daily"].exists()
    assert paths["decisions"].exists()
    assert paths["orders"].exists()
    assert paths["comparison_csv"].exists()
    assert paths["report"].exists()

    eval_paths = write_evaluation_outputs(comparison, per_sku, tmp_path / "evaluation")
    assert eval_paths["comparison_csv"].exists()
    assert eval_paths["metrics_by_sku"].exists()
    assert eval_paths["summary"].exists()
    assert "Holding cost" in eval_paths["report"].read_text(encoding="utf-8")

"""Causal historical simulation tests.

These tests use a stub forecast provider that records its forecast origins so we
can assert the simulation never asks the model for future information.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.forecasting.data_builder import ForecastingPanel
from src.inventory.data_builder import InventoryData
from src.inventory.evaluation import policy_comparison
from src.inventory.simulation import SIMULATION_DAILY_COLUMNS, simulate_policy


class RecordingForecastProvider:
    name = "recording_stub"

    def __init__(self, panel: ForecastingPanel, daily_units: dict[str, float], horizon: int = 10):
        self.panel = panel
        self.daily_units = daily_units
        self.horizon = horizon
        self.origins: list[pd.Timestamp] = []

    def forecast(self, origin, horizon: int = 30) -> pd.DataFrame:
        origin = pd.Timestamp(origin).normalize()
        self.origins.append(origin)
        horizon = int(horizon)
        dates = pd.date_range(origin + pd.Timedelta(days=1), periods=horizon, freq="D")
        rows = []
        for sku, units in self.daily_units.items():
            for date in dates:
                rows.append(
                    {
                        "sku": sku,
                        "date": date,
                        "predicted_units": units,
                        "forecast_source": self.name,
                    }
                )
        return pd.DataFrame(rows)


def _fixture() -> tuple[ForecastingPanel, InventoryData, pd.Timestamp, pd.Timestamp]:
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    rows = []
    for date in dates:
        for sku, category, demand, stock in (
            ("A", "Earrings", 3.0, 10),
            ("B", "Rings", 0.5, 100),
        ):
            rows.append(
                {
                    "date": date,
                    "sku": sku,
                    "category": category,
                    "units_sold": demand,
                    "available_stock": stock,
                    "stockout_flag": 0,
                    "data_source": "synthetic",
                }
            )
    frame = pd.DataFrame(rows)
    panel = ForecastingPanel(
        mode="synthetic",
        frame=frame,
        calendar=pd.DataFrame({"date": dates}),
        category_codes={"Earrings": 0, "Rings": 1},
        first_dates=frame.groupby("sku")["date"].min(),
        event_codes={},
        metadata={"max_horizon": 10},
    )
    master = pd.DataFrame(
        {
            "sku": ["A", "B"],
            "category": ["Earrings", "Rings"],
            "launch_date": [dates[0], dates[0]],
            "supplier_lead_time_days": [2, 2],
        }
    )
    data = InventoryData(
        mode="synthetic",
        daily=frame,
        sku_master=master,
        purchase_orders=pd.DataFrame(),
        snapshots=pd.DataFrame(),
    )
    return panel, data, dates[0], dates[-1]


def test_forecast_origin_is_always_decision_minus_one() -> None:
    panel, data, start, end = _fixture()
    provider = RecordingForecastProvider(panel, {"A": 3.0, "B": 0.5})
    result = simulate_policy(
        panel,
        data,
        {},
        "forecast_driven",
        forecast_provider=provider,
        start_date=start,
        end_date=end,
        decision_frequency_days=3,
        horizon=10,
    )
    decision_dates = sorted(result.decisions["decision_date"].unique())
    assert decision_dates == [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-04"),
        pd.Timestamp("2024-01-07"),
        pd.Timestamp("2024-01-10"),
    ]
    assert provider.origins == [date - pd.Timedelta(days=1) for date in decision_dates]
    assert all(origin < date for origin, date in zip(provider.origins, decision_dates))
    assert set(result.decisions["forecast_origin_date"]) == set(provider.origins)


def test_orders_arrive_after_lead_time_and_stock_is_conserved() -> None:
    panel, data, start, end = _fixture()
    provider = RecordingForecastProvider(panel, {"A": 3.0, "B": 0.5})
    result = simulate_policy(
        panel,
        data,
        {},
        "forecast_driven",
        forecast_provider=provider,
        start_date=start,
        end_date=end,
        decision_frequency_days=3,
        horizon=10,
    )
    for column in SIMULATION_DAILY_COLUMNS:
        assert column in result.daily.columns

    first_order = result.orders.sort_values("order_date").iloc[0]
    assert first_order["sku"] == "A"
    assert first_order["order_date"] == pd.Timestamp("2024-01-01")
    assert first_order["expected_arrival_date"] == pd.Timestamp("2024-01-03")
    assert first_order["lead_time_days"] == 2

    arrival = result.daily.loc[
        (result.daily["sku"] == "A") & (result.daily["date"] == pd.Timestamp("2024-01-03"))
    ].iloc[0]
    assert arrival["received_units"] == first_order["quantity"]

    daily = result.daily
    assert (daily["fulfilled_units"] <= daily["demand_proxy_units"] + 1e-9).all()
    assert (daily["fulfilled_units"] <= daily["opening_on_hand"] + daily["received_units"] + 1e-9).all()
    assert ((daily["opening_on_hand"] + daily["received_units"] - daily["fulfilled_units"]) - daily["closing_on_hand"]).abs().max() < 1e-9
    assert (daily["lost_units"] >= 0).all()


def test_losses_when_opening_stock_is_zero() -> None:
    panel, data, _, _ = _fixture()
    empty_stock = panel.frame.assign(available_stock=0)
    empty_panel = ForecastingPanel(
        mode=panel.mode,
        frame=empty_stock,
        calendar=panel.calendar,
        category_codes=panel.category_codes,
        first_dates=panel.first_dates,
        event_codes=panel.event_codes,
        metadata=panel.metadata,
    )
    empty_data = InventoryData(
        mode=data.mode,
        daily=empty_stock,
        sku_master=data.sku_master,
        purchase_orders=pd.DataFrame(),
        snapshots=pd.DataFrame(),
    )
    provider = RecordingForecastProvider(empty_panel, {"A": 3.0, "B": 0.5})
    result = simulate_policy(
        empty_panel,
        empty_data,
        {},
        "forecast_driven",
        forecast_provider=provider,
        start_date="2024-01-01",
        end_date="2024-01-02",
        decision_frequency_days=5,
        horizon=5,
    )
    first_a = result.daily.loc[result.daily["sku"] == "A"].iloc[0]
    assert first_a["lost_units"] == 3.0
    assert first_a["stockout_flag"] == 1
    assert result.metadata["demand_definition"].startswith("observed units_sold")
    assert "decision_date - 1 day" in result.metadata["look_ahead_guard"]


def test_policy_comparison_uses_common_demand_and_schema() -> None:
    panel, data, start, end = _fixture()
    provider = RecordingForecastProvider(panel, {"A": 3.0, "B": 0.5})
    forecast_driven = simulate_policy(
        panel,
        data,
        {},
        "forecast_driven",
        forecast_provider=provider,
        start_date=start,
        end_date=end,
        decision_frequency_days=3,
        horizon=10,
    )
    simple = simulate_policy(
        panel,
        data,
        {},
        "simple_rop",
        start_date=start,
        end_date=end,
        decision_frequency_days=3,
        horizon=10,
    )
    comparison, per_sku, metrics = policy_comparison(
        {"forecast_driven": forecast_driven, "simple_rop": simple}
    )
    assert set(comparison["policy"]) == {"forecast_driven", "simple_rop"}
    assert comparison["demand_proxy_units"].nunique() == 1
    assert set(per_sku["policy"]) == {"forecast_driven", "simple_rop"}
    assert set(metrics) == {"forecast_driven", "simple_rop"}
    for column in ("fill_rate", "service_level_sku_day", "stockout_frequency", "days_of_inventory"):
        assert column in comparison.columns
    assert (comparison["fill_rate"] >= 0).all()
    assert (comparison["fill_rate"] <= 1).all()
    assert "holding_cost" not in comparison.columns or comparison["holding_cost"].isna().all()


def test_unsupported_policy_and_bad_window_raise() -> None:
    panel, data, start, end = _fixture()
    with pytest.raises(ValueError):
        simulate_policy(panel, data, {}, "not_a_policy")
    with pytest.raises(ValueError):
        simulate_policy(
            panel,
            data,
            {},
            "simple_rop",
            start_date=end,
            end_date=start,
        )

def test_evaluation_metrics_have_known_values() -> None:
    from src.inventory.evaluation import simulation_metrics

    dates = pd.to_datetime(["2024-01-01", "2024-01-02"])
    rows = []
    for sku, opening, demand, fulfilled, lost, closing, flag in [
        ("A", [10.0, 8.0], [2.0, 3.0], [2.0, 2.0], [0.0, 1.0], [8.0, 5.0], [0, 1]),
        ("B", [5.0, 5.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [5.0, 5.0], [0, 0]),
    ]:
        for index, date in enumerate(dates):
            rows.append(
                {
                    "policy": "p",
                    "dataset": "synthetic",
                    "date": date,
                    "sku": sku,
                    "category": "Earrings",
                    "opening_on_hand": opening[index],
                    "received_units": 0.0,
                    "demand_proxy_units": demand[index],
                    "fulfilled_units": fulfilled[index],
                    "lost_units": lost[index],
                    "closing_on_hand": closing[index],
                    "incoming_open_units": 0.0,
                    "inventory_position_after_sales": closing[index],
                    "stockout_flag": flag[index],
                }
            )
    daily = pd.DataFrame(rows)
    orders = pd.DataFrame(
        {
            "policy": ["p"],
            "po_id": ["PO1"],
            "sku": ["A"],
            "order_date": [dates[0]],
            "expected_arrival_date": [dates[1]],
            "quantity": [6],
            "lead_time_days": [1],
        }
    )
    overall, per_sku = simulation_metrics(daily, pd.DataFrame(), orders)
    assert overall["demand_proxy_units"] == 5.0
    assert overall["fulfilled_units"] == 4.0
    assert overall["lost_sales_units"] == 1.0
    assert overall["fill_rate"] == 0.8
    assert overall["stockout_sku_days"] == 1
    assert overall["service_level_sku_day"] == 0.75
    assert overall["stockout_frequency"] == 0.25
    assert overall["average_inventory_units"] == 7.0
    assert abs(overall["inventory_turnover"] - 5.0 / 7.0) < 1e-12
    assert abs(overall["days_of_inventory"] - 5.6) < 1e-12
    assert overall["replenishment_count"] == 1
    assert overall["replenishment_units"] == 6
    assert overall["holding_cost"] is None
    assert set(per_sku["sku"]) == {"A", "B"}


def test_artifact_forecast_provider_filters_to_origin_and_horizon(tmp_path) -> None:
    from src.inventory.forecasting_adapter import ArtifactForecastProvider

    origin = pd.Timestamp("2024-04-01")
    dates = pd.date_range(origin, periods=6, freq="D")
    frame = pd.DataFrame(
        {
            "sku": ["A"] * 12,
            "forecast_origin_date": [origin] * 6 + [origin + pd.Timedelta(days=1)] * 6,
            "forecast_date": list(dates) * 2,
            "predicted_units": [1.0] * 12,
            "model": ["m"] * 12,
        }
    )
    path = tmp_path / "forecast.parquet"
    frame.to_parquet(path, index=False)
    provider = ArtifactForecastProvider(path)
    result = provider.forecast(origin, horizon=3)
    assert set(pd.to_datetime(result["date"])) == set(dates[:4])
    assert (pd.to_datetime(result["date"]) <= origin + pd.Timedelta(days=3)).all()

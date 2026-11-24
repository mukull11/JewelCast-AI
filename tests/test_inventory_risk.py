"""Tests for days cover, stockout risk, overstock and explanations."""

from __future__ import annotations

import math

import pandas as pd

from src.inventory.risk_detection import (
    classify_overstock,
    classify_stockout_risk,
    days_of_cover,
    explain_recommendation,
    projected_stockout_date,
)


def test_days_of_cover_handles_zero_demand() -> None:
    assert days_of_cover(10, 5) == 2.0
    assert math.isnan(days_of_cover(10, 0))
    assert days_of_cover(-5, 2) == 0.0


def test_projected_stockout_includes_incoming_orders() -> None:
    forecast = [5, 5, 5, 5]
    no_incoming = projected_stockout_date("2024-01-01", 10, forecast)
    assert no_incoming == pd.Timestamp("2024-01-04")
    with_incoming = projected_stockout_date(
        "2024-01-01", 10, forecast, [("2024-01-02", 5)]
    )
    assert with_incoming == pd.Timestamp("2024-01-05")
    assert projected_stockout_date("2024-01-01", 100, forecast) is None


def test_stockout_risk_levels() -> None:
    constant = [5.0] * 30
    critical = classify_stockout_risk("2024-01-01", 0, 5, 5, 7, constant)
    assert critical.risk == "Critical"
    assert critical.existing_stockout is True
    assert critical.days_until_stockout is None

    high = classify_stockout_risk("2024-01-01", 15, 3, 7, 7, [3.0] * 30)
    assert high.risk == "High"
    assert high.days_until_stockout == 6.0

    medium = classify_stockout_risk("2024-01-01", 30, 3, 7, 7, [3.0] * 30)
    assert medium.risk == "Medium"

    low = classify_stockout_risk("2024-01-01", 100, 3, 7, 7, [3.0] * 30)
    assert low.risk == "Low"
    assert low.projected_stockout_date is None


def test_stockout_risk_flags_missing_forecast() -> None:
    result = classify_stockout_risk("2024-01-01", 5, 4, 7, 7, [])
    assert "forecast_missing_projected_stockout" in result.flags
    assert result.risk in {"Critical", "High", "Medium", "Low"}


def test_overstock_is_measured_in_units_only() -> None:
    over = classify_overstock(1000, 5, cover_threshold_days=90)
    assert over.overstock is True
    assert over.expected_demand_cover_period == 450
    assert over.excess_units == 550
    assert not hasattr(over, "cost")

    normal = classify_overstock(100, 5, cover_threshold_days=90)
    assert normal.overstock is False
    assert normal.excess_units == 0

    slow = classify_overstock(10, 0.05)
    assert slow.overstock is True
    assert slow.excess_units == 5.5
    assert "slow_moving_demand" in slow.flags


def test_explanation_mentions_quantities_and_flags() -> None:
    row = {
        "reorder_status": "At Risk",
        "stockout_risk": "High",
        "inventory_position": 108,
        "reorder_point": 98,
        "lead_time_demand": 71.2,
        "safety_stock": 27.2,
        "current_stock": 27,
        "incoming_stock": 81,
        "recommended_order_qty": 24,
        "target_stock": 132,
        "days_of_cover": 5.7,
        "lead_time_days": 15,
        "flags": "lead_time_defaulted",
    }
    text = explain_recommendation(row)
    assert "At Risk" in text
    assert "108" in text and "98" in text
    assert "24 units" in text
    assert "lead_time_defaulted" in text
    assert "[]" not in text

"""Unit tests for lead-time demand, safety stock and replenishment arithmetic."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.inventory.lead_time import lead_time_demand, protection_period_demand, sum_demand_over_horizon
from src.inventory.reorder_point import classify_reorder_status, reorder_point
from src.inventory.replenishment import (
    protection_period_days,
    recommended_order_quantity,
    target_stock,
)
from src.inventory.safety_stock import (
    estimate_lead_time_demand_sigma,
    safety_stock,
    service_level_z,
)


def test_lead_time_demand_sums_daily_sequence() -> None:
    # Six-day lead time should sum six daily values, not 5 * 6 average.
    result = lead_time_demand([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 100.0], 6, 5.0)
    assert result.units == 21.0
    assert result.days_forecast == 6
    assert result.days_extrapolated == 0
    assert result.flags == []


def test_lead_time_demand_flags_short_horizon_without_actuals() -> None:
    result = lead_time_demand([5.0, 5.0, 5.0], 6, 5.0)
    assert result.units == 30.0
    assert result.days_extrapolated == 3
    assert "lead_time_exceeds_forecast_horizon" in result.flags
    assert result.fallback_used is False


def test_lead_time_demand_missing_forecast_uses_flagged_fallback() -> None:
    result = lead_time_demand([], 4, 2.5)
    assert result.units == 10.0
    assert result.fallback_used is True
    assert "forecast_missing_fallback" in result.flags


def test_protection_period_and_invalid_days() -> None:
    assert protection_period_days(10, 7) == 17
    assert protection_period_days(10, 7, include_review=False) == 10
    result = protection_period_demand([2, 2], 0, 1)
    assert result.units == 0
    assert "invalid_protection_period_days" in result.flags


def test_service_level_z_values() -> None:
    assert math.isclose(service_level_z(0.90), 1.2815515655, rel_tol=1e-6)
    assert math.isclose(service_level_z(0.95), 1.6448536270, rel_tol=1e-6)
    assert math.isclose(service_level_z(0.99), 2.3263478740, rel_tol=1e-6)
    with pytest.raises(ValueError):
        service_level_z(1.0)


def test_safety_stock_uses_z_times_sigma() -> None:
    value, z = safety_stock(0.95, 10.0)
    assert math.isclose(value, z * 10.0)
    assert value >= 0


def test_demand_volatility_safety_stock_and_zero_demand() -> None:
    assert estimate_lead_time_demand_sigma(np.array([0.0] * 20), 5, min_history=10)[0] == 0.0
    sigma, sigma_daily, fallback, flags = estimate_lead_time_demand_sigma(
        np.array([2.0, 4.0] * 10), 4, min_history=10
    )
    expected_daily = float(np.std([2.0, 4.0] * 10, ddof=1))
    assert math.isclose(sigma_daily, expected_daily, rel_tol=1e-9)
    assert math.isclose(sigma, expected_daily * 2.0, rel_tol=1e-9)
    assert fallback is False
    assert flags == []


def test_short_history_uses_category_fallback() -> None:
    sigma, sigma_daily, fallback, flags = estimate_lead_time_demand_sigma(
        np.array([1.0, 2.0]), 9, min_history=14, fallback_sigma_daily=3.0
    )
    assert fallback is True
    assert sigma_daily == 3.0
    assert math.isclose(sigma, 9.0)
    assert "short_history_category_volatility_fallback" in flags


def test_forecast_error_and_empirical_methods() -> None:
    sigma, _, fallback, _ = estimate_lead_time_demand_sigma(
        np.array([1.0, 1.0, 1.0]), 4, method="forecast_error", forecast_error_std_daily=1.5
    )
    assert math.isclose(sigma, 3.0)
    assert fallback is False
    sigma, _, _, flags = estimate_lead_time_demand_sigma(
        np.array([0.0, 10.0] * 20), 2, method="empirical", service_level=0.90, min_history=10
    )
    assert sigma >= 0
    assert "empirical_percentile_method" in flags


def test_reorder_point_and_target_stock() -> None:
    assert reorder_point(24, 8) == 32
    assert target_stock(48, 8) == 56
    assert reorder_point(-5, -2) == 0


def test_order_quantity_constraints() -> None:
    assert recommended_order_quantity(80, 50).quantity == 30
    assert recommended_order_quantity(80, 100).quantity == 0
    raised = recommended_order_quantity(80, 70, min_order_qty=20)
    assert raised.quantity == 20
    assert "raised_to_min_order_qty" in raised.flags
    multiple = recommended_order_quantity(100, 40, order_multiple=6)
    assert multiple.quantity == 60
    assert multiple.quantity % 6 == 0
    capped = recommended_order_quantity(100, 90, max_storage_units=95)
    assert capped.quantity == 5
    assert "capped_by_max_storage" in capped.flags
    assert recommended_order_quantity(-10, 5).quantity == 0


def test_reorder_status_mapping() -> None:
    assert classify_reorder_status(10, 30, "Low") == "Reorder Required"
    assert classify_reorder_status(10, 30, "High") == "At Risk"
    assert classify_reorder_status(10, 30, "Critical") == "Critical"
    assert classify_reorder_status(40, 30, "Low") == "Healthy"
    assert classify_reorder_status(40, 30, "Medium") == "At Risk"

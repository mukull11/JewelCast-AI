"""Configuration helpers for the M4 inventory optimization engine.

Defaults live here as well as in ``config.yaml`` so that every module can be
used directly with a partial configuration mapping (as in small unit tests)
without hardcoding business assumptions in the calculation code.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping

import pandas as pd

DEFAULT_SERVICE_LEVEL = 0.95
DEFAULT_LEAD_TIME_DAYS = 14
DEFAULT_REVIEW_PERIOD_DAYS = 7
DEFAULT_FORECAST_HORIZON = 30

DEFAULT_INVENTORY_SETTINGS: dict[str, Any] = {
    "mode": "synthetic",
    "service_level": DEFAULT_SERVICE_LEVEL,
    "supported_service_levels": [0.90, 0.95, 0.97, 0.99],
    "default_lead_time_days": DEFAULT_LEAD_TIME_DAYS,
    "review_period_days": DEFAULT_REVIEW_PERIOD_DAYS,
    "forecast_horizon": DEFAULT_FORECAST_HORIZON,
    "safety_stock_method": "demand_volatility",
    "demand_volatility_window_days": 28,
    "min_history_days_for_volatility": 14,
    "use_committed_stock": False,
    "risk": {
        "critical_cover_days": 3,
        "medium_cover_includes_review": True,
    },
    "overstock": {
        "cover_threshold_days": 90,
        "slow_moving_daily_units": 0.10,
        "inactive_history_days": 30,
    },
    "replenishment": {
        "min_order_qty": 0,
        "order_multiple": 1,
        "max_storage_units": None,
        "protection_period_includes_review": True,
    },
    "scenarios": [
        {"name": "baseline", "demand_multiplier": 1.0},
        {"name": "demand_plus_10", "demand_multiplier": 1.10},
        {"name": "demand_plus_20", "demand_multiplier": 1.20},
        {"name": "demand_minus_10", "demand_multiplier": 0.90},
        {"name": "lead_time_plus_7", "lead_time_extra_days": 7},
        {"name": "service_level_97", "service_level": 0.97},
        {"name": "service_level_99", "service_level": 0.99},
    ],
    "simulation": {
        "start_date": "2024-10-01",
        "end_date": "2024-12-31",
        "decision_frequency_days": 7,
        "forecast_mode": "frozen",
        "refit_every_days": 30,
        "horizon": 30,
        "include_opening_purchase_orders": False,
        "max_training_rows": 400000,
        "policies": ["forecast_driven", "simple_rop"],
    },
    "reporting": {
        "output_dir": "reports/inventory",
        "top_n": 25,
    },
}


def _deep_merge(base: dict, override: Mapping[str, Any]) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def inventory_settings(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return the merged inventory configuration with documented defaults."""
    section = dict((config or {}).get("inventory", {}) or {})
    return _deep_merge(DEFAULT_INVENTORY_SETTINGS, section)


def negative_to_zero(value: Any) -> float:
    """Clip a numeric value at zero while tolerating nulls."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if pd.isna(number):
        return 0.0
    return max(0.0, number)


def as_timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError(f"Invalid date value: {value!r}")
    return timestamp.normalize()


def maybe_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None
    return int(value)

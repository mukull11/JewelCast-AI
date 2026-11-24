"""Stockout and overstock risk detection.

Risk labels are transparent, configurable rules based on projected stock cover.
They are model-based warnings, not certainty. Days of cover use on-hand stock;
projected stockouts also account for scheduled incoming purchase orders.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd


@dataclass
class StockoutRiskResult:
    risk: str
    days_of_cover: float
    projected_stockout_date: pd.Timestamp | None
    days_until_stockout: float | None
    existing_stockout: bool
    expected_demand_before_replenishment: float
    flags: list[str] = field(default_factory=list)

    @property
    def flag_string(self) -> str:
        return ";".join(self.flags)


@dataclass
class OverstockResult:
    overstock: bool
    excess_units: float
    expected_demand_cover_period: float
    excess_cover_days: float
    flags: list[str] = field(default_factory=list)

    @property
    def flag_string(self) -> str:
        return ";".join(self.flags)


def days_of_cover(on_hand: float, expected_daily_demand: float) -> float:
    """Days of on-hand cover. NaN when expected demand is non-positive."""
    daily = float(expected_daily_demand or 0.0)
    stock = max(0.0, float(on_hand or 0.0))
    if daily <= 0:
        return float("nan")
    return stock / daily


def projected_stockout_date(
    decision_date: Any,
    on_hand: float,
    forecast_values: Iterable[float],
    incoming_schedule: Iterable[tuple[Any, float]] = (),
) -> pd.Timestamp | None:
    """First date at which forecast demand exceeds available stock + arrivals."""
    decision = pd.Timestamp(decision_date).normalize()
    balance = max(0.0, float(on_hand or 0.0))
    arrivals: dict[pd.Timestamp, float] = {}
    for arrival_date, quantity in incoming_schedule:
        if quantity is None or not np.isfinite(float(quantity)):
            continue
        key = pd.Timestamp(arrival_date).normalize()
        if key <= decision:
            continue
        arrivals[key] = arrivals.get(key, 0.0) + max(0.0, float(quantity))
    values = np.asarray(list(forecast_values), dtype=float)
    values = np.clip(values[np.isfinite(values)], 0.0, None)
    for offset, demand in enumerate(values, start=1):
        date = decision + pd.Timedelta(days=offset)
        balance += arrivals.get(date, 0.0)
        if demand > balance + 1e-9:
            return date
        balance -= demand
    return None


def classify_stockout_risk(
    decision_date: Any,
    on_hand: float,
    expected_daily_demand: float,
    lead_time_days: float,
    review_period_days: float,
    forecast_values: Iterable[float],
    incoming_schedule: Iterable[tuple[Any, float]] = (),
    critical_cover_days: float = 3,
    medium_cover_includes_review: bool = True,
) -> StockoutRiskResult:
    """Classify stockout risk as Low, Medium, High or Critical."""
    decision = pd.Timestamp(decision_date).normalize()
    daily = max(0.0, float(expected_daily_demand or 0.0))
    stock = max(0.0, float(on_hand or 0.0))
    existing = stock <= 0 and daily > 0
    cover = days_of_cover(stock, daily)
    forecast_array = np.asarray(list(forecast_values), dtype=float)
    forecast_array = forecast_array[np.isfinite(forecast_array)]
    projected = projected_stockout_date(decision, stock, forecast_array, incoming_schedule)
    flags: list[str] = []
    if existing:
        flags.append("existing_stockout")
    if forecast_array.size == 0 and daily > 0:
        flags.append("forecast_missing_projected_stockout")
    if np.isnan(cover):
        flags.append("zero_expected_demand")

    lead = max(1.0, float(lead_time_days or 0.0))
    review = max(0.0, float(review_period_days or 0.0))
    medium_boundary = lead + (review if medium_cover_includes_review else 0.0)
    demand_before_replenishment = daily * lead
    days_until = None

    if existing:
        risk = "Critical"
    elif daily <= 0:
        risk = "Low"
    else:
        days_until = float((projected - decision).days) if projected is not None else None
        if projected is not None and days_until is not None:
            if days_until <= float(critical_cover_days):
                risk = "Critical"
            elif days_until < lead:
                risk = "High"
            elif days_until < medium_boundary:
                risk = "Medium"
            else:
                risk = "Low"
        elif not np.isnan(cover) and cover < lead:
            risk = "High"
        elif not np.isnan(cover) and cover < medium_boundary:
            risk = "Medium"
        else:
            risk = "Low"
    return StockoutRiskResult(
        risk=risk,
        days_of_cover=cover,
        projected_stockout_date=projected,
        days_until_stockout=days_until,
        existing_stockout=existing,
        expected_demand_before_replenishment=demand_before_replenishment,
        flags=flags,
    )


def classify_overstock(
    inventory_position_value: float,
    expected_daily_demand: float,
    cover_threshold_days: float = 90,
    slow_moving_daily_units: float = 0.1,
    recent_units: float | None = None,
) -> OverstockResult:
    """Detect stock substantially above expected demand.

    Financial value is never invented: only units and cover days are returned,
    because the dataset does not contain purchase cost.
    """
    position = max(0.0, float(inventory_position_value or 0.0))
    daily = max(0.0, float(expected_daily_demand or 0.0))
    threshold = max(1.0, float(cover_threshold_days or 90))
    expected_cover = daily * threshold
    flags: list[str] = []
    if daily <= max(0.0, float(slow_moving_daily_units or 0.0)) and position > 0:
        flags.append("slow_moving_demand")
    if recent_units is not None and float(recent_units) <= 0 and position > 0:
        flags.append("inactive_recent_sales")

    if daily <= 0:
        excess = position
        excess_cover = float("inf") if position > 0 else 0.0
        overstock = position > 0
    else:
        excess = max(0.0, position - expected_cover)
        excess_cover = max(0.0, position / daily - threshold)
        overstock = position > expected_cover
    return OverstockResult(
        overstock=bool(overstock),
        excess_units=float(excess),
        expected_demand_cover_period=float(expected_cover),
        excess_cover_days=float(excess_cover),
        flags=flags,
    )


def explain_recommendation(row: Any) -> str:
    """Build a human-readable explanation for one recommendation row."""

    def value(name: str, default: float = 0.0) -> float:
        try:
            number = float(row[name])
        except (KeyError, TypeError, ValueError):
            return default
        return default if not np.isfinite(number) else number

    status = str(row.get("reorder_status", ""))
    risk = str(row.get("stockout_risk", ""))
    position = value("inventory_position")
    rop = value("reorder_point")
    pieces: list[str] = []
    if position < rop:
        pieces.append(
            f"{status}: inventory position {position:.0f} is below the reorder point {rop:.0f}."
        )
    elif status == "Healthy":
        pieces.append(
            f"Healthy: inventory position {position:.0f} is at or above the reorder point {rop:.0f}."
        )
    else:
        pieces.append(
            f"{status}: inventory position {position:.0f} is at or above the reorder point "
            f"{rop:.0f}, but scheduled stock is projected to run out before or shortly after replenishment."
        )
    pieces.append(
        f"Expected lead-time demand is {value('lead_time_demand'):.1f} units, safety stock is "
        f"{value('safety_stock'):.1f} units, and current stock is {value('current_stock'):.0f} "
        f"with {value('incoming_stock'):.0f} incoming."
    )
    quantity = value("recommended_order_qty")
    if quantity > 0:
        pieces.append(
            f"Recommended order quantity is {quantity:.0f} units to reach target stock "
            f"{value('target_stock'):.0f}."
        )
    else:
        pieces.append("No replenishment order is recommended today.")
    pieces.append(
        f"Stockout risk is {risk} with {value('days_of_cover'):.1f} days of on-hand cover "
        f"against a {value('lead_time_days'):.0f}-day supplier lead time."
    )
    flags = str(row.get("flags", "") or "")
    if flags:
        pieces.append(f"Assumptions/fallbacks: {flags}.")
    return " ".join(pieces)

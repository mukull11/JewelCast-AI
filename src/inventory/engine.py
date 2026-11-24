"""Recommendation engine shared by static reports and the historical simulator.

For every SKU and decision date the engine:

1. reads the inventory state available at the start of that date;
2. uses a forecast whose origin is the previous day (no same-day or future sales);
3. computes lead-time demand, safety stock, reorder point and target stock;
4. recommends a non-negative integer order quantity;
5. classifies stockout and overstock risk and writes a business explanation.

The same calculation is used by scenario analysis, so scenario differences are
attributable to the changed assumption rather than to a different code path.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from src.inventory.data_builder import (
    InventoryData,
    build_inventory_state,
    daily_demand_history,
    pending_purchase_orders,
)
from src.inventory.inventory_position import inventory_position
from src.inventory.lead_time import lead_time_demand, protection_period_demand
from src.inventory.reorder_point import classify_reorder_status, reorder_point
from src.inventory.replenishment import (
    protection_period_days,
    recommended_order_quantity,
    target_stock,
)
from src.inventory.risk_detection import (
    classify_overstock,
    classify_stockout_risk,
    days_of_cover,
    explain_recommendation,
)
from src.inventory.safety_stock import (
    category_volatility_fallback,
    estimate_lead_time_demand_sigma,
    safety_stock,
)
from src.inventory.settings import as_timestamp, inventory_settings

LOGGER = logging.getLogger(__name__)

RECOMMENDATION_COLUMNS = (
    "dataset",
    "sku",
    "category",
    "decision_date",
    "forecast_origin_date",
    "forecast_cutoff_date",
    "forecast_source",
    "current_stock",
    "incoming_stock",
    "incoming_due_within_lead_time",
    "committed_stock",
    "committed_stock_used",
    "inventory_position",
    "expected_daily_demand",
    "lead_time_days",
    "review_period_days",
    "protection_period_days",
    "lead_time_demand",
    "protection_period_demand",
    "safety_stock",
    "z_value",
    "sigma_lead_time_demand",
    "sigma_daily",
    "reorder_point",
    "target_stock",
    "recommended_order_qty",
    "raw_recommended_qty",
    "days_of_cover",
    "projected_stockout_date",
    "days_until_stockout",
    "existing_stockout",
    "stockout_risk",
    "reorder_status",
    "overstock_flag",
    "excess_units",
    "excess_cover_days",
    "recent_units",
    "service_level",
    "safety_stock_method",
    "flags",
    "explanation",
)


def _unique_flags(*values: Any) -> str:
    seen: list[str] = []
    for value in values:
        if value is None:
            continue
        text = str(value)
        for item in text.split(";"):
            item = item.strip()
            if item and item not in seen:
                seen.append(item)
    return ";".join(seen)


def _incoming_schedule(
    purchase_orders: pd.DataFrame,
    sku: str,
    as_of: pd.Timestamp,
) -> list[tuple[pd.Timestamp, float]]:
    if purchase_orders is None or purchase_orders.empty:
        return []
    pending = pending_purchase_orders(purchase_orders, as_of)
    if pending.empty:
        return []
    view = pending.loc[pending["sku"].astype(str) == str(sku)]
    schedule: list[tuple[pd.Timestamp, float]] = []
    for _, row in view.iterrows():
        arrival = pd.Timestamp(row["expected_arrival_date"]).normalize()
        if arrival <= as_of:
            arrival = as_of + pd.Timedelta(days=1)
        schedule.append((arrival, float(row["quantity"])))
    return schedule


def recommendation_metrics(
    *,
    sku: str,
    category: str,
    decision_date: Any,
    current_stock: float,
    incoming_stock: float,
    incoming_due_within_lead_time: float,
    committed_stock: float,
    committed_stock_used: bool,
    lead_time_base_days: float,
    history: np.ndarray,
    forecast_values: np.ndarray,
    fallback_daily: float,
    recent_units: float = 0.0,
    incoming_schedule: Iterable[tuple[Any, float]] = (),
    config: Mapping[str, Any] | None = None,
    service_level: float | None = None,
    demand_multiplier: float = 1.0,
    lead_time_multiplier: float = 1.0,
    lead_time_extra_days: float = 0.0,
    forecast_source: str = "m3_forecast",
    state_flags: str = "",
    fallback_sigma_daily: float = 0.0,
) -> dict[str, Any]:
    """Compute one SKU recommendation from already-aligned inputs."""
    settings = inventory_settings(config)
    level = float(service_level if service_level is not None else settings["service_level"])
    demand_multiplier = max(0.0, float(demand_multiplier))
    review = float(settings["review_period_days"])
    critical_days = float(settings["risk"]["critical_cover_days"])
    medium_includes_review = bool(settings["risk"]["medium_cover_includes_review"])
    replenishment = settings["replenishment"]
    overstock_settings = settings["overstock"]
    method = str(settings["safety_stock_method"])
    vol_window = int(settings["demand_volatility_window_days"])
    min_history = int(settings["min_history_days_for_volatility"])

    decision = as_timestamp(decision_date)
    origin = decision - pd.Timedelta(days=1)
    lead = max(
        1,
        int(
            round(
                max(0.0, float(lead_time_base_days or 0.0)) * max(0.0, float(lead_time_multiplier))
                + max(0.0, float(lead_time_extra_days))
            )
        ),
    )

    history = np.asarray(history, dtype=float)
    history = history[np.isfinite(history)] * demand_multiplier
    forecast_values = np.asarray(forecast_values, dtype=float)
    forecast_values = np.clip(forecast_values[np.isfinite(forecast_values)], 0.0, None) * demand_multiplier
    fallback = max(0.0, float(fallback_daily or 0.0)) * demand_multiplier
    flags: list[str] = []
    if len(forecast_values) == 0:
        flags.append("forecast_missing_fallback")
        forecast_source = "trailing_mean_fallback"

    ltd = lead_time_demand(forecast_values, lead, fallback)
    include_review = bool(replenishment.get("protection_period_includes_review", True))
    protection = protection_period_days(lead, review, include_review=include_review)
    protection_demand = protection_period_demand(forecast_values, protection, fallback)

    recent_history = history[-vol_window:] if len(history) else history
    sigma, sigma_daily, _fallback_used, sigma_flags = estimate_lead_time_demand_sigma(
        recent_history,
        lead,
        method=method,
        service_level=level,
        min_history=min_history,
        fallback_sigma_daily=max(0.0, float(fallback_sigma_daily or 0.0)) * demand_multiplier,
    )
    ss, z = safety_stock(level, sigma)
    rop = reorder_point(ltd.units, ss)
    target = target_stock(protection_demand.units, ss)
    position = inventory_position(current_stock, incoming_stock, committed_stock)

    order = recommended_order_quantity(
        target,
        position,
        min_order_qty=float(replenishment.get("min_order_qty", 0) or 0),
        order_multiple=float(replenishment.get("order_multiple", 1) or 1),
        max_storage_units=replenishment.get("max_storage_units"),
    )

    expected_daily = float(forecast_values.mean()) if len(forecast_values) else fallback
    cover = days_of_cover(current_stock, expected_daily)
    risk = classify_stockout_risk(
        decision,
        current_stock,
        expected_daily,
        lead,
        review,
        forecast_values,
        incoming_schedule,
        critical_cover_days=critical_days,
        medium_cover_includes_review=medium_includes_review,
    )
    status = classify_reorder_status(position, rop, risk.risk)
    overstock = classify_overstock(
        position,
        expected_daily,
        cover_threshold_days=float(overstock_settings.get("cover_threshold_days", 90)),
        slow_moving_daily_units=float(overstock_settings.get("slow_moving_daily_units", 0.1)),
        recent_units=recent_units,
    )

    row = {
        "sku": str(sku),
        "category": str(category),
        "decision_date": decision,
        "forecast_origin_date": origin,
        "forecast_cutoff_date": origin,
        "forecast_source": forecast_source,
        "current_stock": float(current_stock),
        "incoming_stock": float(incoming_stock),
        "incoming_due_within_lead_time": float(incoming_due_within_lead_time),
        "committed_stock": float(committed_stock),
        "committed_stock_used": bool(committed_stock_used),
        "inventory_position": float(position),
        "expected_daily_demand": float(expected_daily),
        "lead_time_days": int(lead),
        "review_period_days": int(round(review)),
        "protection_period_days": float(protection),
        "lead_time_demand": float(ltd.units),
        "protection_period_demand": float(protection_demand.units),
        "safety_stock": float(ss),
        "z_value": float(z),
        "sigma_lead_time_demand": float(sigma),
        "sigma_daily": float(sigma_daily),
        "reorder_point": float(rop),
        "target_stock": float(target),
        "recommended_order_qty": int(order.quantity),
        "raw_recommended_qty": float(order.raw_quantity),
        "days_of_cover": float(cover),
        "projected_stockout_date": risk.projected_stockout_date,
        "days_until_stockout": risk.days_until_stockout,
        "existing_stockout": bool(risk.existing_stockout),
        "stockout_risk": risk.risk,
        "reorder_status": status,
        "overstock_flag": bool(overstock.overstock),
        "excess_units": float(overstock.excess_units),
        "excess_cover_days": float(overstock.excess_cover_days),
        "recent_units": float(recent_units),
        "service_level": float(level),
        "safety_stock_method": method,
        "flags": _unique_flags(
            state_flags,
            ltd.flag_string,
            protection_demand.flag_string,
            ";".join(sigma_flags),
            order.flag_string,
            risk.flag_string,
            overstock.flag_string,
            ";".join(flags),
        ),
    }
    row["explanation"] = explain_recommendation(row)
    return row


def _category_fallbacks(
    data: InventoryData,
    history_frame: pd.DataFrame,
    state: pd.DataFrame,
    settings: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, float]]:
    histories = {
        str(row["sku"]): np.asarray(row["history"], dtype=float)
        for _, row in history_frame.iterrows()
        if isinstance(row["history"], np.ndarray)
    }
    categories = dict(zip(state["sku"].astype(str), state["category"].astype(str)))
    return category_volatility_fallback(
        histories,
        categories,
        min_history=int(settings["min_history_days_for_volatility"]),
    )


def build_recommendations(
    data: InventoryData,
    decision_date: Any,
    forecast: pd.DataFrame,
    config: Mapping[str, Any] | None = None,
    service_level: float | None = None,
    demand_multiplier: float = 1.0,
    lead_time_multiplier: float = 1.0,
    lead_time_extra_days: float = 0.0,
    forecast_source: str | None = None,
    use_committed_stock: bool | None = None,
) -> pd.DataFrame:
    """Build the SKU-level recommendation table for one decision date."""
    settings = inventory_settings(config)
    decision = as_timestamp(decision_date)
    origin = decision - pd.Timedelta(days=1)
    level = float(service_level if service_level is not None else settings["service_level"])
    use_committed = (
        bool(settings.get("use_committed_stock", False))
        if use_committed_stock is None
        else bool(use_committed_stock)
    )
    state = build_inventory_state(
        data,
        decision,
        use_committed_stock=use_committed,
        default_lead_time_days=int(settings["default_lead_time_days"]),
    )
    if state.empty:
        return pd.DataFrame(columns=list(RECOMMENDATION_COLUMNS))

    hist_window = max(
        int(settings["demand_volatility_window_days"]),
        4 * int(state["lead_time_days"].max()) + 7,
        60,
    )
    history_frame = daily_demand_history(data.daily, origin, hist_window, skus=state["sku"])
    history_lookup = {
        str(row["sku"]): np.asarray(row["history"], dtype=float) if isinstance(row["history"], np.ndarray) else np.array([])
        for _, row in history_frame.iterrows()
    }
    _sigma_map, category_sigma = _category_fallbacks(data, history_frame, state, settings)
    overall_sigma = float(category_sigma.get("__overall__", 0.0))

    forecast = forecast.copy()
    if not forecast.empty:
        forecast["sku"] = forecast["sku"].astype(str)
        forecast["date"] = pd.to_datetime(forecast["date"]).dt.normalize()
        forecast = forecast.sort_values(["sku", "date"]).drop_duplicates(subset=["sku", "date"], keep="last")
    forecast_lookup: dict[str, np.ndarray] = {}
    if forecast_source:
        forecast_source_name = str(forecast_source)
    elif "forecast_source" in forecast.columns and forecast["forecast_source"].notna().any():
        forecast_source_name = str(forecast["forecast_source"].dropna().iloc[0])
    elif "model" in forecast.columns and forecast["model"].notna().any():
        forecast_source_name = str(forecast["model"].dropna().iloc[0])
    else:
        forecast_source_name = "m3_forecast"
    if not forecast.empty:
        for sku, group in forecast.groupby("sku", observed=True):
            view = group.loc[group["date"] >= decision].sort_values("date")
            forecast_lookup[str(sku)] = (
                pd.to_numeric(view["predicted_units"], errors="coerce")
                .fillna(0.0)
                .clip(lower=0.0)
                .to_numpy(dtype=float)
            )

    recent_start = origin - pd.Timedelta(days=int(settings["overstock"]["inactive_history_days"]))
    recent = data.daily.loc[(data.daily["date"] > recent_start) & (data.daily["date"] <= origin)]
    recent_units = recent.groupby("sku", observed=True)["units_sold"].sum().to_dict()

    rows: list[dict[str, Any]] = []
    for _, row in state.iterrows():
        sku = str(row["sku"])
        category = str(row["category"])
        history = history_lookup.get(sku, np.array([], dtype=float))
        recent_window = history[-int(settings["demand_volatility_window_days"]) :] if len(history) else history
        fallback_daily = float(recent_window.mean()) if len(recent_window) else 0.0
        values = forecast_lookup.get(sku, np.array([], dtype=float))
        schedule = _incoming_schedule(data.purchase_orders, sku, decision)
        sigma_fallback = float(category_sigma.get(category, overall_sigma))
        metrics = recommendation_metrics(
            sku=sku,
            category=category,
            decision_date=decision,
            current_stock=float(row["current_stock"]),
            incoming_stock=float(row["incoming_stock"]),
            incoming_due_within_lead_time=float(row["incoming_due_within_lead_time"]),
            committed_stock=float(row["committed_stock"]),
            committed_stock_used=use_committed,
            lead_time_base_days=float(row["lead_time_days"]),
            history=history,
            forecast_values=values,
            fallback_daily=fallback_daily,
            recent_units=float(recent_units.get(sku, 0.0)),
            incoming_schedule=schedule,
            config=config,
            service_level=level,
            demand_multiplier=demand_multiplier,
            lead_time_multiplier=lead_time_multiplier,
            lead_time_extra_days=lead_time_extra_days,
            forecast_source=forecast_source_name,
            state_flags=str(row.get("state_flags", "")),
            fallback_sigma_daily=sigma_fallback,
        )
        metrics["dataset"] = data.mode
        rows.append(metrics)

    frame = pd.DataFrame(rows)
    for column in RECOMMENDATION_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    frame = frame.loc[:, list(RECOMMENDATION_COLUMNS)]
    frame["recommended_order_qty"] = frame["recommended_order_qty"].astype(int)
    frame["projected_stockout_date"] = pd.to_datetime(frame["projected_stockout_date"], errors="coerce")
    return frame.sort_values(["reorder_status", "stockout_risk", "sku"], kind="mergesort").reset_index(drop=True)

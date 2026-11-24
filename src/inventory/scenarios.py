"""Model-based inventory scenario analysis.

Each scenario reruns the same recommendation engine with one changed assumption:
a demand multiplier, a longer lead time or a different target service level.
These are labelled model-based scenarios, not causal forecasts or guaranteed
outcomes. Baseline demand, inventory state and supplier lead times are otherwise
held fixed.
"""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from src.inventory.data_builder import InventoryData
from src.inventory.engine import build_recommendations
from src.inventory.settings import inventory_settings

SCENARIO_SUMMARY_COLUMNS = (
    "scenario",
    "demand_multiplier",
    "lead_time_extra_days",
    "service_level",
    "skus",
    "reorder_required_skus",
    "at_risk_skus",
    "critical_skus",
    "overstock_skus",
    "recommended_order_units",
    "total_current_stock",
    "total_incoming_stock",
    "total_lead_time_demand",
    "total_safety_stock",
    "mean_reorder_point",
    "total_excess_units",
)


def _scenario_parameters(
    scenario: Mapping[str, Any],
    default_service_level: float,
) -> dict[str, Any]:
    return {
        "name": str(scenario.get("name", "scenario")),
        "demand_multiplier": float(scenario.get("demand_multiplier", 1.0)),
        "lead_time_multiplier": float(scenario.get("lead_time_multiplier", 1.0)),
        "lead_time_extra_days": float(scenario.get("lead_time_extra_days", 0.0)),
        "service_level": float(scenario.get("service_level", default_service_level)),
    }


def scenario_summary(frame: pd.DataFrame, parameters: Mapping[str, Any]) -> dict[str, Any]:
    if frame is None or frame.empty:
        return {column: None for column in SCENARIO_SUMMARY_COLUMNS}
    status = frame["reorder_status"].astype(str)
    return {
        "scenario": parameters["name"],
        "demand_multiplier": parameters["demand_multiplier"],
        "lead_time_extra_days": parameters["lead_time_extra_days"],
        "service_level": parameters["service_level"],
        "skus": int(len(frame)),
        "reorder_required_skus": int(status.isin(["Reorder Required", "At Risk", "Critical"]).sum()),
        "at_risk_skus": int((status == "At Risk").sum()),
        "critical_skus": int((status == "Critical").sum()),
        "overstock_skus": int(frame["overstock_flag"].astype(bool).sum()),
        "recommended_order_units": float(frame["recommended_order_qty"].sum()),
        "total_current_stock": float(frame["current_stock"].sum()),
        "total_incoming_stock": float(frame["incoming_stock"].sum()),
        "total_lead_time_demand": float(frame["lead_time_demand"].sum()),
        "total_safety_stock": float(frame["safety_stock"].sum()),
        "mean_reorder_point": float(frame["reorder_point"].mean()),
        "total_excess_units": float(frame["excess_units"].sum()),
    }


def run_scenarios(
    data: InventoryData,
    decision_date: Any,
    forecast: pd.DataFrame,
    config: Mapping[str, Any],
    scenarios: list[Mapping[str, Any]] | None = None,
    service_level: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run configured scenarios and return ``(summary, sku_details)``."""
    settings = inventory_settings(config)
    base_service = float(service_level if service_level is not None else settings["service_level"])
    configured = scenarios if scenarios is not None else list(settings.get("scenarios", []))
    if not configured:
        configured = [{"name": "baseline", "demand_multiplier": 1.0}]

    summaries: list[dict[str, Any]] = []
    details: list[pd.DataFrame] = []
    seen: set[str] = set()
    for scenario in configured:
        params = _scenario_parameters(scenario, base_service)
        name = params["name"]
        if name in seen:
            raise ValueError(f"Duplicate scenario name: {name}")
        seen.add(name)
        recommendations = build_recommendations(
            data,
            decision_date,
            forecast,
            config=config,
            service_level=params["service_level"],
            demand_multiplier=params["demand_multiplier"],
            lead_time_multiplier=params["lead_time_multiplier"],
            lead_time_extra_days=params["lead_time_extra_days"],
        )
        summaries.append(scenario_summary(recommendations, params))
        view = recommendations.copy()
        view.insert(0, "scenario", name)
        view.insert(1, "scenario_demand_multiplier", params["demand_multiplier"])
        view.insert(2, "scenario_lead_time_extra_days", params["lead_time_extra_days"])
        view.insert(3, "scenario_service_level", params["service_level"])
        details.append(view)

    summary = pd.DataFrame(summaries, columns=list(SCENARIO_SUMMARY_COLUMNS))
    detail = pd.concat(details, ignore_index=True) if details else pd.DataFrame()
    return summary, detail


def scenario_deltas(summary: pd.DataFrame, baseline: str = "baseline") -> pd.DataFrame:
    """Add differences versus the baseline scenario where available."""
    out = summary.copy()
    base = out.loc[out["scenario"] == baseline]
    if base.empty:
        return out
    base_row = base.iloc[0]
    metrics = [
        "reorder_required_skus",
        "at_risk_skus",
        "critical_skus",
        "overstock_skus",
        "recommended_order_units",
        "total_lead_time_demand",
        "total_safety_stock",
        "mean_reorder_point",
        "total_excess_units",
    ]
    for metric in metrics:
        if metric in out.columns:
            out[f"{metric}_delta_vs_{baseline}"] = pd.to_numeric(
                out[metric], errors="coerce"
            ) - float(base_row[metric])
    return out

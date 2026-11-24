"""Inventory-simulation evaluation metrics.

Definitions (also documented in ``docs/inventory_optimization.md``)
-------------------------------------------------------------------------
- **Service level (SKU-day cycle service):** share of simulated SKU-days where
  all demand proxy units were fulfilled.
- **Fill rate:** fulfilled units divided by demand proxy units.
- **Stockout frequency:** stockout SKU-days divided by all simulated SKU-days.
- **Lost sales:** unfulfilled demand proxy units. This is not latent demand.
- **Average inventory:** mean start-of-day on-hand stock across SKU-days
  (units per SKU-day).
- **Average portfolio inventory:** average total on-hand units across all SKUs
  per day, i.e. the SKU-day average multiplied by the number of SKUs.
- **Inventory turnover (units-based):** total demand-proxy units divided by the
  average portfolio inventory units over the window. This is the *period*
  turnover. It is **not** a financial (COGS/value) turnover because no validated
  product-cost data exists. An annualised value multiplies by 365/period_days.
- **Days of inventory:** period days divided by the period turnover, so it is
  exactly consistent with the annualised turnover.
- **Excess inventory:** average units above target stock across review
  decisions, based on the forecast-driven target at that decision.
- **Holding cost:** not calculated because no unit holding-cost input exists.

No monetary savings are claimed anywhere.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.inventory.simulation import SimulationResult

OVERALL_METRICS = (
    "skus",
    "sku_days",
    "period_days",
    "demand_proxy_units",
    "fulfilled_units",
    "lost_sales_units",
    "fill_rate",
    "service_level_sku_day",
    "stockout_frequency",
    "stockout_sku_days",
    "skus_with_stockout",
    "average_inventory_units",
    "average_portfolio_inventory_units",
    "average_closing_inventory_units",
    "ending_inventory_units",
    "inventory_turnover",
    "annualized_inventory_turnover",
    "days_of_inventory",
    "excess_inventory_units",
    "overstocked_decision_share",
    "replenishment_count",
    "replenishment_units",
    "holding_cost",
)

PER_SKU_METRICS = (
    "sku",
    "category",
    "sku_days",
    "demand_proxy_units",
    "fulfilled_units",
    "lost_sales_units",
    "fill_rate",
    "stockout_sku_days",
    "stockout_frequency",
    "service_level_sku_day",
    "average_inventory_units",
    "ending_inventory_units",
    "inventory_turnover",
    "days_of_inventory",
    "replenishment_count",
    "replenishment_units",
    "excess_inventory_units",
    "overstocked_decision_share",
)


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator is None or not np.isfinite(float(denominator)) or float(denominator) == 0:
        return float("nan")
    return float(numerator) / float(denominator)


def simulation_metrics(
    daily: pd.DataFrame,
    decisions: pd.DataFrame,
    orders: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Return overall and per-SKU metrics for one simulated policy."""
    if daily is None or daily.empty:
        raise ValueError("simulation daily table is empty")
    overall_demand = float(daily["demand_proxy_units"].sum())
    overall_fulfilled = float(daily["fulfilled_units"].sum())
    stockout_sku_days = int(daily["stockout_flag"].sum())
    sku_days = int(len(daily))
    period_days = int(daily["date"].nunique())
    n_skus = int(daily["sku"].nunique())
    average_inventory = float(daily["opening_on_hand"].mean())
    average_portfolio_inventory = average_inventory * n_skus
    ending = (
        daily.sort_values(["sku", "date"]).groupby("sku", observed=True).tail(1)["closing_on_hand"].sum()
    )
    turnover = _safe_div(overall_demand, average_portfolio_inventory)
    annualized = turnover * 365.0 / period_days if period_days > 0 and np.isfinite(turnover) else float("nan")
    days_of_inventory = _safe_div(float(period_days), turnover) if np.isfinite(turnover) else float("nan")

    if decisions is None or decisions.empty:
        excess_avg = float("nan")
        overstock_share = float("nan")
        excess_by_sku = pd.DataFrame(columns=["sku", "excess_inventory_units", "overstocked_decision_share"])
    else:
        decision_dates = decisions["decision_date"].nunique()
        excess_avg = _safe_div(float(decisions["excess_units"].sum()), decision_dates)
        overstock_share = float(decisions["overstock_flag"].astype(bool).mean())
        excess_by_sku = (
            decisions.groupby("sku", observed=True)
            .agg(
                excess_inventory_units=("excess_units", lambda values: float(values.sum()) / max(1, decision_dates)),
                overstocked_decision_share=("overstock_flag", "mean"),
            )
            .reset_index()
        )

    overall = {
        "skus": int(daily["sku"].nunique()),
        "sku_days": sku_days,
        "period_days": period_days,
        "demand_proxy_units": overall_demand,
        "fulfilled_units": overall_fulfilled,
        "lost_sales_units": float(daily["lost_units"].sum()),
        "fill_rate": _safe_div(overall_fulfilled, overall_demand),
        "service_level_sku_day": 1.0 - _safe_div(stockout_sku_days, sku_days),
        "stockout_frequency": _safe_div(stockout_sku_days, sku_days),
        "stockout_sku_days": stockout_sku_days,
        "skus_with_stockout": int(daily.loc[daily["stockout_flag"] == 1, "sku"].nunique()),
        "average_inventory_units": average_inventory,
        "average_portfolio_inventory_units": average_portfolio_inventory,
        "average_closing_inventory_units": float(daily["closing_on_hand"].mean()),
        "ending_inventory_units": float(ending.sum()),
        "inventory_turnover": turnover,
        "annualized_inventory_turnover": annualized,
        "days_of_inventory": days_of_inventory,
        "excess_inventory_units": excess_avg,
        "overstocked_decision_share": overstock_share,
        "replenishment_count": int(len(orders)) if orders is not None else 0,
        "replenishment_units": float(orders["quantity"].sum()) if orders is not None and not orders.empty else 0.0,
        "holding_cost": None,
        "holding_cost_note": "Not calculated: no validated unit holding-cost input exists.",
    }

    per_sku = (
        daily.groupby("sku", observed=True)
        .agg(
            category=("category", "first"),
            sku_days=("date", "count"),
            demand_proxy_units=("demand_proxy_units", "sum"),
            fulfilled_units=("fulfilled_units", "sum"),
            lost_sales_units=("lost_units", "sum"),
            stockout_sku_days=("stockout_flag", "sum"),
            average_inventory_units=("opening_on_hand", "mean"),
            average_closing_inventory_units=("closing_on_hand", "mean"),
        )
        .reset_index()
    )
    ending_by_sku = daily.sort_values(["sku", "date"]).groupby("sku", observed=True).tail(1)[
        ["sku", "closing_on_hand"]
    ].rename(columns={"closing_on_hand": "ending_inventory_units"})
    per_sku = per_sku.merge(ending_by_sku, on="sku", how="left")
    per_sku["fill_rate"] = per_sku.apply(
        lambda row: _safe_div(row["fulfilled_units"], row["demand_proxy_units"]), axis=1
    )
    per_sku["stockout_frequency"] = per_sku.apply(
        lambda row: _safe_div(row["stockout_sku_days"], row["sku_days"]), axis=1
    )
    per_sku["service_level_sku_day"] = 1.0 - per_sku["stockout_frequency"]
    per_sku["inventory_turnover"] = per_sku.apply(
        lambda row: _safe_div(row["demand_proxy_units"], row["average_inventory_units"]), axis=1
    )
    per_sku["days_of_inventory"] = per_sku.apply(
        lambda row: _safe_div(
            row["average_inventory_units"],
            _safe_div(row["demand_proxy_units"], row["sku_days"]),
        ),
        axis=1,
    )
    orders_by_sku = (
        orders.groupby("sku", observed=True).agg(
            replenishment_count=("po_id", "count"),
            replenishment_units=("quantity", "sum"),
        )
        if orders is not None and not orders.empty
        else pd.DataFrame(columns=["sku", "replenishment_count", "replenishment_units"])
    )
    per_sku = per_sku.merge(orders_by_sku, on="sku", how="left")
    for column in ("replenishment_count", "replenishment_units"):
        per_sku[column] = pd.to_numeric(per_sku[column], errors="coerce").fillna(0.0)
    per_sku["replenishment_count"] = per_sku["replenishment_count"].astype(int)
    per_sku = per_sku.merge(excess_by_sku, on="sku", how="left")
    for column in ("excess_inventory_units", "overstocked_decision_share"):
        per_sku[column] = pd.to_numeric(per_sku[column], errors="coerce").fillna(0.0)
    for column in PER_SKU_METRICS:
        if column not in per_sku.columns:
            per_sku[column] = np.nan
    per_sku = per_sku.loc[:, list(PER_SKU_METRICS)].sort_values("sku").reset_index(drop=True)
    return overall, per_sku


def evaluate_simulation(result: SimulationResult) -> tuple[dict[str, Any], pd.DataFrame]:
    overall, per_sku = simulation_metrics(result.daily, result.decisions, result.orders)
    overall["policy"] = result.policy
    overall["dataset"] = result.metadata.get("dataset", "synthetic")
    per_sku.insert(0, "policy", result.policy)
    return overall, per_sku


def policy_comparison(
    results: Mapping[str, SimulationResult],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]]]:
    """Evaluate several policies and return comparison, per-SKU and raw metrics."""
    overall_rows: list[dict[str, Any]] = []
    per_sku_frames: list[pd.DataFrame] = []
    metrics_by_policy: dict[str, dict[str, Any]] = {}
    for policy, result in results.items():
        overall, per_sku = evaluate_simulation(result)
        metrics_by_policy[policy] = overall
        overall_rows.append(overall)
        per_sku_frames.append(per_sku)
    comparison = pd.DataFrame(overall_rows)
    ordered = ["policy", *[column for column in OVERALL_METRICS if column in comparison.columns], "dataset"]
    comparison = comparison.loc[:, [column for column in ordered if column in comparison.columns]]
    per_sku = pd.concat(per_sku_frames, ignore_index=True) if per_sku_frames else pd.DataFrame()
    return comparison, per_sku, metrics_by_policy


def add_relative_to_policy(
    comparison: pd.DataFrame,
    baseline_policy: str = "forecast_driven",
    metrics: tuple[str, ...] = (
        "fill_rate",
        "lost_sales_units",
        "service_level_sku_day",
        "average_inventory_units",
        "inventory_turnover",
        "replenishment_count",
    ),
) -> pd.DataFrame:
    """Add relative differences versus a baseline policy row."""
    out = comparison.copy()
    baseline = out.loc[out["policy"] == baseline_policy]
    if baseline.empty:
        return out
    base = baseline.iloc[0]
    for metric in metrics:
        if metric not in out.columns:
            continue
        base_value = base[metric]
        if base_value is None or not np.isfinite(float(base_value)) or float(base_value) == 0:
            continue
        out[f"{metric}_rel_vs_{baseline_policy}"] = (
            pd.to_numeric(out[metric], errors="coerce") / float(base_value) - 1.0
        )
    return out

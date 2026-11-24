"""Machine-readable and human-readable inventory reports.

Outputs are written under ``reports/inventory/`` only. Existing M2 EDA, M3
forecasting reports and their schemas are never overwritten.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.inventory.simulation import SIMULATION_DECISION_COLUMNS, SIMULATION_ORDER_COLUMNS
from src.utils.io import save_parquet, write_json
from src.utils.paths import resolve_path

RECOMMENDATION_HEADERS = {
    "sku": "SKU",
    "category": "Category",
    "decision_date": "Decision date",
    "forecast_origin_date": "Forecast origin",
    "current_stock": "Current stock",
    "incoming_stock": "Incoming stock",
    "inventory_position": "Inventory position",
    "lead_time_days": "Lead time (d)",
    "lead_time_demand": "LTD (units)",
    "safety_stock": "Safety stock",
    "reorder_point": "Reorder point",
    "target_stock": "Target stock",
    "recommended_order_qty": "Order qty",
    "days_of_cover": "Days cover",
    "stockout_risk": "Stockout risk",
    "reorder_status": "Status",
    "overstock_flag": "Overstock",
    "excess_units": "Excess units",
    "flags": "Flags",
}

SIMULATION_HEADERS = {
    "policy": "Policy",
    "skus": "SKUs",
    "sku_days": "SKU-days",
    "demand_proxy_units": "Demand proxy units",
    "fulfilled_units": "Fulfilled units",
    "lost_sales_units": "Lost sales proxy units",
    "fill_rate": "Fill rate",
    "service_level_sku_day": "Service level (SKU-day)",
    "stockout_frequency": "Stockout frequency",
    "average_inventory_units": "Avg inventory (units/SKU-day)",
    "average_portfolio_inventory_units": "Avg portfolio inventory (units/day)",
    "inventory_turnover": "Turnover (units, period)",
    "annualized_inventory_turnover": "Annualised turnover (units)",
    "days_of_inventory": "Days of inventory",
    "excess_inventory_units": "Excess units (avg/decision)",
    "replenishment_count": "Orders",
    "replenishment_units": "Order units",
    "holding_cost": "Holding cost",
}


RISK_ORDER = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}


def _risk_rank(values: pd.Series) -> pd.Series:
    return values.astype(str).map(RISK_ORDER).fillna(len(RISK_ORDER)).astype(int)


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (bool, np.bool_)):
        return "yes" if bool(value) else "no"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "n/a"
    return f"{number:,.{digits}f}"


def _pct(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "n/a"
    return f"{number:.2%}"


def _md_table(frame: pd.DataFrame, columns: list[str], headers: Mapping[str, str] | None = None, max_rows: int | None = None) -> str:
    if frame is None or frame.empty:
        return "_No rows._"
    available = [column for column in columns if column in frame.columns]
    if not available:
        return "_No rows._"
    view = frame.loc[:, available]
    if max_rows is not None:
        view = view.head(max_rows)
    headers = headers or {}
    label = [headers.get(column, column) for column in available]
    lines = ["| " + " | ".join(label) + " |", "|" + "|".join(["---"] * len(label)) + "|"]
    for _, row in view.iterrows():
        values = []
        for column in available:
            value = row[column]
            if column in {"fill_rate", "service_level_sku_day", "stockout_frequency", "overstocked_decision_share"}:
                values.append(_pct(value))
            elif isinstance(value, (float, np.floating)):
                values.append(_fmt(value))
            elif isinstance(value, (int, np.integer)):
                values.append(f"{int(value):,}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def recommendation_summary(frame: pd.DataFrame) -> dict[str, Any]:
    if frame is None or frame.empty:
        return {"skus": 0}
    status = frame["reorder_status"].astype(str)
    risk = frame["stockout_risk"].astype(str)
    return {
        "skus": int(len(frame)),
        "reorder_required_skus": int(status.isin(["Reorder Required", "At Risk", "Critical"]).sum()),
        "at_risk_skus": int((status == "At Risk").sum()),
        "critical_skus": int((status == "Critical").sum()),
        "healthy_skus": int((status == "Healthy").sum()),
        "stockout_risk_high": int((risk == "High").sum()),
        "stockout_risk_critical": int((risk == "Critical").sum()),
        "existing_stockouts": int(frame["existing_stockout"].astype(bool).sum()),
        "overstock_skus": int(frame["overstock_flag"].astype(bool).sum()),
        "total_current_stock": float(frame["current_stock"].sum()),
        "total_incoming_stock": float(frame["incoming_stock"].sum()),
        "total_inventory_position": float(frame["inventory_position"].sum()),
        "total_recommended_order_units": float(frame["recommended_order_qty"].sum()),
        "total_excess_units": float(frame["excess_units"].sum()),
        "mean_days_of_cover": float(pd.to_numeric(frame["days_of_cover"], errors="coerce").replace([np.inf, -np.inf], np.nan).mean()),
        "holding_cost": None,
    }


def write_recommendation_outputs(
    mode: str,
    decision_date: Any,
    recommendations: pd.DataFrame,
    output_dir: str | Path,
    metadata: Mapping[str, Any] | None = None,
    scenario_summary: pd.DataFrame | None = None,
    top_n: int = 25,
) -> dict[str, Path]:
    target = resolve_path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    paths["recommendations"] = save_parquet(recommendations, target / "recommendations.parquet")
    paths["recommendations_csv"] = target / "recommendations.csv"
    recommendations.to_csv(paths["recommendations_csv"], index=False, encoding="utf-8")

    reorder = recommendations.loc[
        recommendations["recommended_order_qty"] > 0
    ].copy()
    reorder["_risk_rank"] = _risk_rank(reorder["stockout_risk"])
    reorder = reorder.sort_values(["_risk_rank", "recommended_order_qty"], ascending=[True, False]).drop(
        columns="_risk_rank"
    )
    critical = recommendations.loc[
        recommendations["reorder_status"].astype(str).isin(["At Risk", "Critical"])
        | recommendations["existing_stockout"].astype(bool)
    ].copy()
    critical["_risk_rank"] = _risk_rank(critical["stockout_risk"])
    critical = critical.sort_values(["_risk_rank", "days_of_cover"], ascending=[True, True]).drop(
        columns="_risk_rank"
    )
    overstock = recommendations.loc[recommendations["overstock_flag"].astype(bool)].sort_values(
        "excess_units", ascending=False
    )
    cover = recommendations.sort_values("days_of_cover")
    for name, frame in (
        ("reorder_list.csv", reorder),
        ("critical_stockout_list.csv", critical),
        ("overstock_list.csv", overstock),
        ("days_of_cover.csv", cover),
    ):
        path = target / name
        frame.to_csv(path, index=False, encoding="utf-8")
        paths[name] = path

    summary = {
        "dataset": mode,
        "decision_date": str(pd.Timestamp(decision_date).date()) if decision_date is not None else None,
        "forecast_origin_date": (
            str(recommendations["forecast_origin_date"].iloc[0].date()) if not recommendations.empty else None
        ),
        "forecast_cutoff_date": (
            str(recommendations["forecast_cutoff_date"].iloc[0].date()) if not recommendations.empty else None
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provenance": "Fully simulated Indian jewellery data. Not real company data.",
        "metrics": recommendation_summary(recommendations),
        "assumptions": dict(metadata or {}),
        "scenario_names": list(scenario_summary["scenario"]) if scenario_summary is not None and not scenario_summary.empty else [],
        "limitations": [
            "Inventory optimization is supported only for the synthetic dataset.",
            "UCI and M5 have no inventory or purchase-order records and are never assigned inventory.",
            "Forecasts target observed sales; synthetic sales are stockout-censored and are not latent demand.",
            "Safety stock is an approximation under documented assumptions; a configured service level is a target, not a guarantee.",
            "No supplier cost, purchase cost or holding cost exists in the dataset, so no financial value or savings are claimed.",
            "Risk labels are model-based warnings, not certainty.",
        ],
    }
    paths["summary"] = write_json(summary, target / "summary.json")

    lines: list[str] = []
    lines.append("# Inventory Optimization Report")
    lines.append("")
    lines.append(
        "**Provenance:** Fully simulated Indian jewellery data. Not real company data; "
        "this is not Voylla or any real retailer's inventory."
    )
    lines.append("")
    lines.append(f"- Dataset mode: `{mode}`")
    if decision_date is not None:
        lines.append(f"- Decision date: {pd.Timestamp(decision_date).date()}")
    if not recommendations.empty:
        lines.append(f"- Forecast origin / cutoff: {recommendations['forecast_origin_date'].iloc[0].date()}")
        lines.append(f"- Forecast source: `{recommendations['forecast_source'].iloc[0]}`")
        lines.append(f"- SKUs evaluated: {len(recommendations):,}")
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append("- Inventory position = start-of-day available stock + pending incoming purchase orders - committed stock.")
    lines.append("- Committed stock is excluded from decisions by default because the synthetic snapshots derived it from future demand.")
    lines.append("- Lead-time demand is the sum of the daily M3 forecast over the supplier lead time, not a flat average.")
    lines.append("- Safety stock = z(target service level) x sigma(lead-time demand). The default sigma is trailing daily demand volatility x sqrt(lead time).")
    lines.append("- Reorder point = lead-time demand + safety stock; target stock covers lead time plus review period plus safety stock.")
    lines.append("- A purchase order is pending once per `po_id`; incoming orders are never double-counted.")
    lines.append("- A configured service level is a planning target, not a measured guarantee.")
    lines.append("")
    lines.append("## Current recommendations")
    lines.append("")
    metrics = summary["metrics"]
    metric_rows = pd.DataFrame(
        [
            ["SKUs", metrics.get("skus", 0)],
            ["Reorder required", metrics.get("reorder_required_skus", 0)],
            ["At risk or critical", metrics.get("at_risk_skus", 0) + metrics.get("critical_skus", 0)],
            ["Existing stockouts", metrics.get("existing_stockouts", 0)],
            ["Overstock SKUs", metrics.get("overstock_skus", 0)],
            ["Total current stock (units)", metrics.get("total_current_stock", 0)],
            ["Total incoming stock (units)", metrics.get("total_incoming_stock", 0)],
            ["Recommended order units", metrics.get("total_recommended_order_units", 0)],
            ["Total excess units", metrics.get("total_excess_units", 0)],
        ],
        columns=["Item", "Value"],
    )
    lines.append(_md_table(metric_rows, ["Item", "Value"]))
    lines.append("")
    lines.append("### Reorder list (top by urgency)")
    lines.append("")
    lines.append(
        _md_table(
            reorder,
            [
                "sku",
                "category",
                "inventory_position",
                "lead_time_days",
                "lead_time_demand",
                "safety_stock",
                "reorder_point",
                "target_stock",
                "recommended_order_qty",
                "stockout_risk",
                "reorder_status",
                "flags",
            ],
            RECOMMENDATION_HEADERS,
            max_rows=top_n,
        )
    )
    lines.append("")
    lines.append("### Critical / at-risk stockout list")
    lines.append("")
    lines.append(
        _md_table(
            critical,
            [
                "sku",
                "current_stock",
                "incoming_stock",
                "days_of_cover",
                "lead_time_days",
                "projected_stockout_date",
                "stockout_risk",
                "reorder_status",
                "recommended_order_qty",
                "flags",
            ],
            RECOMMENDATION_HEADERS,
            max_rows=top_n,
        )
    )
    lines.append("")
    lines.append("### Overstock list")
    lines.append("")
    lines.append(
        _md_table(
            overstock,
            [
                "sku",
                "category",
                "inventory_position",
                "expected_daily_demand",
                "days_of_cover",
                "excess_units",
                "excess_cover_days",
                "flags",
            ],
            RECOMMENDATION_HEADERS,
            max_rows=top_n,
        )
    )
    lines.append("")
    if scenario_summary is not None and not scenario_summary.empty:
        lines.append("## Scenario analysis (model-based)")
        lines.append("")
        lines.append(
            "Scenarios change one assumption at a time and rerun the same engine. They are not causal forecasts or guarantees."
        )
        lines.append("")
        lines.append(
            _md_table(
                scenario_summary,
                [
                    "scenario",
                    "demand_multiplier",
                    "lead_time_extra_days",
                    "service_level",
                    "reorder_required_skus",
                    "critical_skus",
                    "recommended_order_units",
                    "total_lead_time_demand",
                    "total_safety_stock",
                    "total_excess_units",
                ],
            )
        )
        lines.append("")
    lines.append("## Explainability example")
    lines.append("")
    if not recommendations.empty:
        example_view = recommendations.copy()
        example_view["_risk_rank"] = _risk_rank(example_view["stockout_risk"])
        example = example_view.sort_values(
            ["_risk_rank", "recommended_order_qty"], ascending=[True, False]
        ).iloc[0]
        lines.append(f"> {example['explanation']}")
    else:
        lines.append("_No recommendations available._")
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    for item in summary["limitations"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Reproduce")
    lines.append("")
    lines.append("```powershell")
    lines.append("python scripts/run_inventory_optimization.py --mode synthetic")
    lines.append("```")
    lines.append("")
    report_path = target / "INVENTORY_REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    paths["report"] = report_path
    return paths


def write_scenario_outputs(
    summary: pd.DataFrame,
    details: pd.DataFrame,
    output_dir: str | Path,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    target = resolve_path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    paths["summary_csv"] = target / "scenario_summary.csv"
    summary.to_csv(paths["summary_csv"], index=False, encoding="utf-8")
    paths["details"] = save_parquet(details, target / "scenario_details.parquet")
    paths["summary_json"] = write_json(
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "provenance": "Fully simulated Indian jewellery data. Not real company data.",
            "label": "Model-based scenario analysis, not causal or guaranteed predictions.",
            "assumptions": dict(metadata or {}),
            "scenarios": summary.to_dict(orient="records"),
        },
        target / "scenarios.json",
    )
    lines = [
        "# Inventory Scenario Analysis",
        "",
        "Model-based scenarios only. These are not causal effects or guaranteed business outcomes.",
        "",
        "Every scenario uses the same inventory state, forecast and supplier lead times except for the one changed assumption.",
        "",
        _md_table(
            summary,
            [
                "scenario",
                "demand_multiplier",
                "lead_time_extra_days",
                "service_level",
                "reorder_required_skus",
                "critical_skus",
                "recommended_order_units",
                "total_lead_time_demand",
                "total_safety_stock",
                "total_excess_units",
            ],
        ),
        "",
        "## Limitations",
        "",
        "- Demand multipliers are applied to the forecast over the whole horizon; they are not observed uplift.",
        "- A higher service level raises safety stock by construction; it does not prove that service is achieved.",
        "- No cost or holding-cost inputs exist, so no monetary savings are reported.",
        "",
        "```powershell",
        "python scripts/run_inventory_optimization.py --mode synthetic",
        "```",
        "",
    ]
    report = target / "SCENARIO_REPORT.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    paths["report"] = report
    return paths


def write_simulation_outputs(
    results: Mapping[str, Any],
    comparison: pd.DataFrame,
    per_sku: pd.DataFrame,
    output_dir: str | Path,
    metadata: Mapping[str, Any] | None = None,
    top_n: int = 25,
) -> dict[str, Path]:
    target = resolve_path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    daily = pd.concat([result.daily for result in results.values()], ignore_index=True)
    decision_frames = [result.decisions for result in results.values() if not result.decisions.empty]
    order_frames = [result.orders for result in results.values() if not result.orders.empty]
    decisions = (
        pd.concat(decision_frames, ignore_index=True)
        if decision_frames
        else pd.DataFrame(columns=list(SIMULATION_DECISION_COLUMNS))
    )
    orders = (
        pd.concat(order_frames, ignore_index=True)
        if order_frames
        else pd.DataFrame(columns=list(SIMULATION_ORDER_COLUMNS))
    )
    paths["daily"] = save_parquet(daily, target / "simulation_daily.parquet")
    paths["decisions"] = save_parquet(decisions, target / "simulation_decisions.parquet")
    paths["orders"] = save_parquet(orders, target / "simulation_orders.parquet")
    paths["comparison_csv"] = target / "policy_comparison.csv"
    comparison.to_csv(paths["comparison_csv"], index=False, encoding="utf-8")
    paths["metrics_by_sku"] = target / "policy_metrics_by_sku.csv"
    per_sku.to_csv(paths["metrics_by_sku"], index=False, encoding="utf-8")
    paths["summary"] = write_json(
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "provenance": "Fully simulated Indian jewellery data. Not real company data.",
            "demand_definition": (
                "Observed stockout-censored units_sold used as a common demand proxy; "
                "true latent demand is not stored by the generator."
            ),
            "look_ahead_guard": "Forecast origins are always decision_date - 1 day.",
            "policies": {name: result.metadata for name, result in results.items()},
            "assumptions": dict(metadata or {}),
            "metrics": comparison.to_dict(orient="records"),
            "holding_cost": None,
            "limitations": [
                "Simulation results measure behaviour under synthetic assumptions, not real business performance.",
                "Synthetic observed sales are stockout-censored, so lost sales are understated relative to latent demand.",
                "Historical replenishment behaviour cannot be reconstructed fairly because latent demand is unavailable.",
                "No holding, purchase or supplier cost data exists; no financial savings are claimed.",
                "A configured service level is a policy target, not a guarantee.",
            ],
        },
        target / "simulation_summary.json",
    )

    lines: list[str] = []
    lines.append("# Historical Inventory Simulation")
    lines.append("")
    lines.append("**Provenance:** Fully simulated Indian jewellery data. Not real company data or a real business result.")
    lines.append("")
    for name, result in results.items():
        meta = result.metadata
        lines.append(
            f"- `{name}`: {meta['skus']} SKUs, {meta['start_date']} to {meta['end_date']}, "
            f"review every {meta['decision_frequency_days']} days, forecast source `{meta['forecast_source']}`"
        )
    lines.append("")
    lines.append("## Methodology")
    lines.append("")
    lines.append("- At each review date the decision uses start-of-day stock and a forecast whose origin is the previous day.")
    lines.append("- Orders arrive after the SKU's supplier lead time; daily observed demand is then served from stock.")
    lines.append("- Every policy starts from the same on-hand stock and (by default) an empty pipeline, and faces the same demand proxy and lead times.")
    lines.append("- Lost sales are unfulfilled demand-proxy units. They are **not** reconstructed latent demand.")
    lines.append("- Inventory turnover is units-based: total demand-proxy units divided by average total on-hand inventory units across all SKUs in the window, annualised by 365/period days. It is not a financial or COGS turnover because no validated cost input exists.")
    lines.append("- Days of inventory = period days / period turnover, so it is exactly consistent with the annualised turnover.")
    lines.append("")
    lines.append("## Policy comparison")
    lines.append("")
    lines.append(
        _md_table(
            comparison,
            [
                "policy",
                "skus",
                "sku_days",
                "demand_proxy_units",
                "fulfilled_units",
                "lost_sales_units",
                "fill_rate",
                "service_level_sku_day",
                "stockout_frequency",
                "average_inventory_units",
                "inventory_turnover",
                "annualized_inventory_turnover",
                "days_of_inventory",
                "excess_inventory_units",
                "replenishment_count",
                "replenishment_units",
            ],
            SIMULATION_HEADERS,
        )
    )
    lines.append("")
    lines.append(
        "Trade-off reminder: a policy that improves service level may also hold more average inventory and place more orders. "
        "Holding cost is not reported because no validated unit holding-cost input exists."
    )
    lines.append("")
    if not per_sku.empty and "policy" in per_sku.columns:
        pivot = per_sku.pivot_table(
            index=["sku", "category"],
            columns="policy",
            values=["fill_rate", "average_inventory_units", "replenishment_count"],
            aggfunc="first",
        )
        pivot.columns = [f"{metric}_{policy}" for metric, policy in pivot.columns]
        pivot = pivot.reset_index()
        lines.append("### Per-SKU differences (top fill-rate gaps)")
        if "fill_rate_forecast_driven" in pivot.columns and "fill_rate_simple_rop" in pivot.columns:
            pivot["fill_rate_gap_forecast_minus_simple"] = pivot["fill_rate_forecast_driven"] - pivot["fill_rate_simple_rop"]
            pivot = pivot.sort_values("fill_rate_gap_forecast_minus_simple")
        lines.append(_md_table(pivot, list(pivot.columns), max_rows=top_n))
        lines.append("")
    lines.append("## Limitations")
    lines.append("")
    for item in [
        "Synthetic observed sales are stockout-censored; absolute fill rates may be optimistic.",
        "Simulation is not proof of real-world performance and does not model supplier reliability, MOQ rules or holding costs.",
        "Historical replenishment behaviour is not reconstructable fairly because true latent demand was not stored.",
        "No financial savings are claimed.",
    ]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("```powershell")
    lines.append("python scripts/simulate_inventory.py --mode synthetic")
    lines.append("python scripts/evaluate_inventory.py")
    lines.append("```")
    lines.append("")
    report = target / "SIMULATION_REPORT.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    paths["report"] = report
    return paths


def write_evaluation_outputs(
    comparison: pd.DataFrame,
    per_sku: pd.DataFrame,
    output_dir: str | Path,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    """Write metric-only outputs for the evaluation script."""
    target = resolve_path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    paths["comparison_csv"] = target / "evaluation_policy_comparison.csv"
    comparison.to_csv(paths["comparison_csv"], index=False, encoding="utf-8")
    paths["metrics_by_sku"] = target / "evaluation_metrics_by_sku.csv"
    per_sku.to_csv(paths["metrics_by_sku"], index=False, encoding="utf-8")
    paths["summary"] = write_json(
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "provenance": "Derived from synthetic inventory simulation outputs.",
            "metrics": comparison.to_dict(orient="records"),
            "holding_cost": None,
            "assumptions": dict(metadata or {}),
            "limitations": [
                "Holding cost is unavailable and not calculated.",
                "Lost sales use the observed-sales demand proxy, not latent demand.",
                "Results are not proof of real business performance.",
            ],
        },
        target / "evaluation_summary.json",
    )
    lines = [
        "# Inventory Policy Evaluation",
        "",
        "Metrics are computed from the persisted synthetic historical simulation. No monetary savings are claimed.",
        "",
        _md_table(
            comparison,
            [
                "policy",
                "demand_proxy_units",
                "fulfilled_units",
                "lost_sales_units",
                "fill_rate",
                "service_level_sku_day",
                "stockout_frequency",
                "average_inventory_units",
                "inventory_turnover",
                "annualized_inventory_turnover",
                "days_of_inventory",
                "replenishment_count",
            ],
            SIMULATION_HEADERS,
        ),
        "",
        "Turnover is units-based (no validated cost/COGS input), annualised by 365/period days; days of inventory = period days / period turnover.",
        "",
        "Holding cost: not calculated (no validated cost input).",
        "",
    ]
    report = target / "EVALUATION_REPORT.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    paths["report"] = report
    return paths

"""Generate SKU-level M4 inventory recommendations and scenario comparisons."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.forecasting.data_builder import load_forecasting_panel
from src.inventory.data_builder import load_inventory_data
from src.inventory.engine import build_recommendations
from src.inventory.forecasting_adapter import (
    ModelForecastProvider,
    SavedModelForecastProvider,
)
from src.inventory.reporting import (
    recommendation_summary,
    write_recommendation_outputs,
    write_scenario_outputs,
)
from src.inventory.scenarios import run_scenarios, scenario_deltas
from src.inventory.settings import inventory_settings
from src.utils.config import load_config
from src.utils.paths import resolve_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run M4 inventory optimization for one decision date.")
    parser.add_argument("--mode", default="synthetic", help="Only synthetic is supported (default).")
    parser.add_argument("--decision-date", default=None, help="Decision date (default: last observed date).")
    parser.add_argument("--horizon", type=int, default=None, help="Forecast horizon in days (default: inventory.forecast_horizon).")
    parser.add_argument("--service-level", type=float, default=None, help="Override target service level.")
    parser.add_argument("--sku-limit", type=int, default=None, help="Limit to the top-N SKUs by units.")
    parser.add_argument("--processed-dir", default=None, help="Override the processed data directory.")
    parser.add_argument("--output", default=None, help="Override the report output directory.")
    parser.add_argument("--model-dir", default=None, help="Use an existing causal M3 model artefact instead of refitting.")
    parser.add_argument("--no-scenarios", action="store_true", help="Skip scenario analysis.")
    parser.add_argument("--top-n", type=int, default=None, help="Rows shown in Markdown tables.")
    args = parser.parse_args()

    config = load_config()
    settings = inventory_settings(config)
    mode = str(args.mode or settings["mode"])
    if mode != "synthetic":
        print("Inventory optimization is supported only for the synthetic dataset.")
        return 2

    processed_dir = resolve_path(args.processed_dir or config["paths"]["data_processed"])
    output_dir = resolve_path(args.output or settings["reporting"]["output_dir"])
    horizon = int(args.horizon or settings["forecast_horizon"])
    top_n = int(args.top_n or settings["reporting"]["top_n"])

    print(f"Loading synthetic inventory data and forecasting panel ({mode})...")
    data = load_inventory_data(mode, config, processed_dir=processed_dir, sku_limit=args.sku_limit)
    panel = load_forecasting_panel(mode, config, processed_dir=processed_dir, sku_limit=args.sku_limit)
    decision_date = pd.Timestamp(args.decision_date).normalize() if args.decision_date else panel.frame["date"].max().normalize()
    origin = decision_date - pd.Timedelta(days=1)
    if origin < panel.frame["date"].min():
        print(f"Forecast origin {origin.date()} is before the first observed date.")
        return 2

    if args.model_dir:
        model_dir = resolve_path(args.model_dir)
        print(f"Using saved M3 model artefact: {model_dir}")
        provider = SavedModelForecastProvider(panel, config, model_dir)
    else:
        print(f"Fitting a causal M3 model at forecast origin {origin.date()}...")
        provider = ModelForecastProvider(
            panel,
            config,
            mode="frozen",
            max_training_rows=int(settings["simulation"].get("max_training_rows", 400000)),
        )

    forecast = provider.forecast(origin, horizon)
    print(f"Forecast rows: {len(forecast):,} from origin {origin.date()}")

    recommendations = build_recommendations(
        data,
        decision_date,
        forecast,
        config=config,
        service_level=args.service_level,
        forecast_source=getattr(provider, "name", "m3_forecast"),
    )
    metrics = recommendation_summary(recommendations)
    print(
        f"Decision {decision_date.date()}: SKUs={metrics.get('skus', 0)} "
        f"reorder={metrics.get('reorder_required_skus', 0)} "
        f"critical={metrics.get('critical_skus', 0)} "
        f"order_units={metrics.get('total_recommended_order_units', 0):,.0f}"
    )

    metadata = {
        "mode": mode,
        "decision_date": str(decision_date.date()),
        "forecast_origin_date": str(origin.date()),
        "forecast_horizon": horizon,
        "forecast_source": getattr(provider, "name", "m3_forecast"),
        "forecast_fit_origins": [str(value.date()) for value in getattr(provider, "fit_origins", []) if value is not None],
        "service_level": float(args.service_level if args.service_level is not None else settings["service_level"]),
        "safety_stock_method": str(settings["safety_stock_method"]),
        "review_period_days": int(settings["review_period_days"]),
        "committed_stock_used": bool(settings.get("use_committed_stock", False)),
        "min_order_qty": settings["replenishment"].get("min_order_qty", 0),
        "order_multiple": settings["replenishment"].get("order_multiple", 1),
        "max_storage_units": settings["replenishment"].get("max_storage_units"),
        "inventory_data": data.metadata,
        "scenario_label": "model-based scenario analysis, not causal prediction",
    }

    scenario_summary = None
    scenario_details = None
    if not args.no_scenarios:
        print("Running inventory scenarios...")
        scenario_summary, scenario_details = run_scenarios(
            data,
            decision_date,
            forecast,
            config,
            service_level=args.service_level,
        )
        scenario_summary = scenario_deltas(scenario_summary)

    paths = write_recommendation_outputs(
        mode,
        decision_date,
        recommendations,
        output_dir / "recommendations",
        metadata=metadata,
        scenario_summary=scenario_summary,
        top_n=top_n,
    )
    print(f"Recommendations: {paths['recommendations']}")
    print(f"Report: {paths['report']}")

    if scenario_summary is not None:
        scenario_paths = write_scenario_outputs(
            scenario_summary,
            scenario_details,
            output_dir / "scenarios",
            metadata=metadata,
        )
        print(f"Scenario summary: {scenario_paths['summary_csv']}")
        print(f"Scenario report: {scenario_paths['report']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

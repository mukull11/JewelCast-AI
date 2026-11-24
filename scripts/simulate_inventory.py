"""Run the M4 historical inventory simulation and compare replenishment policies."""

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
from src.inventory.evaluation import add_relative_to_policy, policy_comparison
from src.inventory.forecasting_adapter import ModelForecastProvider
from src.inventory.reporting import write_simulation_outputs
from src.inventory.settings import inventory_settings
from src.inventory.simulation import SUPPORTED_POLICIES, simulate_policy
from src.utils.config import load_config
from src.utils.paths import resolve_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the historical inventory simulation (synthetic only).")
    parser.add_argument("--mode", default="synthetic", help="Only synthetic is supported.")
    parser.add_argument("--start", default=None, help="Simulation start date.")
    parser.add_argument("--end", default=None, help="Simulation end date.")
    parser.add_argument("--decision-frequency", type=int, default=None, help="Review period in days.")
    parser.add_argument("--horizon", type=int, default=None, help="Forecast horizon in days.")
    parser.add_argument("--forecast-mode", choices=["frozen", "rolling"], default=None, help="Causal model refit mode.")
    parser.add_argument("--refit-every-days", type=int, default=None, help="Rolling refit interval.")
    parser.add_argument("--service-level", type=float, default=None, help="Override target service level.")
    parser.add_argument("--policies", nargs="+", default=None, choices=list(SUPPORTED_POLICIES))
    parser.add_argument("--sku-limit", type=int, default=None, help="Limit to the top-N SKUs by units.")
    parser.add_argument("--processed-dir", default=None, help="Override the processed data directory.")
    parser.add_argument("--output", default=None, help="Override the report output directory.")
    parser.add_argument("--top-n", type=int, default=None, help="Rows shown in the Markdown report.")
    args = parser.parse_args()

    config = load_config()
    settings = inventory_settings(config)
    mode = str(args.mode or settings["mode"])
    if mode != "synthetic":
        print("Inventory simulation is supported only for the synthetic dataset.")
        return 2
    sim = settings["simulation"]
    start = pd.Timestamp(args.start or sim["start_date"]).normalize()
    end = pd.Timestamp(args.end or sim["end_date"]).normalize()
    policies = list(args.policies or sim.get("policies", ["forecast_driven", "simple_rop"]))
    frequency = int(args.decision_frequency or sim["decision_frequency_days"])
    horizon = int(args.horizon or sim["horizon"])
    forecast_mode = str(args.forecast_mode or sim.get("forecast_mode", "frozen"))
    refit_every = int(args.refit_every_days or sim.get("refit_every_days", 30))
    top_n = int(args.top_n or settings["reporting"]["top_n"])

    processed_dir = resolve_path(args.processed_dir or config["paths"]["data_processed"])
    output_dir = resolve_path(args.output or settings["reporting"]["output_dir"])

    print(f"Loading synthetic inventory data and forecasting panel ({mode})...")
    data = load_inventory_data(mode, config, processed_dir=processed_dir, sku_limit=args.sku_limit)
    panel = load_forecasting_panel(mode, config, processed_dir=processed_dir, sku_limit=args.sku_limit)

    provider = None
    if "forecast_driven" in policies:
        print(
            f"Preparing causal M3 forecast provider (mode={forecast_mode}, "
            f"refit_every={refit_every}d, horizon={horizon}d)..."
        )
        provider = ModelForecastProvider(
            panel,
            config,
            mode=forecast_mode,
            refit_every_days=refit_every,
            max_training_rows=int(sim.get("max_training_rows", 400000)),
        )

    results = {}
    for policy in policies:
        print(f"Simulating policy: {policy} ({start.date()} to {end.date()})...")
        results[policy] = simulate_policy(
            panel,
            data,
            config,
            policy=policy,
            forecast_provider=provider,
            start_date=start,
            end_date=end,
            decision_frequency_days=frequency,
            horizon=horizon,
            service_level=args.service_level,
        )

    comparison, per_sku, _ = policy_comparison(results)
    comparison = add_relative_to_policy(comparison, baseline_policy="forecast_driven")
    metadata = {
        "mode": mode,
        "start_date": str(start.date()),
        "end_date": str(end.date()),
        "decision_frequency_days": frequency,
        "forecast_horizon": horizon,
        "forecast_mode": forecast_mode,
        "refit_every_days": refit_every,
        "service_level": float(args.service_level if args.service_level is not None else settings["service_level"]),
        "forecast_fit_origins": [
            str(origin.date()) for origin in getattr(provider, "fit_origins", []) if origin is not None
        ],
        "simulation_settings": sim,
        "replenishment_settings": settings["replenishment"],
        "inventory_data": data.metadata,
    }
    paths = write_simulation_outputs(
        results,
        comparison,
        per_sku,
        output_dir / "simulation",
        metadata=metadata,
        top_n=top_n,
    )
    print("Policy comparison:")
    print(comparison.to_string(index=False))
    print(f"Comparison: {paths['comparison_csv']}")
    print(f"Simulation report: {paths['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

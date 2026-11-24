"""Evaluate persisted M4 inventory-simulation results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inventory.evaluation import add_relative_to_policy, simulation_metrics
from src.inventory.reporting import write_evaluation_outputs
from src.inventory.settings import inventory_settings
from src.utils.config import load_config
from src.utils.io import read_parquet
from src.utils.paths import resolve_path


def _load(path: Path, name: str) -> pd.DataFrame:
    file_path = path / name
    if not file_path.exists():
        raise FileNotFoundError(
            f"Simulation artefact not found: {file_path}. Run python scripts/simulate_inventory.py first."
        )
    return pd.read_parquet(file_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate persisted inventory simulation metrics.")
    parser.add_argument("--input", default=None, help="Simulation output directory.")
    parser.add_argument("--output", default=None, help="Evaluation output directory.")
    parser.add_argument("--baseline", default="forecast_driven", help="Baseline policy for relative differences.")
    args = parser.parse_args()

    config = load_config()
    settings = inventory_settings(config)
    base = resolve_path(settings["reporting"]["output_dir"])
    input_dir = resolve_path(args.input or (base / "simulation"))
    output_dir = resolve_path(args.output or (base / "evaluation"))

    daily = _load(input_dir, "simulation_daily.parquet")
    decisions = _load(input_dir, "simulation_decisions.parquet")
    orders = _load(input_dir, "simulation_orders.parquet")

    overall_rows = []
    per_sku_frames = []
    for policy, group in daily.groupby("policy", observed=True):
        overall, per_sku = simulation_metrics(
            group,
            decisions.loc[decisions["policy"] == policy],
            orders.loc[orders["policy"] == policy] if not orders.empty else orders,
        )
        overall["policy"] = policy
        overall["dataset"] = str(group["dataset"].iloc[0]) if "dataset" in group.columns else "synthetic"
        overall_rows.append(overall)
        per_sku.insert(0, "policy", policy)
        per_sku_frames.append(per_sku)

    comparison = pd.DataFrame(overall_rows)
    comparison = add_relative_to_policy(comparison, baseline_policy=args.baseline)
    per_sku_all = pd.concat(per_sku_frames, ignore_index=True) if per_sku_frames else pd.DataFrame()

    metadata = {
        "input_dir": str(input_dir),
        "baseline": args.baseline,
        "definitions": {
            "fill_rate": "fulfilled units / demand proxy units",
            "service_level_sku_day": "share of SKU-days with no unfulfilled demand proxy",
            "stockout_frequency": "stockout SKU-days / all SKU-days",
            "inventory_turnover": "demand proxy units / average on-hand inventory in the window",
            "days_of_inventory": "average inventory / average daily demand per SKU-day",
        },
    }
    paths = write_evaluation_outputs(comparison, per_sku_all, output_dir, metadata=metadata)
    print(comparison.to_string(index=False))
    print(f"Evaluation comparison: {paths['comparison_csv']}")
    print(f"Evaluation report: {paths['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

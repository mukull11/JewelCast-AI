"""Generate production forecasts with the trained JewelCast AI M3 engine."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.forecasting.data_builder import load_forecasting_panel
from src.forecasting.forecasting import production_forecast
from src.forecasting.models import GlobalDemandModel
from src.utils.config import load_config
from src.utils.paths import resolve_path


def _available_modes(processed_dir: Path) -> list[str]:
    return [
        mode
        for mode in ("synthetic", "uci", "m5")
        if (processed_dir / mode / "daily_sales.parquet").exists()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate M3 production forecasts.")
    parser.add_argument("--modes", nargs="+", default=None, help="Dataset modes to forecast.")
    parser.add_argument("--all", action="store_true", help="Forecast synthetic, UCI and M5.")
    parser.add_argument("--model-dir", default=None, help="Override the trained model directory.")
    parser.add_argument("--horizon", type=int, default=None, help="Forecast horizon in days. Default: from config.")
    parser.add_argument("--origin", default=None, help="Forecast origin date. Default: last observed date.")
    parser.add_argument("--sku-limit", type=int, default=None, help="Limit forecasting to the top-N SKUs by units.")
    parser.add_argument("--processed-dir", default=None, help="Override the processed data directory.")
    parser.add_argument("--output", default=None, help="Override the forecast output directory.")
    args = parser.parse_args()

    config = load_config()
    processed_dir = resolve_path(args.processed_dir or config["paths"]["data_processed"])
    available = _available_modes(processed_dir)
    if not available:
        print("No processed datasets found. Run: python scripts/build_datasets.py --all")
        return 1

    modes = available if args.all else list(args.modes or available)
    forecasts = {}
    for mode in modes:
        print(f"Generating forecasts: {mode}")
        model_dir = resolve_path(args.model_dir or (Path(config["paths"]["models"]) / "forecasting" / mode))
        model = GlobalDemandModel.load(model_dir, config)
        panel = load_forecasting_panel(mode, config, processed_dir=processed_dir, sku_limit=args.sku_limit)
        origin = args.origin or str(panel.frame["date"].max().date())
        horizon = args.horizon if args.horizon is not None else int((config.get("forecasting") or {}).get("horizons", [30])[0])
        frame = production_forecast(
            panel,
            model,
            origin=pd.Timestamp(origin).normalize(),
            horizon=horizon,
            feature_names=list(model.feature_names_),
            config=config,
            model_name=model.model_type,
        )
        forecasts[mode] = {
            "model": model.model_type,
            "origin": str(origin),
            "horizon": horizon,
            "rows": int(len(frame)),
        }
        print(f"Model: {model.model_type} Origin: {origin} Horizon: {horizon} Rows: {len(frame)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
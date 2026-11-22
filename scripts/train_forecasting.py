"""Train the JewelCast AI M3 forecasting engine for one or more dataset modes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.forecasting.forecasting import run_forecasting_pipeline
from src.forecasting.reporting import save_model_artifacts
from src.utils.config import load_config
from src.utils.paths import resolve_path


def _available_modes(processed_dir: Path) -> list[str]:
    return [
        mode
        for mode in ("synthetic", "uci", "m5")
        if (processed_dir / mode / "daily_sales.parquet").exists()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the JewelCast AI forecasting engine.")
    parser.add_argument("--modes", nargs="+", default=None, help="Dataset modes to train. Default: every processed mode.")
    parser.add_argument("--all", action="store_true", help="Train synthetic, UCI and M5.")
    parser.add_argument("--model-type", default=None, help="Force lightgbm | xgboost | sklearn_hist.")
    parser.add_argument("--backtest-origins", type=int, default=None, help="Rolling-origin backtests in validation.")
    parser.add_argument("--test-origins", type=int, default=None, help="Rolling-origin evaluations in the test period.")
    parser.add_argument("--sku-limit", type=int, default=None, help="Limit training to the top-N SKUs by units.")
    parser.add_argument("--processed-dir", default=None, help="Override the processed data directory.")
    parser.add_argument("--output", default=None, help="Override the model output directory.")
    args = parser.parse_args()

    config = load_config()
    processed_dir = resolve_path(args.processed_dir or config["paths"]["data_processed"])
    available = _available_modes(processed_dir)
    if not available:
        print("No processed datasets found. Run: python scripts/build_datasets.py --all")
        return 1

    modes = available if args.all else list(args.modes or available)
    results = {}
    for mode in modes:
        print(f"Training forecasting engine: {mode}")
        run = run_forecasting_pipeline(
            mode,
            config,
            processed_dir=processed_dir,
            sku_limit=args.sku_limit,
            model_type=args.model_type,
            backtest_origins=args.backtest_origins,
            test_origins=args.test_origins,
        )
        model_dir = resolve_path(args.output or (Path(config["paths"]["models"]) / "forecasting" / mode))
        paths = save_model_artifacts(run, config, model_dir)
        results[mode] = {
            "selected_model": run.selected_model_name,
            "model_dir": str(paths.get("selection", model_dir)),
            "validation_origins": [str(origin.date()) for origin in run.validation_origins],
            "test_origins": [str(origin.date()) for origin in run.test_origins],
            "runtime_seconds": run.runtime_seconds,
        }
        print(f"Selected model: {run.selected_model_name}")
        print(f"Model selection: {paths.get('selection')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
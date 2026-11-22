"""Run the JewelCast AI exploratory data analysis pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eda.pipeline import SUPPORTED_MODES, run_eda
from src.utils.config import load_config
from src.utils.io import write_json
from src.utils.paths import resolve_path


def _available_modes(processed_dir: Path) -> list[str]:
    return [
        mode
        for mode in SUPPORTED_MODES
        if (processed_dir / mode / "daily_sales.parquet").exists()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run JewelCast AI EDA for one or more dataset modes.")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=list(SUPPORTED_MODES),
        default=None,
        help="Dataset modes to analyse. Default: every available mode.",
    )
    parser.add_argument("--all", action="store_true", help="Analyse synthetic, UCI and M5.")
    parser.add_argument("--skus", nargs="+", default=None, help="Optional explicit SKU subset.")
    parser.add_argument("--sku-limit", type=int, default=None, help="Analyse only the top-N SKUs by units.")
    parser.add_argument("--top-n", type=int, default=20, help="Rows shown in top/bottom tables and charts.")
    parser.add_argument("--output", type=str, default=None, help="Base output directory (default: reports/eda).")
    parser.add_argument("--processed-dir", type=str, default=None, help="Override the processed data directory.")
    parser.add_argument("--no-png", action="store_true", help="Skip static PNG charts.")
    parser.add_argument("--no-html", action="store_true", help="Skip interactive HTML charts.")
    parser.add_argument(
        "--include-non-products",
        action="store_true",
        help="For UCI, keep postage/manual/bad-debt/fee codes that are excluded by default.",
    )
    args = parser.parse_args()

    config = load_config()
    processed_dir = resolve_path(args.processed_dir or config["paths"]["data_processed"])
    available = _available_modes(processed_dir)
    if not available:
        print("No processed datasets found. Run: python scripts/build_datasets.py --all")
        return 1

    if args.all:
        modes = available
    elif args.modes:
        modes = [mode for mode in args.modes if mode in available]
        missing = sorted(set(args.modes) - set(modes))
        if missing:
            print(f"Skipping unavailable modes: {', '.join(missing)}")
    else:
        modes = available

    base_output = resolve_path(args.output or (Path(config["paths"]["reports"]) / "eda"))
    results: dict[str, dict] = {}
    for mode in modes:
        print(f"\n=== EDA: {mode} ===")
        result = run_eda(
            mode,
            config=config,
            output_dir=base_output / mode,
            processed_dir=processed_dir,
            sku_subset=args.skus,
            sku_limit=args.sku_limit,
            top_n=args.top_n,
            make_png=not args.no_png,
            make_html=not args.no_html,
            exclude_non_products=not args.include_non_products,
        )
        results[mode] = {
            "output_dir": str(result.output_dir),
            "summary_path": str(result.summary_path),
            "report_path": str(result.report_path),
            "tables": {name: str(path) for name, path in sorted(result.tables.items())},
            "charts": {
                name: {kind: str(path) for kind, path in sorted(paths.items())}
                for name, paths in sorted(result.charts.items())
            },
            "overview": result.summary.get("overview"),
            "abc": result.summary.get("abc"),
            "intermittent": result.summary.get("intermittent"),
            "concentration": result.summary.get("concentration"),
        }
        print(json.dumps(result.summary.get("overview", {}), indent=2, default=str))
        print(f"Report: {result.report_path}")
        print(f"Tables: {len(result.tables)} | Charts: {sum(len(paths) for paths in result.charts.values())}")

    manifest_path = write_json(
        {
            "modes": modes,
            "skipped": sorted(set(SUPPORTED_MODES) - set(modes)),
            "output_dir": str(base_output),
            "results": results,
        },
        base_output / "manifest.json",
    )
    print(f"\nEDA manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

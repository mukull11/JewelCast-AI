"""Generate the synthetic Indian jewellery dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.synthetic import generate_synthetic_dataset
from src.utils.config import load_config
from src.utils.paths import resolve_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate reproducible synthetic jewellery data.")
    parser.add_argument("--n-skus", type=int, default=None)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    config = load_config()
    overrides = {}
    if args.n_skus is not None:
        overrides["n_skus"] = args.n_skus
    if args.start is not None:
        overrides["start_date"] = args.start
    if args.end is not None:
        overrides["end_date"] = args.end
    if args.seed is not None:
        overrides["seed"] = args.seed

    output = resolve_path(args.output or config["paths"]["data_synthetic"])
    result = generate_synthetic_dataset(config, output_dir=output, overrides=overrides or None)
    print(f"Synthetic products: {len(result.products):,}")
    print(f"Synthetic SKU-day rows: {len(result.daily_sales):,}")
    print(f"Inventory snapshots: {len(result.inventory_snapshots):,}")
    print(f"Purchase orders: {len(result.purchase_orders):,}")
    print(f"Written to: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

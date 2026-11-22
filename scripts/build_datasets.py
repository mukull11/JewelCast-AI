"""Build processed datasets for synthetic, UCI and/or M5 modes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.metadata import write_dataset_metadata
from src.preprocessing.pipeline import build_all
from src.utils.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Build JewelCast AI processed datasets.")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["synthetic", "uci", "m5"],
        default=None,
        help="Modes to build (default: all three).",
    )
    parser.add_argument("--all", action="store_true", help="Build synthetic, UCI and M5.")
    parser.add_argument("--no-metadata", action="store_true", help="Skip dataset_metadata.json generation.")
    args = parser.parse_args()

    config = load_config()
    modes = ["synthetic", "uci", "m5"] if args.all or not args.modes else args.modes
    summaries = build_all(config, modes=modes, write=True)

    print(json.dumps(summaries, indent=2, default=str))

    if not args.no_metadata:
        path = write_dataset_metadata(config)
        print(f"\nMetadata written to: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

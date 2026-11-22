"""Rebuild data/dataset_metadata.json from raw and processed artefacts."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.metadata import write_dataset_metadata
from src.utils.config import load_config


def main() -> int:
    config = load_config()
    path = write_dataset_metadata(config)
    print(f"Metadata written to: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

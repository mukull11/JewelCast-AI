"""Extract locally supplied dataset archives. Does not download data.

Place the official archives in the repository root:
- online+retail+ii.zip
- m5-forecasting-accuracy.zip

The archives are preserved; only their contents are extracted into data/raw/.
If archives are missing, manual download instructions are printed.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.utils.paths import resolve_path

INSTRUCTIONS = {
    "online+retail+ii.zip": (
        "UCI Online Retail II: https://archive.ics.uci.edu/dataset/502/online+retail+ii\n"
        "  Download the ZIP and place it at the repository root as online+retail+ii.zip."
    ),
    "m5-forecasting-accuracy.zip": (
        "M5 Forecasting Accuracy: https://www.kaggle.com/competitions/m5-forecasting-accuracy/data\n"
        "  Download the competition ZIP (Kaggle account required) and place it at the repository root\n"
        "  as m5-forecasting-accuracy.zip."
    ),
}


def extract_archive(archive: Path, destination: Path, force: bool = False) -> list[str]:
    destination.mkdir(parents=True, exist_ok=True)
    extracted: list[str] = []
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            if member.is_dir():
                continue
            target = destination / member.filename
            if target.exists() and not force:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(member) as source, target.open("wb") as sink:
                sink.write(source.read())
            extracted.append(member.filename)
    return extracted


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract JewelCast AI datasets (no downloads).")
    parser.add_argument("--force", action="store_true", help="Overwrite already extracted files.")
    args = parser.parse_args()

    config = load_config()
    raw_dir = resolve_path(config["paths"]["data_raw"])
    found = 0
    for name, instructions in INSTRUCTIONS.items():
        archive = ROOT / name
        if not archive.exists():
            print(f"[missing] {name}\n  {instructions}")
            continue
        found += 1
        extracted = extract_archive(archive, raw_dir, force=args.force)
        if extracted:
            print(f"[extracted] {name} -> {raw_dir} ({len(extracted)} files)")
        else:
            print(f"[present] {name} already extracted in {raw_dir}")

    if found == 0:
        print("\nNo archives found. The synthetic dataset can still be generated locally.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

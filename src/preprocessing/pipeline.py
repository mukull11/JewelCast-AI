"""Dataset build orchestration for synthetic, UCI and M5 modes."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.data.m5_loader import load_m5
from src.data.synthetic import generate_synthetic_dataset
from src.data.uci_loader import load_uci
from src.preprocessing.clean import clean_daily
from src.preprocessing.quality import build_quality_report, write_quality_report
from src.utils.io import save_parquet, write_json
from src.utils.paths import resolve_path

LIMITATIONS = {
    "synthetic": [
        "This dataset is fully simulated and must not be presented as real company data.",
        "Demand, promotions, festival effects and inventory are generated from documented assumptions.",
        "Inventory and purchase orders are simulated, not observed.",
    ],
    "uci": [
        "UCI Online Retail II contains UK online-retail transactions in GBP; it is not Indian retail data.",
        "The dataset is not jewellery-only. Jewellery categories come from keyword rules and are validated by SKU count.",
        "No inventory records exist, so true lost demand and stockout-censored demand cannot be identified.",
        "Dates are historical (2009-2011); no current market inference should be drawn.",
    ],
    "m5": [
        "M5 is a Walmart retail benchmark; it is not jewellery data and not Voylla data.",
        "Only a configurable item-store subset is processed to keep memory bounded.",
        "No store inventory records exist, so true lost demand cannot be identified.",
        "event_name reflects US M5 calendar events, not Indian festivals.",
    ],
}


def _write_provenance(
    output: Path,
    mode: str,
    audit: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> None:
    write_json(
        {
            "mode": mode,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "provenance": audit.get("provenance"),
            "limitations": LIMITATIONS.get(mode, []),
            "audit": dict(audit),
            "summary": dict(summary),
        },
        output / "provenance.json",
    )


def _finalize(
    mode: str,
    daily: pd.DataFrame,
    sku_master: pd.DataFrame,
    audit: Mapping[str, Any],
    config: Mapping[str, Any],
    require_inventory: bool,
    extra_audit: Mapping[str, Any] | None = None,
) -> dict:
    cleaned, clean_summary = clean_daily(daily, mode=mode, require_inventory=require_inventory)
    output = resolve_path(config["paths"]["data_processed"]) / mode
    output.mkdir(parents=True, exist_ok=True)

    save_parquet(cleaned, output / "daily_sales.parquet")
    save_parquet(sku_master, output / "sku_master.parquet")

    combined_audit = {**dict(audit), **(dict(extra_audit) if extra_audit else {})}
    report = build_quality_report(
        cleaned,
        mode=mode,
        audit=combined_audit,
        limitations=LIMITATIONS.get(mode, []),
    )
    write_quality_report(report, output)

    summary = {
        "mode": mode,
        "rows": int(len(cleaned)),
        "skus": int(cleaned["sku"].nunique()),
        "categories": int(cleaned["category"].nunique(dropna=True)),
        "date_min": cleaned["date"].min(),
        "date_max": cleaned["date"].max(),
        "units_sold": float(cleaned["units_sold"].sum()),
        "revenue": float(cleaned["revenue"].sum()),
        "inventory_available": bool(cleaned["available_stock"].notna().any()),
        "clean": clean_summary,
    }
    write_json(summary, output / "dataset_summary.json")
    _write_provenance(output, mode, combined_audit, summary)
    return summary


def build_synthetic(
    config: Mapping[str, Any],
    write: bool = True,
    overrides: Mapping[str, Any] | None = None,
) -> dict:
    synthetic_dir = resolve_path(config["paths"]["data_synthetic"])
    result = generate_synthetic_dataset(config, output_dir=synthetic_dir if write else None, overrides=overrides)

    audit = {
        "provenance": "Fully simulated Indian fashion-jewellery business dataset. Not real company data.",
        "seed": int(result.settings["seed"]),
        "n_skus": int(result.products["sku"].nunique()),
        "cold_start_skus": int(result.products["is_cold_start"].sum()),
        "inventory_snapshot_rows": int(len(result.inventory_snapshots)),
        "purchase_order_rows": int(len(result.purchase_orders)),
        "festival_days_configured": int(len(result.festivals)),
        "festival_flag_rows": int(result.daily_sales["festival_flag"].sum()),
        "stockout_flagged_rows": int(result.daily_sales["stockout_flag"].sum()),
    }

    sku_master = result.products.copy()
    summary = _finalize(
        "synthetic",
        result.daily_sales,
        sku_master,
        audit,
        config,
        require_inventory=True,
        extra_audit={"synthetic_settings": result.settings},
    )

    if write:
        output = resolve_path(config["paths"]["data_processed"]) / "synthetic"
        save_parquet(result.inventory_snapshots, output / "inventory_snapshots.parquet")
        save_parquet(result.purchase_orders, output / "purchase_orders.parquet")
    return summary


def build_uci(config: Mapping[str, Any], write: bool = True) -> dict:
    result = load_uci(config, write=write)
    summary = _finalize(
        "uci",
        result["daily_sales"],
        result["sku_master"],
        result["audit"],
        config,
        require_inventory=False,
    )
    return summary


def build_m5(config: Mapping[str, Any], write: bool = True, subset_overrides: Mapping[str, Any] | None = None) -> dict:
    result = load_m5(config, write=write, subset_overrides=subset_overrides)
    summary = _finalize(
        "m5",
        result["daily_sales"],
        result["sku_master"],
        result["audit"],
        config,
        require_inventory=False,
    )
    return summary


def build_mode(
    mode: str,
    config: Mapping[str, Any],
    write: bool = True,
    overrides: Mapping[str, Any] | None = None,
) -> dict:
    if mode == "synthetic":
        return build_synthetic(config, write=write, overrides=overrides)
    if mode == "uci":
        return build_uci(config, write=write)
    if mode == "m5":
        return build_m5(config, write=write, subset_overrides=overrides)
    raise ValueError(f"Unknown data mode: {mode}")


def build_all(
    config: Mapping[str, Any],
    modes: list[str] | None = None,
    write: bool = True,
) -> dict:
    modes = modes or ["synthetic", "uci", "m5"]
    summaries: dict[str, Any] = {}
    for mode in modes:
        summaries[mode] = build_mode(mode, config, write=write)
    if write:
        write_json(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "modes": summaries,
            },
            resolve_path(config["paths"]["data_processed"]) / "manifest.json",
        )
    return summaries

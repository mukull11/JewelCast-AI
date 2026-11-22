"""High-level EDA pipeline for one dataset mode."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.utils.config import load_config
from src.utils.paths import resolve_path

from .loading import (
    load_daily,
    load_inventory_snapshots,
    load_provenance,
    load_purchase_orders,
)
from .report import build_context, run_report

SUPPORTED_MODES = ("synthetic", "uci", "m5")


@dataclass
class EDAResult:
    mode: str
    output_dir: Path
    summary: dict[str, Any]
    summary_path: Path
    report_path: Path
    tables: dict[str, Path] = field(default_factory=dict)
    charts: dict[str, dict[str, Path]] = field(default_factory=dict)


def run_eda(
    mode: str,
    config: Mapping[str, Any] | None = None,
    output_dir: str | Path | None = None,
    processed_dir: str | Path | None = None,
    sku_subset: Iterable[str] | None = None,
    sku_limit: int | None = None,
    top_n: int = 20,
    make_png: bool = True,
    make_html: bool = True,
    exclude_non_products: bool = True,
) -> EDAResult:
    """Run the full EDA report for one mode and return generated artefact paths.

    ``sku_subset`` takes precedence over ``sku_limit``. A limit selects the
    highest-unit SKUs for the descriptive report only; it does not create or
    modify any forecasting feature. For UCI, explicit non-product codes are
    excluded unless ``exclude_non_products`` is false.
    """
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"Unsupported EDA mode: {mode}. Choose from {SUPPORTED_MODES}")
    config = config or load_config()
    if int(top_n) <= 0:
        raise ValueError("top_n must be a positive integer")

    output = resolve_path(output_dir or (Path(config["paths"]["reports"]) / "eda" / mode))
    daily, load_meta = load_daily(
        mode,
        config,
        processed_dir=processed_dir,
        sku_subset=sku_subset,
        sku_limit=sku_limit,
        exclude_non_products=exclude_non_products,
        return_meta=True,
    )
    provenance = load_provenance(mode, config, processed_dir=processed_dir)

    snapshots = load_inventory_snapshots(config, processed_dir=processed_dir) if mode == "synthetic" else None
    purchase_orders = load_purchase_orders(config, processed_dir=processed_dir) if mode == "synthetic" else None
    if sku_subset or sku_limit:
        selected = set(daily["sku"].astype(str).unique())
        if snapshots is not None and not snapshots.empty:
            snapshots = snapshots.loc[snapshots["sku"].astype(str).isin(selected)].copy()
        if purchase_orders is not None and not purchase_orders.empty:
            purchase_orders = purchase_orders.loc[purchase_orders["sku"].astype(str).isin(selected)].copy()

    context = build_context(
        daily,
        mode=mode,
        config=config,
        snapshots=snapshots,
        purchase_orders=purchase_orders,
    )
    context["load_meta"] = load_meta
    report = run_report(
        context,
        mode=mode,
        config=config,
        provenance=provenance,
        output_dir=output,
        top_n=int(top_n),
        make_png=make_png,
        make_html=make_html,
    )
    return EDAResult(
        mode=mode,
        output_dir=output,
        summary=report["summary"],
        summary_path=report["summary_path"],
        report_path=report["report_path"],
        tables=report["tables"],
        charts=report["charts"],
    )

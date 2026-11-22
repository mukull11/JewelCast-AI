"""Data-quality reporting for the unified daily dataset."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.utils.io import write_json
from src.utils.paths import resolve_path


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(result) or np.isinf(result):
        return None
    return result


def build_quality_report(
    frame: pd.DataFrame,
    mode: str,
    audit: Mapping[str, Any] | None = None,
    limitations: list[str] | None = None,
) -> dict:
    """Compute a structured, JSON-serialisable quality report."""
    rows = int(len(frame))
    missing: dict[str, dict[str, Any]] = {}
    for column in frame.columns:
        count = int(frame[column].isna().sum())
        missing[column] = {"count": count, "percentage": round(100.0 * count / max(rows, 1), 3)}

    per_sku_zero = frame.groupby("sku", observed=True)["units_sold"].apply(lambda s: float((s == 0).mean()))
    units = pd.to_numeric(frame["units_sold"], errors="coerce")
    revenue = pd.to_numeric(frame["revenue"], errors="coerce")
    q1, q3 = units.quantile([0.25, 0.75])
    iqr = q3 - q1
    outlier_threshold = q3 + 3 * iqr

    category_summary = (
        frame.groupby("category", observed=True)
        .agg(units_sold=("units_sold", "sum"), revenue=("revenue", "sum"), skus=("sku", "nunique"))
        .reset_index()
        .sort_values("revenue", ascending=False)
        .to_dict(orient="records")
    )

    report = {
        "mode": mode,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rows": rows,
        "skus": int(frame["sku"].nunique()),
        "categories": int(frame["category"].nunique(dropna=True)),
        "date_min": frame["date"].min(),
        "date_max": frame["date"].max(),
        "days": int(frame["date"].nunique()),
        "totals": {
            "units_sold": _safe_float(units.sum()),
            "revenue": _safe_float(revenue.sum()),
            "zero_sales_rows": int((units == 0).sum()),
            "zero_sales_share": round(float((units == 0).mean()), 4),
        },
        "missing_values": missing,
        "duplicate_sku_date_rows": int(frame.duplicated(subset=["sku", "date"]).sum()),
        "negative_units_rows": int((units < 0).sum()),
        "negative_revenue_rows": int((revenue < 0).sum()),
        "null_price_rows": int(pd.to_numeric(frame["selling_price"], errors="coerce").isna().sum()),
        "null_inventory_rows": int(pd.to_numeric(frame["available_stock"], errors="coerce").isna().sum()),
        "stockout_flagged_rows": int(pd.to_numeric(frame.get("stockout_flag"), errors="coerce").fillna(0).sum())
        if "stockout_flag" in frame.columns
        else 0,
        "festival_rows": int(pd.to_numeric(frame["festival_flag"], errors="coerce").fillna(0).sum()),
        "promotion_rows": int(pd.to_numeric(frame["promotion_flag"], errors="coerce").fillna(0).sum()),
        "date_imputed_rows": int(pd.to_numeric(frame.get("date_imputed"), errors="coerce").fillna(0).sum())
        if "date_imputed" in frame.columns
        else 0,
        "intermittent_demand": {
            "skus_with_any_zero_sales": int((per_sku_zero > 0).sum()),
            "skus_over_50pct_zero_sales": int((per_sku_zero > 0.5).sum()),
            "mean_zero_sales_share": round(float(per_sku_zero.mean()), 4) if len(per_sku_zero) else None,
        },
        "outliers": {
            "units_q1": _safe_float(q1),
            "units_q3": _safe_float(q3),
            "extreme_upper_bound": _safe_float(outlier_threshold),
            "rows_above_extreme_bound": int((units > outlier_threshold).sum()),
            "treatment": "Flagged only. Outliers are not deleted.",
        },
        "category_summary": category_summary,
        "source_audit": dict(audit or {}),
        "limitations": limitations or [],
    }
    return report


def _markdown_table(rows: list[list[Any]], headers: list[str]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def write_quality_report(report: Mapping[str, Any], output_dir: str | Path) -> dict[str, Path]:
    """Write JSON and Markdown versions of a quality report."""
    output = resolve_path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = write_json(report, output / "quality_report.json")

    missing_rows = [
        [column, values["count"], f"{values['percentage']}%"]
        for column, values in report["missing_values"].items()
        if values["count"] > 0
    ]
    category_rows = [
        [row["category"], row["skus"], row["units_sold"], round(float(row["revenue"]), 2)]
        for row in report["category_summary"]
    ]

    lines = [
        f"# Data Quality Report - {report['mode']}",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "## Coverage",
        "",
        f"- Rows: {report['rows']}",
        f"- SKUs: {report['skus']}",
        f"- Categories: {report['categories']}",
        f"- Dates: {report['date_min']} to {report['date_max']} ({report['days']} distinct days)",
        f"- Units sold: {report['totals']['units_sold']}",
        f"- Revenue: {report['totals']['revenue']}",
        f"- Zero-sales rows: {report['totals']['zero_sales_rows']} ({report['totals']['zero_sales_share']:.1%})",
        "",
        "## Integrity checks",
        "",
        f"- Duplicate SKU-date rows: {report['duplicate_sku_date_rows']}",
        f"- Negative units rows: {report['negative_units_rows']}",
        f"- Negative revenue rows: {report['negative_revenue_rows']}",
        f"- Null selling price rows: {report['null_price_rows']}",
        f"- Null inventory rows: {report['null_inventory_rows']}",
        f"- Stockout-flagged rows: {report['stockout_flagged_rows']}",
        f"- Date-imputed rows: {report['date_imputed_rows']}",
        f"- Festival rows: {report['festival_rows']}",
        f"- Promotion rows: {report['promotion_rows']}",
        "",
        "## Intermittent demand",
        "",
        f"- SKUs with any zero-sales day: {report['intermittent_demand']['skus_with_any_zero_sales']}",
        f"- SKUs with >50% zero-sales days: {report['intermittent_demand']['skus_over_50pct_zero_sales']}",
        f"- Mean zero-sales share: {report['intermittent_demand']['mean_zero_sales_share']}",
        "",
        "## Outliers",
        "",
        f"- Extreme upper bound (Q3 + 3*IQR): {report['outliers']['extreme_upper_bound']}",
        f"- Rows above bound: {report['outliers']['rows_above_extreme_bound']}",
        f"- Treatment: {report['outliers']['treatment']}",
        "",
        "## Missing values",
        "",
    ]
    if missing_rows:
        lines.append(_markdown_table(missing_rows, ["Column", "Nulls", "Share"]))
    else:
        lines.append("No missing values.")

    lines += ["", "## Category summary", ""]
    if category_rows:
        lines.append(_markdown_table(category_rows, ["Category", "SKUs", "Units", "Revenue"]))
    else:
        lines.append("No categories available.")

    if report["limitations"]:
        lines += ["", "## Limitations", ""]
        lines.extend(f"- {item}" for item in report["limitations"])

    if report.get("source_audit"):
        lines += ["", "## Source audit", "", "```json", str(report["source_audit"]), "```"]

    markdown_path = output / "quality_report.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path}

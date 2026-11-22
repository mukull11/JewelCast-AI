"""EDA orchestration, table export and Markdown report writing."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.utils.io import write_json

from . import metrics
from .plotting import generate_charts

MODE_LABELS = {
    "synthetic": "Synthetic Indian fashion jewellery (simulated, INR)",
    "uci": "UCI Online Retail II (UK benchmark, GBP)",
    "m5": "M5 Walmart benchmark (USD)",
}
CURRENCIES = {"synthetic": "INR", "uci": "GBP", "m5": "USD"}
PROVENANCE = {
    "synthetic": "Fully simulated Indian fashion-jewellery business dataset. Not real company data.",
    "uci": "UCI Online Retail II - UK online-retail transactions in GBP. Not Indian retail data and not jewellery-only.",
    "m5": "M5 Forecasting Accuracy (Walmart) - a separate benchmark, not jewellery data and not Voylla data.",
}
LIMITATIONS = {
    "synthetic": [
        "All demand, promotions, festival effects and inventory are simulated from documented assumptions.",
        "Festival dates and strengths are configured assumptions, not observed market events.",
        "Inventory and purchase-order findings describe the simulator, not a real business.",
    ],
    "uci": [
        "UK online-retail transactions in GBP; not Indian retail data.",
        "Not jewellery-only; categories are derived from keyword rules and include an Other group.",
        "No inventory records exist, so stockout and lost-demand metrics are not computed.",
        "Calendar analysis uses only the documented UK bank-holiday calendar.",
    ],
    "m5": [
        "Walmart M5 benchmark; not jewellery data and not Voylla data.",
        "Only the configured item-store subset is analysed.",
        "No inventory records exist, so stockout and lost-demand metrics are not computed.",
        "Calendar analysis uses M5 event_name features, not Indian festival labels.",
    ],
}


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    return frame.to_dict(orient="records")


def _write_csv(frame: pd.DataFrame, path: Path) -> Path | None:
    if frame is None or frame.empty:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8")
    return path


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not pd.notna(number):
        return "n/a"
    return f"{number:,.{digits}f}"


def _pct(value: Any, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not pd.notna(number):
        return "n/a"
    return f"{number * 100:,.{digits}f}%"


def _md_table(
    frame: pd.DataFrame,
    columns: list[str],
    headers: list[str] | None = None,
    max_rows: int | None = None,
    percent_columns: tuple[str, ...] = (),
    int_columns: tuple[str, ...] = (),
    decimal_columns: tuple[str, ...] = (),
) -> str:
    if frame is None or frame.empty:
        return "_No data available._"
    available = [column for column in columns if column in frame.columns]
    if not available:
        return "_No data available._"
    view = frame.loc[:, available]
    if max_rows is not None:
        view = view.head(max_rows)
    headers = headers or available
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, row in view.iterrows():
        values = []
        for column in available:
            value = row[column]
            if column in percent_columns:
                values.append(_pct(value))
            elif column in int_columns:
                values.append(_fmt(value, 0) if pd.notna(value) else "n/a")
            elif column in decimal_columns:
                values.append(_fmt(value, 3))
            elif isinstance(value, float):
                values.append(_fmt(value))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_context(
    daily: pd.DataFrame,
    mode: str,
    config: Mapping[str, Any],
    snapshots: pd.DataFrame | None = None,
    purchase_orders: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Compute every EDA summary and table for one dataset mode."""
    trends = metrics.demand_trends(daily)
    overview = metrics.overview_metrics(daily)
    coverage = metrics.sku_date_coverage(daily)
    sku_metrics = metrics.compute_sku_metrics(daily)
    abc = metrics.abc_summary(sku_metrics)
    intermittent = metrics.intermittent_summary(sku_metrics)
    concentration = metrics.demand_concentration(sku_metrics)
    categories = metrics.category_metrics(daily)
    category_monthly = metrics.category_monthly(daily)
    category_seasonality = metrics.category_seasonality(daily)
    weekday = metrics.weekday_profile(daily)
    month = metrics.month_profile(daily)
    festival = metrics.festival_analysis(daily, mode=mode, config=config)
    price = metrics.price_summary(daily)
    price_bins = metrics.price_demand_bins(daily)
    sku_price_corr = metrics.sku_price_demand_correlation(daily)
    category_price = metrics.category_price_sensitivity(daily, correlations=sku_price_corr)
    discount_corr = metrics.discount_demand_correlation(daily)
    promotion = metrics.promotion_analysis(daily)
    inventory = metrics.inventory_analysis(daily, snapshots=snapshots, purchase_orders=purchase_orders)

    return {
        "daily": daily,
        "trends": trends,
        "overview": overview,
        "coverage": coverage,
        "sku_metrics": sku_metrics,
        "abc_summary": abc,
        "intermittent_summary": intermittent,
        "concentration": concentration,
        "category_metrics": categories,
        "category_monthly": category_monthly,
        "category_seasonality": category_seasonality,
        "weekday": weekday,
        "month_profile": month,
        "festival": festival,
        "price": price,
        "price_demand_bins": price_bins,
        "sku_price_correlations": sku_price_corr,
        "category_price_sensitivity": category_price,
        "discount_demand_correlations": discount_corr,
        "promotion": promotion,
        "inventory": inventory,
    }


def write_tables(context: dict[str, Any], tables_dir: Path, top_n: int = 20) -> dict[str, Path]:
    """Write all numerical summary tables as CSV."""
    tables_dir.mkdir(parents=True, exist_ok=True)
    sku_metrics = context["sku_metrics"]
    top_columns = [
        "sku",
        "category",
        "units_sold",
        "revenue",
        "units_share",
        "revenue_share",
        "abc_class",
        "intermittent_class",
        "adi",
        "cv2_nonzero",
        "zero_sales_share",
        "mean_price",
    ]
    top_columns = [column for column in top_columns if column in sku_metrics.columns]
    outputs: dict[str, Path] = {}
    candidates: dict[str, pd.DataFrame] = {
        "demand_daily.csv": context["trends"]["daily"],
        "demand_weekly.csv": context["trends"]["weekly"],
        "demand_monthly.csv": context["trends"]["monthly"],
        "sku_metrics.csv": sku_metrics,
        "sku_date_coverage.csv": context["coverage"],
        "sku_top_units.csv": sku_metrics.nlargest(top_n, "units_sold").loc[:, top_columns] if not sku_metrics.empty else pd.DataFrame(),
        "sku_bottom_units.csv": sku_metrics.nsmallest(top_n, "units_sold").loc[:, top_columns] if not sku_metrics.empty else pd.DataFrame(),
        "sku_top_revenue.csv": sku_metrics.nlargest(top_n, "revenue").loc[:, top_columns] if not sku_metrics.empty else pd.DataFrame(),
        "sku_bottom_revenue.csv": sku_metrics.nsmallest(top_n, "revenue").loc[:, top_columns] if not sku_metrics.empty else pd.DataFrame(),
        "abc_summary.csv": context["abc_summary"],
        "intermittent_summary.csv": context["intermittent_summary"],
        "category_metrics.csv": context["category_metrics"],
        "category_monthly.csv": context["category_monthly"],
        "category_seasonality.csv": context["category_seasonality"],
        "weekday_profile.csv": context["weekday"]["by_weekday"],
        "weekday_vs_weekend.csv": context["weekday"]["weekday_vs_weekend"],
        "month_profile.csv": context["month_profile"],
        "price_summary_by_category.csv": context["price"]["by_category"],
        "price_demand_bins.csv": context["price_demand_bins"],
        "sku_price_correlations.csv": context["sku_price_correlations"],
        "category_price_sensitivity.csv": context["category_price_sensitivity"],
        "discount_demand_correlations.csv": context["discount_demand_correlations"],
    }
    festival = context["festival"]
    candidates["festival_segments.csv"] = festival.get("festival_segments", pd.DataFrame())
    candidates["festival_by_name.csv"] = festival.get("by_festival", pd.DataFrame())
    candidates["wedding_segments.csv"] = festival.get("wedding_segments", pd.DataFrame())
    candidates["wedding_by_name.csv"] = festival.get("by_wedding", pd.DataFrame())
    candidates["holiday_segments.csv"] = festival.get("holiday_segments", pd.DataFrame())
    candidates["holiday_by_name.csv"] = festival.get("by_holiday", pd.DataFrame())
    candidates["event_segments.csv"] = festival.get("event_segments", pd.DataFrame())
    candidates["event_by_name.csv"] = festival.get("by_event", pd.DataFrame())
    promotion = context["promotion"]
    candidates["promotion_segments.csv"] = promotion.get("segments", pd.DataFrame()) if promotion.get("available") else pd.DataFrame()
    candidates["promotion_by_category.csv"] = promotion.get("by_category", pd.DataFrame()) if promotion.get("available") else pd.DataFrame()
    inventory = context["inventory"]
    if inventory.get("available"):
        candidates["inventory_by_sku.csv"] = inventory.get("by_sku", pd.DataFrame())
        candidates["purchase_orders_by_sku.csv"] = inventory.get("purchase_orders_by_sku", pd.DataFrame())
    for name, frame in candidates.items():
        path = _write_csv(frame, tables_dir / name)
        if path is not None:
            outputs[name] = path
    return outputs


def build_summary_json(
    context: dict[str, Any],
    mode: str,
    config: Mapping[str, Any],
    provenance: Mapping[str, Any] | None,
    table_paths: Mapping[str, Path],
    chart_paths: Mapping[str, Mapping[str, Path]],
    top_n: int,
) -> dict[str, Any]:
    festival = context["festival"]
    weekday = context["weekday"]
    promotion = context["promotion"]
    inventory = context["inventory"]
    price = context["price"]
    summary = {
        "mode": mode,
        "label": MODE_LABELS.get(mode, mode),
        "currency": CURRENCIES.get(mode, ""),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provenance": (provenance or {}).get("provenance") or PROVENANCE.get(mode),
        "limitations": (provenance or {}).get("limitations") or LIMITATIONS.get(mode, []),
        "scope": {
            "rows": context["overview"]["rows"],
            "skus": context["overview"]["skus"],
            "categories": context["overview"]["categories"],
            "date_min": context["overview"]["date_min"],
            "date_max": context["overview"]["date_max"],
            "days": context["overview"]["days"],
            "top_n": int(top_n),
            "load_meta": context.get("load_meta", {}),
        },
        "overview": context["overview"],
        "concentration": context["concentration"],
        "abc": _records(context["abc_summary"]),
        "intermittent": _records(context["intermittent_summary"]),
        "weekday": {
            "weekend_uplift": weekday.get("weekend_uplift"),
            "by_weekday": _records(weekday.get("by_weekday")),
            "weekday_vs_weekend": _records(weekday.get("weekday_vs_weekend")),
        },
        "month_profile": _records(context["month_profile"]),
        "festival": {
            key: value
            for key, value in {
                "festival_vs_non": festival.get("festival_vs_non"),
                "wedding_vs_non": festival.get("wedding_vs_non"),
                "holiday_vs_non": festival.get("holiday_vs_non"),
                "event_vs_non": festival.get("event_vs_non"),
                "note": festival.get("note"),
                "by_festival": _records(festival.get("by_festival", pd.DataFrame())),
                "by_wedding": _records(festival.get("by_wedding", pd.DataFrame())),
                "by_holiday": _records(festival.get("by_holiday", pd.DataFrame())),
                "by_event": _records(festival.get("by_event", pd.DataFrame())),
            }.items()
            if value not in (None, [], {})
        },
        "pricing": {
            "price": price.get("price"),
            "discount": price.get("discount"),
            "price_null_share": price.get("price_null_share"),
            "discount_null_share": price.get("discount_null_share"),
            "discount_available": price.get("discount_available"),
            "positive_discount_rows": price.get("positive_discount_rows"),
            "zero_discount_rows": price.get("zero_discount_rows"),
            "price_demand_bins": _records(context["price_demand_bins"]),
            "category_price_sensitivity": _records(context["category_price_sensitivity"]),
            "causality_note": "Price and discount associations are exploratory correlations, not causal elasticity estimates.",
        },
        "promotion": {
            "available": bool(promotion.get("available")),
            "reason": promotion.get("reason"),
            "summary": promotion.get("summary"),
            "segments": _records(promotion.get("segments", pd.DataFrame())) if promotion.get("available") else [],
            "by_category": _records(promotion.get("by_category", pd.DataFrame())) if promotion.get("available") else [],
        },
        "inventory": {
            "available": bool(inventory.get("available")),
            "reason": inventory.get("reason"),
            "summary": inventory.get("summary"),
            "snapshots": inventory.get("snapshots"),
            "purchase_orders": inventory.get("purchase_orders"),
        },
        "outputs": {
            "tables": {name: str(path) for name, path in sorted(table_paths.items())},
            "charts": {
                name: {kind: str(path) for kind, path in sorted(paths.items())}
                for name, paths in sorted(chart_paths.items())
            },
        },
    }
    return summary


def write_markdown_report(
    context: dict[str, Any],
    summary: Mapping[str, Any],
    output_path: Path,
    top_n: int = 20,
) -> Path:
    """Write a human-readable EDA report for one dataset mode."""
    mode = summary["mode"]
    overview = context["overview"]
    coverage = overview["coverage"]
    concentration = context["concentration"]
    weekday = context["weekday"]
    festival = context["festival"]
    promotion = context["promotion"]
    inventory = context["inventory"]
    price = context["price"]
    currency = CURRENCIES.get(mode, "")
    lines: list[str] = []
    lines.append(f"# EDA Report - {MODE_LABELS.get(mode, mode)}")
    lines.append("")
    lines.append(f"Generated: {summary['generated_at']}")
    lines.append("")
    lines.append(f"**Provenance:** {summary['provenance']}")
    lines.append("")
    lines.append("This is a descriptive analysis of one dataset mode. Modes are never combined, and no forecasting features are created here.")
    lines.append("")

    lines.append("## 1. Overall demand")
    lines.append("")
    lines.append(f"- Rows: {overview['rows']:,}")
    lines.append(f"- SKUs: {overview['skus']:,} ({overview['active_skus']:,} active, {overview['inactive_skus']:,} inactive)")
    lines.append(f"- Categories: {overview['categories']:,}")
    lines.append(f"- Date range: {overview['date_min']} to {overview['date_max']} ({overview['days']:,} observed days)")
    lines.append(f"- Units sold: {_fmt(overview['units_sold'], 0)}")
    lines.append(f"- Revenue: {_fmt(overview['revenue'], 2)} {currency}")
    lines.append(f"- Zero-sales SKU-days: {overview['zero_sales_rows']:,} ({_pct(overview['zero_sales_share'])})")
    lines.append(f"- Mean units per SKU-day: {_fmt(overview['mean_units_per_sku_day'])}")
    lines.append(f"- Median units per SKU-day: {_fmt(overview['median_units_per_sku_day'])}")
    load_meta = context.get("load_meta", {})
    if load_meta.get("excluded_non_product_rows"):
        lines.append(
            f"- Excluded UCI non-product rows (postage/manual/bad debt/fees): "
            f"{load_meta['excluded_non_product_rows']:,} rows across "
            f"{load_meta.get('excluded_non_product_skus', 0):,} codes. They remain in the processed audit table."
        )
    lines.append("")
    lines.append("### Date coverage")
    lines.append("")
    lines.append(f"- Calendar span: {coverage['calendar_span_days']:,} days; observed dates: {coverage['observed_dates']:,}")
    lines.append(f"- Missing calendar days in span: {coverage['missing_calendar_days']:,}")
    lines.append(f"- Imputed rows (`date_imputed=1`): {coverage['imputed_rows']:,} ({_pct(coverage['imputed_share'])})")
    lines.append(f"- Expected SKU-days across each SKU's first-to-last span: {coverage['expected_sku_days']:,}")
    lines.append(f"- Missing SKU-days after date completion: {coverage['missing_sku_days']:,}")
    lines.append("")
    lines.append("### Monthly trend")
    lines.append("")
    monthly = context["trends"]["monthly"].copy()
    if not monthly.empty:
        monthly["period_start"] = monthly["period_start"].dt.strftime("%Y-%m")
        lines.append(_md_table(monthly, ["period_start", "units_sold", "revenue", "active_skus", "zero_sales_share"], ["Month", "Units", f"Revenue ({currency})", "Active SKUs", "Zero-sales share"], int_columns=("units_sold", "active_skus"), percent_columns=("zero_sales_share",)))
    lines.append("")

    lines.append("## 2. SKU-level analysis")
    lines.append("")
    lines.append("### ABC classification")
    lines.append("")
    lines.append(_md_table(context["abc_summary"], ["abc_class", "skus", "sku_share", "units_sold", "revenue", "revenue_share"], ["Class", "SKUs", "SKU share", "Units", "Revenue", "Revenue share"], int_columns=("skus", "units_sold"), percent_columns=("sku_share", "revenue_share")))
    lines.append("")
    lines.append("### Intermittent demand (Syntetos-Boylan)")
    lines.append("")
    lines.append(_md_table(context["intermittent_summary"], ["intermittent_class", "skus", "sku_share", "units_sold", "revenue", "mean_adi", "mean_cv2"], ["Class", "SKUs", "SKU share", "Units", "Revenue", "Mean ADI", "Mean CV²"], int_columns=("skus", "units_sold"), percent_columns=("sku_share",), decimal_columns=("mean_adi", "mean_cv2")))
    lines.append("")
    lines.append("### Demand concentration")
    lines.append("")
    lines.append(f"- Units Gini: {_fmt(concentration['units_gini'], 3)}; revenue Gini: {_fmt(concentration['revenue_gini'], 3)}")
    lines.append(f"- Units HHI: {_fmt(concentration['units_hhi'], 4)}; revenue HHI: {_fmt(concentration['revenue_hhi'], 4)}")
    lines.append(f"- Top-10 SKU share of units: {_pct(concentration['top10_units_share'])}; revenue: {_pct(concentration['top10_revenue_share'])}")
    lines.append(f"- Top-20 SKU share of units: {_pct(concentration['top20_units_share'])}; revenue: {_pct(concentration['top20_revenue_share'])}")
    lines.append("")
    top_units = context["sku_metrics"].nlargest(top_n, "units_sold")
    bottom_units = context["sku_metrics"].nsmallest(top_n, "units_sold")
    lines.append(f"### Top {top_n} SKUs by units")
    lines.append("")
    lines.append(_md_table(top_units, ["sku", "category", "units_sold", "revenue", "revenue_share", "abc_class", "intermittent_class", "adi", "cv2_nonzero"], ["SKU", "Category", "Units", "Revenue", "Revenue share", "ABC", "Demand class", "ADI", "CV²"], int_columns=("units_sold",), percent_columns=("revenue_share",), decimal_columns=("adi", "cv2_nonzero")))
    lines.append("")
    lines.append(f"### Bottom {top_n} SKUs by units")
    lines.append("")
    lines.append(_md_table(bottom_units, ["sku", "category", "units_sold", "revenue", "zero_sales_share", "abc_class", "intermittent_class"], ["SKU", "Category", "Units", "Revenue", "Zero-sales share", "ABC", "Demand class"], int_columns=("units_sold",), percent_columns=("zero_sales_share",)))
    lines.append("")

    lines.append("## 3. Category analysis")
    lines.append("")
    lines.append(_md_table(context["category_metrics"], ["category", "skus", "active_skus", "units_sold", "revenue", "units_share", "revenue_share", "avg_selling_price", "zero_sales_share"], ["Category", "SKUs", "Active SKUs", "Units", f"Revenue ({currency})", "Unit share", "Revenue share", "Avg price", "Zero-sales share"], int_columns=("skus", "active_skus", "units_sold"), percent_columns=("units_share", "revenue_share", "zero_sales_share")))
    lines.append("")
    lines.append("Category-level seasonality is written to `tables/category_seasonality.csv`; a monthly category heatmap is saved under `charts/`.")
    lines.append("")

    lines.append("## 4. Seasonal and calendar analysis")
    lines.append("")
    lines.append("### Weekday profile")
    lines.append("")
    lines.append(_md_table(weekday["by_weekday"], ["weekday", "days", "units_sold", "revenue", "avg_daily_units", "units_index"], ["Weekday", "Days", "Units", "Revenue", "Avg units/day", "Index"], int_columns=("days", "units_sold"), decimal_columns=("units_index",)))
    lines.append("")
    if weekday.get("weekend_uplift") is not None:
        lines.append(f"Weekend versus weekday uplift in average daily units: {_pct(weekday['weekend_uplift'])}.")
        lines.append("")
    lines.append("### Month-of-year profile")
    lines.append("")
    lines.append(_md_table(context["month_profile"], ["month_name", "days", "units_sold", "revenue", "avg_daily_units", "units_index"], ["Month", "Days", "Units", "Revenue", "Avg units/day", "Index"], int_columns=("days", "units_sold"), decimal_columns=("units_index",)))
    lines.append("")
    lines.append("### Calendar events")
    lines.append("")
    if mode == "synthetic":
        summary_row = festival.get("festival_vs_non")
        if summary_row:
            lines.append(f"- Festival versus non-festival average daily units uplift: {_pct(summary_row.get('uplift'))}.")
        wedding_row = festival.get("wedding_vs_non")
        if wedding_row:
            lines.append(f"- Wedding-season versus non-wedding average daily units uplift: {_pct(wedding_row.get('uplift'))}.")
        lines.append("")
        lines.append(_md_table(festival.get("by_festival", pd.DataFrame()), ["festival_name", "festival_type", "days", "units_sold", "revenue", "avg_daily_units", "uplift_vs_non_festival"], ["Festival", "Type", "Days", "Units", "Revenue", "Avg units/day", "Uplift vs non-festival"], int_columns=("days", "units_sold"), percent_columns=("uplift_vs_non_festival",)))
    elif mode == "uci":
        summary_row = festival.get("holiday_vs_non")
        if summary_row:
            lines.append(f"- UK bank-holiday versus non-holiday average daily units uplift: {_pct(summary_row.get('uplift'))}.")
            lines.append(f"- Active holiday days: {summary_row.get('active_days', 0)}; non-holiday days: {summary_row.get('inactive_days', 0)}.")
        lines.append("")
        lines.append(_md_table(festival.get("by_holiday", pd.DataFrame()), ["holiday_name", "days", "units_sold", "revenue"], ["Holiday", "Days", "Units", "Revenue"], int_columns=("days", "units_sold")))
        lines.append("")
        lines.append("Only the documented UK bank-holiday calendar is used. Indian festival labels are not applied to UCI.")
    elif mode == "m5":
        summary_row = festival.get("event_vs_non")
        if summary_row:
            lines.append(f"- M5 event versus non-event average daily units uplift: {_pct(summary_row.get('uplift'))}.")
            lines.append(f"- Event days: {summary_row.get('active_days', 0)}; non-event days: {summary_row.get('inactive_days', 0)}.")
        lines.append("")
        lines.append(_md_table(festival.get("by_event", pd.DataFrame()), ["event_name", "days", "units_sold", "revenue", "avg_daily_units"], ["Event", "Days", "Units", "Revenue", "Avg units/day"], int_columns=("days", "units_sold")))
        lines.append("")
        lines.append("M5 calendar analysis uses its own `event_name` feature. Indian festival labels are not applied to M5.")
    lines.append("")

    lines.append("## 5. Pricing and promotions")
    lines.append("")
    price_stats = price.get("price") or {}
    discount_stats = price.get("discount") or {}
    lines.append(f"- Price available on {price_stats.get('rows', 0):,} rows; mean {_fmt(price_stats.get('mean'))} {currency}, median {_fmt(price_stats.get('median'))}.")
    lines.append(f"- Price null share: {_pct(price.get('price_null_share'))}.")
    if price.get("discount_available"):
        lines.append(f"- Discount available on {discount_stats.get('rows', 0):,} rows; mean {_fmt(discount_stats.get('mean'))}%, median {_fmt(discount_stats.get('median'))}%.")
        lines.append(f"- Rows with a positive discount: {price.get('positive_discount_rows', 0):,}.")
    else:
        lines.append("- Discount data is not available in this dataset; discount metrics are not fabricated.")
    lines.append("- Price-demand and discount-demand correlations are exploratory associations, not causal effects.")
    lines.append("")
    lines.append(_md_table(context["price_demand_bins"], ["price_bin", "rows", "units_sold", "revenue", "avg_price", "avg_units_per_row", "zero_sales_share"], ["Price bin", "Rows", "Units", "Revenue", "Avg price", "Avg units/row", "Zero-sales share"], int_columns=("rows", "units_sold"), percent_columns=("zero_sales_share",)))
    lines.append("")
    if promotion.get("available"):
        promotion_summary = promotion.get("summary") or {}
        lines.append(
            f"- Promoted versus non-promoted SKU-day average unit uplift: {_pct(promotion_summary.get('uplift'))} "
            f"across {promotion_summary.get('skus_compared', 0):,} SKUs with both segments."
        )
        lines.append(f"- Median per-SKU promotion uplift: {_pct(promotion_summary.get('median_sku_uplift'))}.")
        lines.append("")
        lines.append(_md_table(promotion.get("by_category", pd.DataFrame()), ["category", "segment", "skus_compared", "avg_daily_units"], ["Category", "Segment", "SKUs compared", "Avg units/SKU-day"], int_columns=("skus_compared",)))
    else:
        lines.append(f"- Promotion analysis unavailable: {promotion.get('reason')}.")
    lines.append("")

    lines.append("## 6. Inventory")
    lines.append("")
    if inventory.get("available"):
        inv = inventory["summary"]
        lines.append(f"- Mean on-hand stock: {_fmt(inv['mean_stock'])}; median: {_fmt(inv['median_stock'])}.")
        lines.append(f"- Stockout rows: {inv['stockout_rows']:,} ({_pct(inv['stockout_row_share'])}); SKUs with any stockout: {inv['skus_with_stockout']:,} ({_pct(inv['sku_stockout_share'])}).")
        lines.append(f"- Mean inventory turnover: {_fmt(inv['mean_inventory_turnover'], 3)} per period; mean annualised turnover: {_fmt(inv['mean_annualised_turnover'], 3)}.")
        lines.append(f"- Mean days of inventory cover: {_fmt(inv['mean_days_of_cover'])}; median: {_fmt(inv['median_days_of_cover'])}.")
        if inventory.get("purchase_orders"):
            po = inventory["purchase_orders"]
            lines.append(f"- Purchase orders: {po['rows']:,}; total quantity {_fmt(po['total_quantity'], 0)}; mean lead time {_fmt(po['mean_lead_time_days'])} days.")
            lines.append(f"- Orders per SKU per year: {_fmt(po['orders_per_sku_per_year'])}.")
        lines.append("")
        lines.append(_md_table(inventory.get("by_sku", pd.DataFrame()), ["sku", "avg_stock", "end_stock", "stockout_days", "stockout_frequency", "inventory_turnover", "annualised_turnover", "days_of_cover"], ["SKU", "Avg stock", "End stock", "Stockout days", "Stockout freq.", "Turnover", "Annualised", "Days cover"], max_rows=top_n, int_columns=("stockout_days",), percent_columns=("stockout_frequency",), decimal_columns=("inventory_turnover", "annualised_turnover", "days_of_cover")))
    else:
        lines.append(f"Inventory analysis is not available: {inventory.get('reason')}")
    lines.append("")

    lines.append("## Methodology and caveats")
    lines.append("")
    lines.append("- Full formulas and metric definitions are documented in `docs/eda_methodology.md`.")
    lines.append("- This EDA is descriptive. It does not create lags, rolling features or future targets, so it introduces no leakage into later forecasting features.")
    lines.append("- Correlation is not causation. Price, discount, festival and promotion associations may be confounded by category mix, seasonality and product lifecycle.")
    for item in summary.get("limitations", []):
        lines.append(f"- {item}")
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def run_report(
    context: dict[str, Any],
    mode: str,
    config: Mapping[str, Any],
    provenance: Mapping[str, Any] | None,
    output_dir: Path,
    top_n: int = 20,
    make_png: bool = True,
    make_html: bool = True,
) -> dict[str, Any]:
    tables_dir = output_dir / "tables"
    charts_dir = output_dir / "charts"
    table_paths = write_tables(context, tables_dir, top_n=top_n)
    chart_paths = generate_charts(
        context,
        charts_dir,
        mode=mode,
        currency=CURRENCIES.get(mode, ""),
        top_n=top_n,
        make_png=make_png,
        make_html=make_html,
    )
    summary = build_summary_json(context, mode, config, provenance, table_paths, chart_paths, top_n)
    summary_path = write_json(summary, output_dir / "summary.json")
    report_path = write_markdown_report(context, summary, output_dir / "EDA_REPORT.md", top_n=top_n)
    return {
        "summary": summary,
        "summary_path": summary_path,
        "report_path": report_path,
        "tables": table_paths,
        "charts": chart_paths,
    }

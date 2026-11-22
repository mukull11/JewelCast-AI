"""Descriptive EDA metrics for the unified SKU-day dataset.

All functions are pure: they read a DataFrame and return small summary objects.
Formulas are documented in ``docs/eda_methodology.md``. The module never creates
forecasting features (no lags, no rolling windows, no future targets).
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

WEEKDAY_ORDER = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
MONTH_ORDER = tuple(range(1, 13))
MONTH_NAMES = (
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
)


def _as_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.normalize()


def _safe_float(value: Any, digits: int | None = None) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return round(result, digits) if digits is not None else result


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator is None or denominator == 0 or not np.isfinite(denominator):
        return None
    return float(numerator) / float(denominator)


def classify_intermittent(adi: float | None, cv2: float | None) -> str:
    """Classify demand using the Syntetos-Boylan (SBC) four-way scheme.

    ``ADI`` is the average inter-demand interval and ``cv2`` is the squared
    coefficient of variation of non-zero demand sizes. Thresholds are the
    documented SBC cut-offs: ADI 1.32 and CV² 0.49.
    """
    if adi is None or cv2 is None:
        return "no_demand"
    if not np.isfinite(adi) or not np.isfinite(cv2):
        return "no_demand"
    if adi < 1.32:
        return "smooth" if cv2 < 0.49 else "erratic"
    return "intermittent" if cv2 < 0.49 else "lumpy"


def abc_classify(
    frame: pd.DataFrame,
    revenue_col: str = "revenue",
    thresholds: tuple[float, float] = (0.8, 0.95),
) -> pd.DataFrame:
    """Add revenue-share, cumulative-share and ABC classes to a SKU table.

    The item that crosses the 80% boundary is included in class A, and the item
    that crosses 95% is included in class B. Ties are stable-sorted by revenue.
    A zero-revenue population is assigned to class C.
    """
    if revenue_col not in frame.columns:
        raise KeyError(f"Missing revenue column: {revenue_col}")
    if not 0 < thresholds[0] < thresholds[1] < 1:
        raise ValueError("ABC thresholds must satisfy 0 < A < B < 1")

    out = frame.copy()
    out = out.sort_values(revenue_col, ascending=False, kind="mergesort").reset_index(drop=True)
    total = float(pd.to_numeric(out[revenue_col], errors="coerce").fillna(0.0).sum())
    if total <= 0:
        out["revenue_share"] = 0.0
        out["cumulative_revenue_share"] = 0.0
        out["abc_class"] = "C"
        return out

    share = pd.to_numeric(out[revenue_col], errors="coerce").fillna(0.0).astype(float) / total
    out["revenue_share"] = share
    out["cumulative_revenue_share"] = share.cumsum()
    previous = out["cumulative_revenue_share"].shift(fill_value=0.0)
    lower, upper = thresholds
    out["abc_class"] = np.where(
        previous < lower,
        "A",
        np.where(previous < upper, "B", "C"),
    )
    return out


def abc_summary(sku_metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarise SKU counts, units and revenue for each ABC class."""
    if sku_metrics.empty or "abc_class" not in sku_metrics.columns:
        return pd.DataFrame(columns=["abc_class", "skus", "units_sold", "revenue", "sku_share", "revenue_share"])
    grouped = (
        sku_metrics.groupby("abc_class", observed=True)
        .agg(skus=("sku", "nunique"), units_sold=("units_sold", "sum"), revenue=("revenue", "sum"))
        .reindex(["A", "B", "C"])
        .dropna(how="all")
        .reset_index()
    )
    total_skus = float(grouped["skus"].sum()) or 1.0
    total_revenue = float(grouped["revenue"].sum()) or 1.0
    grouped["sku_share"] = grouped["skus"] / total_skus
    grouped["revenue_share"] = grouped["revenue"] / total_revenue
    return grouped


def intermittent_summary(sku_metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarise SKU counts and demand statistics by intermittent class."""
    classes = ["smooth", "erratic", "intermittent", "lumpy", "no_demand"]
    if sku_metrics.empty or "intermittent_class" not in sku_metrics.columns:
        return pd.DataFrame(columns=["intermittent_class", "skus", "units_sold", "revenue", "mean_adi", "median_adi", "mean_cv2"])
    grouped = (
        sku_metrics.groupby("intermittent_class", observed=True)
        .agg(
            skus=("sku", "nunique"),
            units_sold=("units_sold", "sum"),
            revenue=("revenue", "sum"),
            mean_adi=("adi", "mean"),
            median_adi=("adi", "median"),
            mean_cv2=("cv2_nonzero", "mean"),
        )
        .reindex(classes)
        .dropna(how="all")
        .reset_index()
    )
    grouped["sku_share"] = grouped["skus"] / max(float(grouped["skus"].sum()), 1.0)
    grouped["skus"] = grouped["skus"].astype(int)
    grouped["units_sold"] = grouped["units_sold"].astype(int)
    return grouped


def _period_frame(daily: pd.DataFrame, frequency: str) -> pd.DataFrame:
    working = daily.loc[:, ["date", "sku", "units_sold", "revenue", "date_imputed"]].copy()
    working["date"] = _as_date(working["date"])
    if frequency == "D":
        working["period_start"] = working["date"]
    else:
        working["period_start"] = working["date"].dt.to_period(frequency).dt.start_time
    working["_zero"] = (working["units_sold"] == 0).astype("int8")
    working["_imputed"] = pd.to_numeric(working["date_imputed"], errors="coerce").fillna(0).astype("int8")
    grouped = working.groupby("period_start", observed=True).agg(
        units_sold=("units_sold", "sum"),
        revenue=("revenue", "sum"),
        rows=("sku", "size"),
        zero_sales_rows=("_zero", "sum"),
        date_imputed_rows=("_imputed", "sum"),
    )
    active = (
        working.loc[working["units_sold"] > 0]
        .groupby("period_start", observed=True)["sku"]
        .nunique()
    )
    days = working.groupby("period_start", observed=True)["date"].nunique()
    grouped["active_skus"] = active.reindex(grouped.index, fill_value=0).astype(int)
    grouped["days"] = days.reindex(grouped.index, fill_value=0).astype(int)
    grouped["zero_sales_share"] = grouped["zero_sales_rows"] / grouped["rows"].replace(0, np.nan)
    return grouped.reset_index()


def demand_trends(daily: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return daily, weekly (Mon-Sun) and monthly sales aggregates.

    Weekly periods use ``W-SUN`` so ``period_start`` is a Monday. Revenue and
    units are summed; active SKUs are distinct SKUs with positive units in the
    period; days is the number of distinct observed dates.
    """
    return {
        "daily": _period_frame(daily, "D"),
        "weekly": _period_frame(daily, "W-SUN"),
        "monthly": _period_frame(daily, "M"),
    }


def missing_date_summary(daily: pd.DataFrame) -> dict[str, Any]:
    """Describe calendar coverage, imputed rows and missing SKU-days.

    ``missing_sku_days`` compares each SKU's first-to-last calendar span with the
    number of observed SKU-day rows. It should be zero after date completion; a
    non-zero value indicates a gap in the processed dataset.
    """
    dates = _as_date(daily["date"])
    if dates.empty:
        return {
            "calendar_span_days": 0,
            "observed_dates": 0,
            "missing_calendar_days": 0,
            "imputed_rows": 0,
            "imputed_share": 0.0,
            "expected_sku_days": 0,
            "observed_sku_days": 0,
            "missing_sku_days": 0,
        }
    span = int((dates.max() - dates.min()).days) + 1
    observed_dates = int(dates.nunique())
    imputed = int(pd.to_numeric(daily.get("date_imputed"), errors="coerce").fillna(0).sum()) if "date_imputed" in daily else 0
    bounds = (
        daily.assign(date=dates)
        .groupby("sku", observed=True)["date"]
        .agg(first_date="min", last_date="max", observed_days="size")
    )
    expected = int(((bounds["last_date"] - bounds["first_date"]).dt.days + 1).sum())
    observed = int(bounds["observed_days"].sum())
    return {
        "calendar_span_days": span,
        "observed_dates": observed_dates,
        "missing_calendar_days": int(span - observed_dates),
        "imputed_rows": imputed,
        "imputed_share": round(float(imputed / max(len(daily), 1)), 4),
        "expected_sku_days": expected,
        "observed_sku_days": observed,
        "missing_sku_days": int(expected - observed),
    }


def sku_date_coverage(daily: pd.DataFrame) -> pd.DataFrame:
    """Per-SKU first/last date, expected days, observed days and gaps."""
    working = daily.loc[:, ["sku", "date"]].copy()
    working["date"] = _as_date(working["date"])
    bounds = (
        working.groupby("sku", observed=True)["date"]
        .agg(first_date="min", last_date="max", observed_days="size")
        .reset_index()
    )
    bounds["expected_days"] = (bounds["last_date"] - bounds["first_date"]).dt.days + 1
    bounds["missing_days"] = (bounds["expected_days"] - bounds["observed_days"]).clip(lower=0)
    if "date_imputed" in daily.columns:
        imputed = (
            daily.assign(date_imputed=pd.to_numeric(daily["date_imputed"], errors="coerce").fillna(0))
            .groupby("sku", observed=True)["date_imputed"]
            .sum()
            .rename("imputed_days")
            .reset_index()
        )
        bounds = bounds.merge(imputed, on="sku", how="left")
        bounds["imputed_days"] = bounds["imputed_days"].fillna(0).astype(int)
    return bounds


def overview_metrics(daily: pd.DataFrame) -> dict[str, Any]:
    """Compute headline demand, activity and coverage metrics."""
    units = pd.to_numeric(daily["units_sold"], errors="coerce").fillna(0.0)
    revenue = pd.to_numeric(daily["revenue"], errors="coerce").fillna(0.0)
    sku_totals = daily.assign(units_sold=units).groupby("sku", observed=True)["units_sold"].sum()
    active_skus = int((sku_totals > 0).sum())
    inactive_skus = int((sku_totals <= 0).sum())
    zero_rows = int((units == 0).sum())
    coverage = missing_date_summary(daily)
    price = pd.to_numeric(daily.get("selling_price"), errors="coerce") if "selling_price" in daily else pd.Series(dtype=float)
    inventory = pd.to_numeric(daily.get("available_stock"), errors="coerce") if "available_stock" in daily else pd.Series(dtype=float)
    return {
        "rows": int(len(daily)),
        "skus": int(daily["sku"].nunique()),
        "active_skus": active_skus,
        "inactive_skus": inactive_skus,
        "active_sku_share": round(active_skus / max(len(sku_totals), 1), 4),
        "categories": int(daily["category"].nunique(dropna=True)),
        "date_min": str(_as_date(daily["date"]).min().date()) if len(daily) else None,
        "date_max": str(_as_date(daily["date"]).max().date()) if len(daily) else None,
        "days": int(_as_date(daily["date"]).nunique()),
        "units_sold": float(units.sum()),
        "revenue": float(revenue.sum()),
        "zero_sales_rows": zero_rows,
        "zero_sales_share": round(zero_rows / max(len(daily), 1), 4),
        "mean_units_per_sku_day": _safe_float(units.mean()),
        "median_units_per_sku_day": _safe_float(units.median()),
        "price_available": bool(price.notna().any()) if len(price) else False,
        "price_null_share": round(float(price.isna().mean()), 4) if len(price) else None,
        "inventory_available": bool(inventory.notna().any()) if len(inventory) else False,
        "coverage": coverage,
    }


def compute_sku_metrics(daily: pd.DataFrame) -> pd.DataFrame:
    """Build one row per SKU with demand, variability, ABC and SBC metrics.

    ADI = observed days / non-zero demand days.
    CV² (non-zero) = (std(non-zero units) / mean(non-zero units))².
    CV (all days) = std(all daily units) / mean(all daily units).
    """
    working = daily.loc[
        :,
        ["sku", "category", "date", "units_sold", "revenue", "selling_price", "promotion_flag"],
    ].copy()
    working["date"] = _as_date(working["date"])
    working = working.sort_values(["sku", "date"], kind="mergesort")
    working["_zero"] = (working["units_sold"] == 0).astype("int8")
    working["_nonzero"] = (working["units_sold"] > 0).astype("int8")

    grouped = working.groupby("sku", observed=True).agg(
        category=("category", "first"),
        first_date=("date", "min"),
        last_date=("date", "max"),
        days_observed=("date", "size"),
        units_sold=("units_sold", "sum"),
        revenue=("revenue", "sum"),
        zero_days=("_zero", "sum"),
        non_zero_days=("_nonzero", "sum"),
        mean_units=("units_sold", "mean"),
        std_units=("units_sold", "std"),
        mean_price=("selling_price", "mean"),
        min_price=("selling_price", "min"),
        max_price=("selling_price", "max"),
        promotion_days=("promotion_flag", "sum"),
    ).reset_index()

    nonzero = (
        working.loc[working["units_sold"] > 0]
        .groupby("sku", observed=True)["units_sold"]
        .agg(nonzero_mean="mean", nonzero_std="std")
        .reset_index()
    )
    metrics = grouped.merge(nonzero, on="sku", how="left")
    metrics["std_units"] = metrics["std_units"].fillna(0.0)
    metrics["cv_all"] = metrics["std_units"] / metrics["mean_units"].replace(0, np.nan)
    metrics["cv2_nonzero"] = ((metrics["nonzero_std"] / metrics["nonzero_mean"]) ** 2).fillna(0.0)
    metrics["adi"] = metrics["days_observed"] / metrics["non_zero_days"].replace(0, np.nan)
    metrics["zero_sales_share"] = metrics["zero_days"] / metrics["days_observed"].replace(0, np.nan)
    metrics["is_active"] = metrics["units_sold"] > 0
    metrics["intermittent_class"] = [
        classify_intermittent(adi, cv2)
        for adi, cv2 in zip(metrics["adi"], metrics["cv2_nonzero"])
    ]

    total_units = float(metrics["units_sold"].sum())
    total_revenue = float(metrics["revenue"].sum())
    metrics["units_share"] = metrics["units_sold"] / total_units if total_units else 0.0
    metrics["revenue_share"] = metrics["revenue"] / total_revenue if total_revenue else 0.0
    metrics = abc_classify(metrics, revenue_col="revenue")
    return metrics


def demand_concentration(sku_metrics: pd.DataFrame) -> dict[str, Any]:
    """Measure demand concentration with top shares, Gini and HHI.

    Gini is computed on non-negative SKU totals; HHI is the sum of squared unit
    or revenue shares (on the 0-1 scale). Top-N shares use the highest-N SKUs.
    """
    def _gini(values: np.ndarray) -> float:
        values = np.sort(np.asarray(values, dtype=float))
        if values.size == 0 or values.sum() <= 0:
            return 0.0
        cumulative = np.cumsum(values)
        n = values.size
        return float((n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n)

    def _hhi(values: np.ndarray) -> float:
        values = np.asarray(values, dtype=float)
        total = values.sum()
        if total <= 0:
            return 0.0
        shares = values / total
        return float(np.sum(shares**2))

    def _top_share(values: np.ndarray, n: int) -> float:
        values = np.sort(np.asarray(values, dtype=float))[::-1]
        total = values.sum()
        if total <= 0:
            return 0.0
        return float(values[: min(n, values.size)].sum() / total)

    units = sku_metrics["units_sold"].to_numpy(dtype=float)
    revenue = sku_metrics["revenue"].to_numpy(dtype=float)
    return {
        "skus": int(len(sku_metrics)),
        "units_gini": _gini(units),
        "revenue_gini": _gini(revenue),
        "units_hhi": _hhi(units),
        "revenue_hhi": _hhi(revenue),
        "top10_units_share": _top_share(units, 10),
        "top20_units_share": _top_share(units, 20),
        "top10_revenue_share": _top_share(revenue, 10),
        "top20_revenue_share": _top_share(revenue, 20),
    }


def category_metrics(daily: pd.DataFrame) -> pd.DataFrame:
    """Category-level units, revenue, SKU counts, price and promotion mix."""
    working = daily.loc[
        :, ["category", "sku", "units_sold", "revenue", "selling_price", "promotion_flag"]
    ].copy()
    working["_zero"] = (working["units_sold"] == 0).astype("int8")
    grouped = working.groupby("category", observed=True).agg(
        skus=("sku", "nunique"),
        rows=("units_sold", "size"),
        units_sold=("units_sold", "sum"),
        revenue=("revenue", "sum"),
        zero_sales_rows=("_zero", "sum"),
        avg_selling_price=("selling_price", "mean"),
        promotion_rows=("promotion_flag", "sum"),
    )
    active = (
        working.loc[working["units_sold"] > 0]
        .groupby("category", observed=True)["sku"]
        .nunique()
    )
    grouped["active_skus"] = active.reindex(grouped.index, fill_value=0).astype(int)
    grouped["zero_sales_share"] = grouped["zero_sales_rows"] / grouped["rows"].replace(0, np.nan)
    grouped["units_share"] = grouped["units_sold"] / max(float(grouped["units_sold"].sum()), 1.0)
    grouped["revenue_share"] = grouped["revenue"] / max(float(grouped["revenue"].sum()), 1.0)
    grouped["promotion_share"] = grouped["promotion_rows"] / grouped["rows"].replace(0, np.nan)
    return grouped.sort_values("revenue", ascending=False).reset_index()


def category_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    """Category x calendar-month units, revenue and average daily demand."""
    working = daily.loc[:, ["date", "category", "units_sold", "revenue"]].copy()
    working["date"] = _as_date(working["date"])
    working["month_start"] = working["date"].dt.to_period("M").dt.start_time
    grouped = working.groupby(["category", "month_start"], observed=True).agg(
        units_sold=("units_sold", "sum"),
        revenue=("revenue", "sum"),
        days=("date", "nunique"),
    ).reset_index()
    grouped["avg_daily_units"] = grouped["units_sold"] / grouped["days"].replace(0, np.nan)
    return grouped.sort_values(["category", "month_start"]).reset_index(drop=True)


def category_seasonality(daily: pd.DataFrame) -> pd.DataFrame:
    """Category x month-of-year seasonality index.

    ``avg_daily_units`` divides category-month units by the number of observed
    dates in that category-month. ``seasonality_index`` divides it by the
    category's overall average daily units, so 1.0 is the category baseline.
    """
    working = daily.loc[:, ["date", "category", "units_sold"]].copy()
    working["date"] = _as_date(working["date"])
    working["month"] = working["date"].dt.month
    monthly = working.groupby(["category", "month"], observed=True).agg(
        units_sold=("units_sold", "sum"),
        days=("date", "nunique"),
    ).reset_index()
    monthly["avg_daily_units"] = monthly["units_sold"] / monthly["days"].replace(0, np.nan)
    overall = working.groupby("category", observed=True).agg(
        total_units=("units_sold", "sum"),
        total_days=("date", "nunique"),
    )
    overall["category_avg_daily_units"] = overall["total_units"] / overall["total_days"].replace(0, np.nan)
    monthly = monthly.merge(
        overall[["category_avg_daily_units"]].reset_index(),
        on="category",
        how="left",
    )
    monthly["seasonality_index"] = monthly["avg_daily_units"] / monthly["category_avg_daily_units"].replace(0, np.nan)
    monthly["month_name"] = monthly["month"].map(dict(zip(MONTH_ORDER, MONTH_NAMES)))
    return monthly.sort_values(["category", "month"]).reset_index(drop=True)


def weekday_profile(daily: pd.DataFrame) -> dict[str, Any]:
    """Weekday demand profile and a weekday-versus-weekend comparison."""
    base = daily.loc[:, ["date", "sku", "units_sold", "revenue"]].copy()
    base["date"] = _as_date(base["date"])
    date_level = base.groupby("date", observed=True).agg(
        units_sold=("units_sold", "sum"),
        revenue=("revenue", "sum"),
        active_skus=("sku", "nunique"),
    ).reset_index()
    date_level["weekday"] = date_level["date"].dt.day_name()
    date_level["weekday_number"] = date_level["date"].dt.dayofweek
    grouped = date_level.groupby(["weekday_number", "weekday"], observed=True).agg(
        days=("date", "nunique"),
        units_sold=("units_sold", "sum"),
        revenue=("revenue", "sum"),
    ).reset_index()
    grouped["avg_daily_units"] = grouped["units_sold"] / grouped["days"].replace(0, np.nan)
    grouped["avg_daily_revenue"] = grouped["revenue"] / grouped["days"].replace(0, np.nan)
    overall_avg = grouped["units_sold"].sum() / max(grouped["days"].sum(), 1)
    grouped["units_index"] = grouped["avg_daily_units"] / overall_avg if overall_avg else np.nan
    grouped = grouped.sort_values("weekday_number").reset_index(drop=True)

    date_level["is_weekend"] = date_level["weekday_number"] >= 5
    split = date_level.groupby("is_weekend", observed=True).agg(
        days=("date", "nunique"),
        units_sold=("units_sold", "sum"),
        revenue=("revenue", "sum"),
    ).reset_index()
    split["avg_daily_units"] = split["units_sold"] / split["days"].replace(0, np.nan)
    split["avg_daily_revenue"] = split["revenue"] / split["days"].replace(0, np.nan)
    weekend = split.loc[split["is_weekend"]]
    weekday = split.loc[~split["is_weekend"]]
    weekend_uplift = None
    if not weekend.empty and not weekday.empty and weekday["avg_daily_units"].iloc[0]:
        weekend_uplift = float(
            weekend["avg_daily_units"].iloc[0] / weekday["avg_daily_units"].iloc[0] - 1.0
        )
    return {
        "by_weekday": grouped,
        "weekday_vs_weekend": split.drop(columns=["is_weekend"]).assign(
            segment=lambda frame: np.where(frame.index == 0, "weekday", "weekend")
        ),
        "weekend_uplift": weekend_uplift,
    }


def month_profile(daily: pd.DataFrame) -> pd.DataFrame:
    """Month-of-year demand with a seasonality index (1.0 = overall average)."""
    base = daily.loc[:, ["date", "units_sold", "revenue"]].copy()
    base["date"] = _as_date(base["date"])
    date_level = base.groupby("date", observed=True).agg(
        units_sold=("units_sold", "sum"),
        revenue=("revenue", "sum"),
    ).reset_index()
    date_level["month"] = date_level["date"].dt.month
    grouped = date_level.groupby("month", observed=True).agg(
        days=("date", "nunique"),
        units_sold=("units_sold", "sum"),
        revenue=("revenue", "sum"),
    ).reset_index()
    grouped["avg_daily_units"] = grouped["units_sold"] / grouped["days"].replace(0, np.nan)
    grouped["avg_daily_revenue"] = grouped["revenue"] / grouped["days"].replace(0, np.nan)
    overall_avg = grouped["units_sold"].sum() / max(grouped["days"].sum(), 1)
    grouped["units_index"] = grouped["avg_daily_units"] / overall_avg if overall_avg else np.nan
    grouped["month_name"] = grouped["month"].map(dict(zip(MONTH_ORDER, MONTH_NAMES)))
    return grouped.sort_values("month").reset_index(drop=True)


def _segment_comparison(
    date_level: pd.DataFrame,
    flag_column: str,
    label: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    working = date_level.copy()
    working[flag_column] = pd.to_numeric(working[flag_column], errors="coerce").fillna(0).astype(int)
    grouped = working.groupby(flag_column, observed=True).agg(
        days=("date", "nunique"),
        units_sold=("units_sold", "sum"),
        revenue=("revenue", "sum"),
    ).reset_index()
    grouped["avg_daily_units"] = grouped["units_sold"] / grouped["days"].replace(0, np.nan)
    grouped["avg_daily_revenue"] = grouped["revenue"] / grouped["days"].replace(0, np.nan)
    grouped["segment"] = grouped[flag_column].map({1: label, 0: f"non_{label}"})
    active = grouped.loc[grouped[flag_column] == 1]
    inactive = grouped.loc[grouped[flag_column] == 0]
    uplift = None
    if not active.empty and not inactive.empty and inactive["avg_daily_units"].iloc[0]:
        uplift = float(active["avg_daily_units"].iloc[0] / inactive["avg_daily_units"].iloc[0] - 1.0)
    summary = {
        "segment": label,
        "active_days": int(active["days"].iloc[0]) if not active.empty else 0,
        "inactive_days": int(inactive["days"].iloc[0]) if not inactive.empty else 0,
        "active_avg_daily_units": _safe_float(active["avg_daily_units"].iloc[0]) if not active.empty else None,
        "inactive_avg_daily_units": _safe_float(inactive["avg_daily_units"].iloc[0]) if not inactive.empty else None,
        "uplift": uplift,
    }
    return summary, grouped.drop(columns=[flag_column])


def festival_analysis(daily: pd.DataFrame, mode: str, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Mode-specific calendar-event analysis.

    Synthetic uses Indian festivals and configured wedding seasons. UCI uses only
    the documented UK bank-holiday flag. M5 uses its own ``event_name`` calendar
    feature. Indian labels are never applied to UCI or M5.
    """
    base = daily.loc[:, ["date", "sku", "units_sold", "revenue"]].copy()
    base["date"] = _as_date(base["date"])
    date_level = base.groupby("date", observed=True).agg(
        units_sold=("units_sold", "sum"),
        revenue=("revenue", "sum"),
        active_skus=("sku", "nunique"),
    ).reset_index()

    result: dict[str, Any] = {"mode": mode}

    if mode == "synthetic":
        flags = (
            daily.loc[:, ["date", "festival_flag", "festival_name"]]
            .assign(date=lambda frame: _as_date(frame["date"]))
            .groupby("date", observed=True)
            .agg(
                festival_flag=("festival_flag", "max"),
                festival_name=("festival_name", lambda values: values.dropna().iloc[0] if values.notna().any() else None),
            )
            .reset_index()
        )
        merged = date_level.merge(flags, on="date", how="left")
        merged["festival_flag"] = merged["festival_flag"].fillna(0).astype(int)
        summary, segments = _segment_comparison(merged, "festival_flag", "festival")
        result["festival_vs_non"] = summary
        result["festival_segments"] = segments
        by_name = (
            merged.loc[merged["festival_flag"] == 1]
            .groupby("festival_name", observed=True)
            .agg(
                days=("date", "nunique"),
                units_sold=("units_sold", "sum"),
                revenue=("revenue", "sum"),
            )
            .reset_index()
        )
        by_name["avg_daily_units"] = by_name["units_sold"] / by_name["days"].replace(0, np.nan)
        by_name["avg_daily_revenue"] = by_name["revenue"] / by_name["days"].replace(0, np.nan)
        non_festival_avg = summary.get("inactive_avg_daily_units")
        by_name["uplift_vs_non_festival"] = (
            by_name["avg_daily_units"] / non_festival_avg - 1.0 if non_festival_avg else np.nan
        )
        festival_types: dict[str, str] = {}
        if config:
            for item in (config.get("festivals", {}) or {}).get("indian", []) or []:
                festival_types[str(item.get("name"))] = str(item.get("type", "festival"))
        by_name["festival_type"] = by_name["festival_name"].map(festival_types)
        result["by_festival"] = by_name.sort_values("units_sold", ascending=False).reset_index(drop=True)

        wedding_rows: list[dict[str, Any]] = []
        if config:
            for item in (config.get("festivals", {}) or {}).get("wedding_seasons", []) or []:
                wedding_rows.append(
                    {
                        "name": item.get("name"),
                        "start": pd.Timestamp(item["start"]).normalize(),
                        "end": pd.Timestamp(item["end"]).normalize(),
                    }
                )
        if wedding_rows:
            wedding_flag = pd.Series(0, index=merged.index, dtype=int)
            wedding_name = pd.Series(None, index=merged.index, dtype=object)
            for item in wedding_rows:
                mask = (merged["date"] >= item["start"]) & (merged["date"] <= item["end"])
                wedding_flag.loc[mask] = 1
                wedding_name.loc[mask] = item["name"]
            merged["wedding_flag"] = wedding_flag
            merged["wedding_name"] = wedding_name
            wedding_summary, wedding_segments = _segment_comparison(merged, "wedding_flag", "wedding_season")
            result["wedding_vs_non"] = wedding_summary
            result["wedding_segments"] = wedding_segments
            by_wedding = (
                merged.loc[merged["wedding_flag"] == 1]
                .groupby("wedding_name", observed=True)
                .agg(
                    days=("date", "nunique"),
                    units_sold=("units_sold", "sum"),
                    revenue=("revenue", "sum"),
                )
                .reset_index()
            )
            by_wedding["avg_daily_units"] = by_wedding["units_sold"] / by_wedding["days"].replace(0, np.nan)
            by_wedding["avg_daily_revenue"] = by_wedding["revenue"] / by_wedding["days"].replace(0, np.nan)
            result["by_wedding"] = by_wedding
        else:
            result["wedding_vs_non"] = None
            result["wedding_segments"] = pd.DataFrame()
            result["by_wedding"] = pd.DataFrame()

    elif mode == "uci":
        flags = (
            daily.loc[:, ["date", "festival_flag", "festival_name"]]
            .assign(date=lambda frame: _as_date(frame["date"]))
            .groupby("date", observed=True)
            .agg(
                holiday_flag=("festival_flag", "max"),
                holiday_name=("festival_name", lambda values: values.dropna().iloc[0] if values.notna().any() else None),
            )
            .reset_index()
        )
        merged = date_level.merge(flags, on="date", how="left")
        merged["holiday_flag"] = merged["holiday_flag"].fillna(0).astype(int)
        summary, segments = _segment_comparison(merged, "holiday_flag", "uk_bank_holiday")
        result["holiday_vs_non"] = summary
        result["holiday_segments"] = segments
        result["by_holiday"] = (
            merged.loc[merged["holiday_flag"] == 1]
            .groupby("holiday_name", observed=True)
            .agg(days=("date", "nunique"), units_sold=("units_sold", "sum"), revenue=("revenue", "sum"))
            .reset_index()
        )
        result["note"] = "UCI uses only the documented UK bank-holiday calendar. Indian festivals are not applied."

    elif mode == "m5":
        events = (
            daily.loc[:, ["date", "event_name"]]
            .assign(date=lambda frame: _as_date(frame["date"]))
            .groupby("date", observed=True)
            .agg(event_name=("event_name", lambda values: values.dropna().iloc[0] if values.notna().any() else None))
            .reset_index()
        )
        merged = date_level.merge(events, on="date", how="left")
        merged["event_flag"] = merged["event_name"].notna().astype(int)
        summary, segments = _segment_comparison(merged, "event_flag", "m5_event")
        result["event_vs_non"] = summary
        result["event_segments"] = segments
        by_event = (
            merged.loc[merged["event_flag"] == 1]
            .groupby("event_name", observed=True)
            .agg(days=("date", "nunique"), units_sold=("units_sold", "sum"), revenue=("revenue", "sum"))
            .reset_index()
        )
        by_event["avg_daily_units"] = by_event["units_sold"] / by_event["days"].replace(0, np.nan)
        by_event["avg_daily_revenue"] = by_event["revenue"] / by_event["days"].replace(0, np.nan)
        result["by_event"] = by_event.sort_values("units_sold", ascending=False).reset_index(drop=True)
        result["note"] = "M5 uses its own US calendar event_name feature. Indian festivals are not applied."

    else:
        raise ValueError(f"Unsupported EDA mode: {mode}")

    result["date_level"] = date_level
    return result


def price_summary(daily: pd.DataFrame) -> dict[str, Any]:
    """Price and discount distribution summaries plus category-level detail."""
    price = pd.to_numeric(daily["selling_price"], errors="coerce")
    if "discount_percentage" in daily.columns:
        discount = pd.to_numeric(daily["discount_percentage"], errors="coerce")
    else:
        discount = pd.Series(np.nan, index=daily.index, dtype=float)
    valid_price = price.dropna()
    valid_discount = discount.dropna()

    def _stats(series: pd.Series) -> dict[str, Any]:
        if series.empty:
            return {
                "rows": 0,
                "mean": None,
                "std": None,
                "min": None,
                "p25": None,
                "median": None,
                "p75": None,
                "p95": None,
                "max": None,
            }
        return {
            "rows": int(series.size),
            "mean": _safe_float(series.mean()),
            "std": _safe_float(series.std()),
            "min": _safe_float(series.min()),
            "p25": _safe_float(series.quantile(0.25)),
            "median": _safe_float(series.median()),
            "p75": _safe_float(series.quantile(0.75)),
            "p95": _safe_float(series.quantile(0.95)),
            "max": _safe_float(series.max()),
        }

    by_category = (
        daily.assign(selling_price=price, discount_percentage=discount)
        .groupby("category", observed=True)
        .agg(
            rows=("selling_price", "size"),
            price_non_null=("selling_price", "count"),
            mean_price=("selling_price", "mean"),
            median_price=("selling_price", "median"),
            discount_non_null=("discount_percentage", "count"),
            mean_discount=("discount_percentage", "mean"),
        )
        .reset_index()
    )
    by_category["discount_share"] = by_category["discount_non_null"] / by_category["rows"].replace(0, np.nan)
    return {
        "price": _stats(valid_price),
        "discount": _stats(valid_discount),
        "price_null_share": round(float(price.isna().mean()), 4),
        "discount_null_share": round(float(discount.isna().mean()), 4),
        "discount_available": bool(discount.notna().any()),
        "positive_discount_rows": int((valid_discount > 0).sum()),
        "zero_discount_rows": int((valid_discount == 0).sum()),
        "by_category": by_category.sort_values("rows", ascending=False),
    }


def price_demand_bins(daily: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """Pooled price-quantile bins with units, revenue and zero-sales frequency."""
    working = daily.loc[:, ["selling_price", "units_sold", "revenue"]].copy()
    working["selling_price"] = pd.to_numeric(working["selling_price"], errors="coerce")
    working = working.loc[working["selling_price"].notna() & (working["selling_price"] > 0)]
    if working.empty:
        return pd.DataFrame(columns=["price_bin", "rows", "units_sold", "revenue", "avg_price", "avg_units_per_row", "zero_sales_share"])
    unique_prices = int(working["selling_price"].nunique())
    bins = max(1, min(int(n_bins), unique_prices))
    if bins == 1:
        working["price_bin"] = "all_prices"
    else:
        working["price_bin"] = pd.qcut(working["selling_price"], q=bins, duplicates="drop")
    working["_zero"] = (working["units_sold"] == 0).astype("int8")
    grouped = working.groupby("price_bin", observed=True).agg(
        rows=("selling_price", "size"),
        units_sold=("units_sold", "sum"),
        revenue=("revenue", "sum"),
        avg_price=("selling_price", "mean"),
        min_price=("selling_price", "min"),
        max_price=("selling_price", "max"),
        zero_sales_rows=("_zero", "sum"),
    ).reset_index()
    grouped["price_bin"] = grouped["price_bin"].astype(str)
    grouped["avg_units_per_row"] = grouped["units_sold"] / grouped["rows"].replace(0, np.nan)
    grouped["zero_sales_share"] = grouped["zero_sales_rows"] / grouped["rows"].replace(0, np.nan)
    grouped["revenue_share"] = grouped["revenue"] / max(float(grouped["revenue"].sum()), 1.0)
    return grouped


def sku_price_demand_correlation(daily: pd.DataFrame, min_obs: int = 30) -> pd.DataFrame:
    """Within-SKU Spearman correlation between selling price and units.

    Only SKUs with at least ``min_obs`` priced observations and at least two
    distinct prices are included. The result is an exploratory association, not
    a causal elasticity estimate.
    """
    working = daily.loc[:, ["sku", "category", "selling_price", "units_sold"]].copy()
    working["selling_price"] = pd.to_numeric(working["selling_price"], errors="coerce")
    working = working.loc[working["selling_price"].notna() & working["units_sold"].notna()]
    if working.empty:
        return pd.DataFrame(columns=["sku", "category", "n_obs", "price_demand_spearman"])
    stats = working.groupby("sku", observed=True)["selling_price"].agg(n_obs="count", n_prices="nunique")
    eligible = stats.index[(stats["n_obs"] >= int(min_obs)) & (stats["n_prices"] > 1)]
    subset = working.loc[working["sku"].isin(eligible)]
    rows: list[dict[str, Any]] = []
    for sku, group in subset.groupby("sku", observed=True):
        if len(group) < int(min_obs) or group["selling_price"].nunique() < 2:
            continue
        correlation = group["selling_price"].corr(group["units_sold"], method="spearman")
        if pd.isna(correlation):
            continue
        rows.append(
            {
                "sku": sku,
                "category": group["category"].iloc[0],
                "n_obs": int(len(group)),
                "price_demand_spearman": float(correlation),
            }
        )
    return pd.DataFrame(rows, columns=["sku", "category", "n_obs", "price_demand_spearman"])


def category_price_sensitivity(
    daily: pd.DataFrame,
    min_obs: int = 30,
    correlations: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Summarise within-SKU price-demand correlations by category."""
    if correlations is None:
        correlations = sku_price_demand_correlation(daily, min_obs=min_obs)
    if correlations.empty:
        return pd.DataFrame(columns=["category", "skus", "mean_price_demand_spearman", "median_price_demand_spearman", "negative_share"])
    grouped = correlations.groupby("category", observed=True).agg(
        skus=("sku", "nunique"),
        mean_price_demand_spearman=("price_demand_spearman", "mean"),
        median_price_demand_spearman=("price_demand_spearman", "median"),
        negative_share=("price_demand_spearman", lambda values: float((values < 0).mean())),
    ).reset_index()
    return grouped.sort_values("skus", ascending=False)


def discount_demand_correlation(daily: pd.DataFrame, min_obs: int = 30) -> pd.DataFrame:
    """Within-SKU Spearman correlation between discount percentage and units.

    Returns an empty frame when discount is unavailable (UCI/M5). Associations
    are exploratory and are not causal price-elasticity estimates.
    """
    if "discount_percentage" not in daily.columns:
        return pd.DataFrame(columns=["sku", "category", "n_obs", "discount_demand_spearman"])
    working = daily.loc[:, ["sku", "category", "discount_percentage", "units_sold"]].copy()
    working["discount_percentage"] = pd.to_numeric(working["discount_percentage"], errors="coerce")
    working = working.loc[working["discount_percentage"].notna() & working["units_sold"].notna()]
    if working.empty:
        return pd.DataFrame(columns=["sku", "category", "n_obs", "discount_demand_spearman"])
    stats = working.groupby("sku", observed=True)["discount_percentage"].agg(n_obs="count", n_values="nunique")
    eligible = stats.index[(stats["n_obs"] >= int(min_obs)) & (stats["n_values"] > 1)]
    subset = working.loc[working["sku"].isin(eligible)]
    rows: list[dict[str, Any]] = []
    for sku, group in subset.groupby("sku", observed=True):
        if len(group) < int(min_obs) or group["discount_percentage"].nunique() < 2:
            continue
        correlation = group["discount_percentage"].corr(group["units_sold"], method="spearman")
        if pd.isna(correlation):
            continue
        rows.append(
            {
                "sku": sku,
                "category": group["category"].iloc[0],
                "n_obs": int(len(group)),
                "discount_demand_spearman": float(correlation),
            }
        )
    return pd.DataFrame(rows, columns=["sku", "category", "n_obs", "discount_demand_spearman"])


def promotion_analysis(daily: pd.DataFrame) -> dict[str, Any]:
    """Compare demand on promoted versus non-promoted SKU-days.

    Promotions are SKU-specific in the synthetic dataset, so a date is not
    classified as promoted merely because one SKU is on promotion. Instead, each
    SKU's average units on promoted and non-promoted days are compared. Only SKUs
    with both segments are used for the paired uplift.
    """
    if "promotion_flag" not in daily.columns:
        return {"available": False, "reason": "promotion_flag is not available"}
    promo = pd.to_numeric(daily["promotion_flag"], errors="coerce").fillna(0).astype(int)
    if not bool((promo == 1).any()):
        return {"available": False, "reason": "No promotion days are recorded in this dataset"}

    base = daily.loc[:, ["date", "sku", "category", "units_sold", "revenue"]].copy()
    base["date"] = _as_date(base["date"])
    base["promotion_flag"] = promo
    sku_segment = (
        base.groupby(["sku", "category", "promotion_flag"], observed=True)
        .agg(
            days=("date", "nunique"),
            rows=("units_sold", "size"),
            units_sold=("units_sold", "sum"),
            revenue=("revenue", "sum"),
        )
        .reset_index()
    )
    sku_segment["avg_daily_units"] = sku_segment["units_sold"] / sku_segment["days"].replace(0, np.nan)
    paired = (
        sku_segment.pivot_table(
            index=["sku", "category"],
            columns="promotion_flag",
            values="avg_daily_units",
            aggfunc="first",
        )
        .rename(columns={0: "non_promotion_avg", 1: "promotion_avg"})
        .dropna(subset=["promotion_avg", "non_promotion_avg"])
        .reset_index()
    )
    paired["uplift"] = paired["promotion_avg"] / paired["non_promotion_avg"].replace(0, np.nan) - 1.0
    paired = paired.replace([np.inf, -np.inf], np.nan).dropna(subset=["uplift"])

    promotion_rows = int((promo == 1).sum())
    non_promotion_rows = int((promo == 0).sum())
    if paired.empty:
        return {
            "available": False,
            "reason": "No SKU has both promoted and non-promoted observations",
        }
    total_dates = int(base["date"].nunique())
    dates_with_any_promotion = int(base.loc[base["promotion_flag"] == 1, "date"].nunique())

    summary = {
        "segment": "promotion",
        "skus_compared": int(len(paired)),
        "promotion_rows": promotion_rows,
        "non_promotion_rows": non_promotion_rows,
        "promotion_row_share": round(promotion_rows / max(len(base), 1), 4),
        "dates_with_any_promotion": dates_with_any_promotion,
        "dates_without_any_promotion": total_dates - dates_with_any_promotion,
        "active_avg_daily_units": _safe_float(paired["promotion_avg"].mean()),
        "inactive_avg_daily_units": _safe_float(paired["non_promotion_avg"].mean()),
        "uplift": _safe_float(paired["promotion_avg"].mean() / paired["non_promotion_avg"].mean() - 1.0)
        if paired["non_promotion_avg"].mean()
        else None,
        "median_sku_uplift": _safe_float(paired["uplift"].median()),
    }
    segments = pd.DataFrame(
        {
            "segment": ["promotion", "non_promotion"],
            "skus": [int(len(paired)), int(len(paired))],
            "avg_daily_units": [
                summary["active_avg_daily_units"],
                summary["inactive_avg_daily_units"],
            ],
        }
    )
    by_category = (
        paired.groupby("category", observed=True)
        .agg(
            skus_compared=("sku", "nunique"),
            promotion_avg_daily_units=("promotion_avg", "mean"),
            non_promotion_avg_daily_units=("non_promotion_avg", "mean"),
            median_sku_uplift=("uplift", "median"),
        )
        .reset_index()
    )
    by_category["uplift"] = (
        by_category["promotion_avg_daily_units"]
        / by_category["non_promotion_avg_daily_units"].replace(0, np.nan)
        - 1.0
    )
    long_rows: list[dict[str, Any]] = []
    for _, row in by_category.iterrows():
        long_rows.append(
            {
                "category": row["category"],
                "segment": "promotion",
                "skus_compared": int(row["skus_compared"]),
                "avg_daily_units": row["promotion_avg_daily_units"],
            }
        )
        long_rows.append(
            {
                "category": row["category"],
                "segment": "non_promotion",
                "skus_compared": int(row["skus_compared"]),
                "avg_daily_units": row["non_promotion_avg_daily_units"],
            }
        )
    return {
        "available": True,
        "summary": summary,
        "segments": segments,
        "by_category": pd.DataFrame(long_rows).sort_values(["category", "segment"]).reset_index(drop=True),
    }


def inventory_analysis(
    daily: pd.DataFrame,
    snapshots: pd.DataFrame | None = None,
    purchase_orders: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Inventory metrics for datasets that actually carry inventory (synthetic).

    Inventory turnover = total units sold / mean on-hand stock over observed days.
    Annualised turnover = turnover x 365 / observed days. Days of cover =
    mean on-hand stock / mean daily units sold. UCI and M5 must not call this.
    """
    if "available_stock" not in daily.columns or not bool(pd.to_numeric(daily["available_stock"], errors="coerce").notna().any()):
        return {
            "available": False,
            "reason": "No inventory records exist in this dataset. Inventory metrics are not fabricated.",
        }

    working = daily.loc[:, ["sku", "date", "units_sold", "available_stock", "stockout_flag"]].copy()
    working["date"] = _as_date(working["date"])
    working["available_stock"] = pd.to_numeric(working["available_stock"], errors="coerce")
    working["stockout_flag"] = pd.to_numeric(working.get("stockout_flag"), errors="coerce").fillna(0)
    working = working.sort_values(["sku", "date"], kind="mergesort")
    per_sku = working.groupby("sku", observed=True).agg(
        days_observed=("date", "size"),
        total_units=("units_sold", "sum"),
        avg_stock=("available_stock", "mean"),
        min_stock=("available_stock", "min"),
        max_stock=("available_stock", "max"),
        end_stock=("available_stock", "last"),
        stockout_days=("stockout_flag", "sum"),
    ).reset_index()
    per_sku["avg_daily_units"] = per_sku["total_units"] / per_sku["days_observed"].replace(0, np.nan)
    per_sku["inventory_turnover"] = per_sku["total_units"] / per_sku["avg_stock"].replace(0, np.nan)
    per_sku["annualised_turnover"] = per_sku["inventory_turnover"] * 365.0 / per_sku["days_observed"].replace(0, np.nan)
    per_sku["days_of_cover"] = per_sku["avg_stock"] / per_sku["avg_daily_units"].replace(0, np.nan)
    per_sku["stockout_frequency"] = per_sku["stockout_days"] / per_sku["days_observed"].replace(0, np.nan)
    per_sku["is_active"] = per_sku["total_units"] > 0

    summary = {
        "available": True,
        "rows": int(len(working)),
        "skus": int(per_sku["sku"].nunique()),
        "mean_stock": _safe_float(working["available_stock"].mean()),
        "median_stock": _safe_float(working["available_stock"].median()),
        "min_stock": _safe_float(working["available_stock"].min()),
        "max_stock": _safe_float(working["available_stock"].max()),
        "zero_stock_rows": int((working["available_stock"] == 0).sum()),
        "zero_stock_share": round(float((working["available_stock"] == 0).mean()), 4),
        "stockout_rows": int(working["stockout_flag"].sum()),
        "stockout_row_share": round(float(working["stockout_flag"].mean()), 4),
        "skus_with_stockout": int((per_sku["stockout_days"] > 0).sum()),
        "sku_stockout_share": round(float((per_sku["stockout_days"] > 0).mean()), 4),
        "mean_inventory_turnover": _safe_float(per_sku["inventory_turnover"].mean()),
        "median_inventory_turnover": _safe_float(per_sku["inventory_turnover"].median()),
        "mean_annualised_turnover": _safe_float(per_sku["annualised_turnover"].mean()),
        "mean_days_of_cover": _safe_float(per_sku["days_of_cover"].mean()),
        "median_days_of_cover": _safe_float(per_sku["days_of_cover"].median()),
    }

    snapshot_summary: dict[str, Any] | None = None
    if snapshots is not None and not snapshots.empty:
        snap = snapshots.copy()
        for column in ("available_stock", "incoming_po_qty", "committed_qty"):
            if column in snap.columns:
                snap[column] = pd.to_numeric(snap[column], errors="coerce")
        snapshot_summary = {
            "rows": int(len(snap)),
            "skus": int(snap["sku"].nunique()) if "sku" in snap.columns else 0,
            "snapshot_dates": int(pd.to_datetime(snap["snapshot_date"]).nunique()) if "snapshot_date" in snap.columns else 0,
            "mean_available_stock": _safe_float(snap["available_stock"].mean()) if "available_stock" in snap else None,
            "mean_incoming_po_qty": _safe_float(snap["incoming_po_qty"].mean()) if "incoming_po_qty" in snap else None,
            "mean_committed_qty": _safe_float(snap["committed_qty"].mean()) if "committed_qty" in snap else None,
            "zero_stock_share": round(float((snap["available_stock"] == 0).mean()), 4) if "available_stock" in snap else None,
        }

    po_summary: dict[str, Any] | None = None
    po_by_sku = pd.DataFrame()
    if purchase_orders is not None and not purchase_orders.empty:
        po = purchase_orders.copy()
        for column in ("quantity", "supplier_lead_time_days"):
            if column in po.columns:
                po[column] = pd.to_numeric(po[column], errors="coerce")
        if "order_date" in po.columns:
            po["order_date"] = pd.to_datetime(po["order_date"])
        total_quantity = float(po["quantity"].sum()) if "quantity" in po else 0.0
        sku_count = int(po["sku"].nunique()) if "sku" in po else 0
        date_span = None
        if "order_date" in po.columns and po["order_date"].notna().any():
            date_span = max((po["order_date"].max() - po["order_date"].min()).days + 1, 1)
        po_summary = {
            "rows": int(len(po)),
            "skus": sku_count,
            "total_quantity": total_quantity,
            "mean_quantity": _safe_float(po["quantity"].mean()) if "quantity" in po else None,
            "median_quantity": _safe_float(po["quantity"].median()) if "quantity" in po else None,
            "mean_lead_time_days": _safe_float(po["supplier_lead_time_days"].mean()) if "supplier_lead_time_days" in po else None,
            "orders_per_sku": round(len(po) / sku_count, 4) if sku_count else None,
            "orders_per_sku_per_year": round((len(po) / sku_count) * 365.0 / date_span, 4) if sku_count and date_span else None,
            "status_counts": po["status"].value_counts().to_dict() if "status" in po else {},
            "date_min": str(po["order_date"].min().date()) if "order_date" in po.columns and po["order_date"].notna().any() else None,
            "date_max": str(po["order_date"].max().date()) if "order_date" in po.columns and po["order_date"].notna().any() else None,
        }
        if "sku" in po.columns:
            po_by_sku = (
                po.groupby("sku", observed=True)
                .agg(
                    orders=("quantity", "size"),
                    total_quantity=("quantity", "sum"),
                    mean_quantity=("quantity", "mean"),
                    mean_lead_time_days=("supplier_lead_time_days", "mean"),
                )
                .reset_index()
            )

    return {
        "available": True,
        "summary": summary,
        "by_sku": per_sku.sort_values("inventory_turnover", ascending=False).reset_index(drop=True),
        "snapshots": snapshot_summary,
        "purchase_orders": po_summary,
        "purchase_orders_by_sku": po_by_sku,
    }

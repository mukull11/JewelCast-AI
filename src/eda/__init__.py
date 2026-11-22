"""Reusable exploratory data analysis for JewelCast AI.

The package is deliberately descriptive: it computes summaries and charts from
the processed SKU-day datasets and never creates lagged, rolling or future
features. Each dataset mode is analysed independently and is never combined with
another mode.
"""

from .metrics import (
    abc_classify,
    abc_summary,
    category_metrics,
    category_monthly,
    category_seasonality,
    classify_intermittent,
    compute_sku_metrics,
    demand_concentration,
    demand_trends,
    discount_demand_correlation,
    festival_analysis,
    inventory_analysis,
    intermittent_summary,
    missing_date_summary,
    month_profile,
    overview_metrics,
    price_demand_bins,
    price_summary,
    promotion_analysis,
    sku_date_coverage,
    sku_price_demand_correlation,
    weekday_profile,
)
from .pipeline import EDAResult, run_eda

__all__ = [
    "EDAResult",
    "run_eda",
    "abc_classify",
    "abc_summary",
    "category_metrics",
    "category_monthly",
    "category_seasonality",
    "classify_intermittent",
    "compute_sku_metrics",
    "demand_concentration",
    "demand_trends",
    "discount_demand_correlation",
    "festival_analysis",
    "inventory_analysis",
    "intermittent_summary",
    "missing_date_summary",
    "month_profile",
    "overview_metrics",
    "price_demand_bins",
    "price_summary",
    "promotion_analysis",
    "sku_date_coverage",
    "sku_price_demand_correlation",
    "weekday_profile",
]

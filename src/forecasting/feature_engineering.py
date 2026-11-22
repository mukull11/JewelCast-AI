"""Leakage-safe feature engineering for the forecasting engine.

Every historical feature is derived from observations at or before the feature
date. Lag features are grouped by SKU/series. Rolling features are computed on
the lagged target, so the target day is never included in its own rolling
statistic. Calendar features are known in advance. Price, discount and promotion
features are lagged because future commercial decisions are not known at
forecast time.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

LAGS: tuple[int, ...] = (1, 7, 14, 28, 30)
ROLLING_WINDOWS: tuple[int, ...] = (7, 14, 28)
MAX_LAG = max(LAGS)
MAX_ROLLING = max(ROLLING_WINDOWS)
HISTORY_DAYS = max(MAX_LAG, MAX_ROLLING) + 1

IDENTIFIER_COLUMNS = {
    "date",
    "sku",
    "product_name",
    "category",
    "data_source",
    "festival_name",
    "event_name",
    "date_imputed",
    "stockout_flag",
    "available_stock",
    "selling_price",
    "discount_percentage",
    "promotion_flag",
    "units_sold",
    "revenue",
}

COMMON_FEATURES = {
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "lag_30",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_28",
    "rolling_std_7",
    "rolling_std_28",
    "rolling_zero_share_28",
    "price_lag1",
    "discount_lag1",
    "promotion_lag1",
    "rolling_promo_share_7",
    "category_price_median",
    "price_ratio_category",
    "series_age_days",
    "category_code",
    "day_of_week",
    "day_of_month",
    "month",
    "quarter",
    "week_of_year",
    "day_of_year",
    "is_weekend",
    "is_month_end",
    "days_to_month_end",
    "event_flag",
    "event_code",
}

MODE_FEATURES = {
    "synthetic": {"festival_flag", "festival_strength", "wedding_season_flag", "wedding_strength"},
    "uci": {"festival_flag"},
    "m5": set(),
}


def _grouped_shift(frame: pd.DataFrame, column: str, periods: int) -> pd.Series:
    return frame.groupby("sku", sort=False, observed=True)[column].shift(periods)


def _grouped_rolling(
    frame: pd.DataFrame,
    shifted: pd.Series,
    window: int,
    operation: str,
) -> pd.Series:
    grouped = shifted.groupby(frame["sku"], sort=False, observed=True)
    if operation == "mean":
        rolled = grouped.rolling(window, min_periods=1).mean()
    elif operation == "std":
        rolled = grouped.rolling(window, min_periods=2).std()
    else:
        raise ValueError(f"Unsupported rolling operation: {operation}")
    aligned = rolled.reset_index(level=0, drop=True)
    return aligned.reindex(frame.index).astype("float32")


def add_lag_features(frame: pd.DataFrame, lags: Sequence[int] = LAGS) -> pd.DataFrame:
    """Add grouped lag features for the demand target."""
    out = frame
    for lag in lags:
        out[f"lag_{int(lag)}"] = _grouped_shift(out, "units_sold", int(lag)).astype("float32")
    return out


def add_rolling_features(
    frame: pd.DataFrame,
    windows: Sequence[int] = ROLLING_WINDOWS,
) -> pd.DataFrame:
    """Add shifted rolling mean/std and zero-share features.

    The shift by one observation ensures that the target day is excluded from
    its own rolling statistic.
    """
    out = frame
    shifted_units = _grouped_shift(out, "units_sold", 1)
    for window in windows:
        out[f"rolling_mean_{int(window)}"] = _grouped_rolling(out, shifted_units, int(window), "mean")
    for window in (7, 28):
        if window in windows or window in (7, 28):
            out[f"rolling_std_{int(window)}"] = _grouped_rolling(out, shifted_units, int(window), "std")
    zero = (out["units_sold"].fillna(0) == 0).astype("float32")
    shifted_zero = zero.groupby(out["sku"], sort=False, observed=True).shift(1)
    out["rolling_zero_share_28"] = _grouped_rolling(out, shifted_zero, 28, "mean")
    return out


def add_price_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add lagged price/discount/promotion features available at forecast time."""
    out = frame
    price = out.groupby("sku", sort=False, observed=True)["selling_price"].ffill()
    discount = out.groupby("sku", sort=False, observed=True)["discount_percentage"].ffill()
    out["price_lag1"] = price.groupby(out["sku"], sort=False, observed=True).shift(1).astype("float32")
    out["discount_lag1"] = (
        discount.groupby(out["sku"], sort=False, observed=True).shift(1).fillna(0.0).astype("float32")
    )
    out["promotion_lag1"] = _grouped_shift(out, "promotion_flag", 1).fillna(0.0).astype("float32")
    promo = out["promotion_flag"].fillna(0).astype("float32")
    shifted_promo = promo.groupby(out["sku"], sort=False, observed=True).shift(1)
    out["rolling_promo_share_7"] = _grouped_rolling(out, shifted_promo, 7, "mean")
    return out


def add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add calendar features that are known in advance for every forecast date."""
    out = frame
    dates = pd.to_datetime(out["date"])
    out["day_of_week"] = dates.dt.dayofweek.astype("int8")
    out["day_of_month"] = dates.dt.day.astype("int8")
    out["month"] = dates.dt.month.astype("int8")
    out["quarter"] = dates.dt.quarter.astype("int8")
    out["week_of_year"] = dates.dt.isocalendar().week.astype("int16")
    out["day_of_year"] = dates.dt.dayofyear.astype("int16")
    out["is_weekend"] = (dates.dt.dayofweek >= 5).astype("int8")
    out["is_month_end"] = dates.dt.is_month_end.astype("int8")
    out["days_to_month_end"] = (dates.dt.days_in_month - dates.dt.day).astype("int8")
    for column, default in (
        ("festival_flag", 0),
        ("festival_strength", 1.0),
        ("wedding_season_flag", 0),
        ("wedding_strength", 1.0),
        ("event_flag", 0),
        ("event_code", -1),
    ):
        if column not in out.columns:
            out[column] = default
    out["festival_flag"] = pd.to_numeric(out["festival_flag"], errors="coerce").fillna(0).astype("int8")
    out["wedding_season_flag"] = pd.to_numeric(
        out["wedding_season_flag"], errors="coerce"
    ).fillna(0).astype("int8")
    out["festival_strength"] = pd.to_numeric(out["festival_strength"], errors="coerce").fillna(1.0).astype("float32")
    out["wedding_strength"] = pd.to_numeric(out["wedding_strength"], errors="coerce").fillna(1.0).astype("float32")
    out["event_flag"] = pd.to_numeric(out["event_flag"], errors="coerce").fillna(0).astype("int8")
    out["event_code"] = pd.to_numeric(out["event_code"], errors="coerce").fillna(-1).astype("int16")
    return out


def add_series_age(frame: pd.DataFrame, first_dates: Mapping[str, pd.Timestamp] | pd.Series | None = None) -> pd.DataFrame:
    out = frame
    if first_dates is None:
        first = out.groupby("sku", observed=True)["date"].min()
    else:
        first = pd.Series(first_dates)
    out["series_age_days"] = (out["date"] - out["sku"].map(first)).dt.days.astype("int32")
    return out


def add_category_price_feature(
    frame: pd.DataFrame,
    category_price_map: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    out = frame
    if category_price_map is None:
        category_price_map = (
            out.groupby("category", observed=True)["selling_price"].median().dropna().to_dict()
        )
    out["category_price_median"] = out["category"].map(category_price_map).astype("float32")
    denominator = out["category_price_median"].replace(0, np.nan)
    out["price_ratio_category"] = (out["price_lag1"] / denominator).astype("float32")
    return out


def build_features(
    frame: pd.DataFrame,
    mode: str,
    first_dates: Mapping[str, pd.Timestamp] | pd.Series | None = None,
    category_price_map: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Build the full leakage-safe feature frame for one mode."""
    if mode not in MODE_FEATURES:
        raise ValueError(f"Unsupported forecasting mode: {mode}")
    out = frame.copy()
    out = out.sort_values(["sku", "date"], kind="mergesort").reset_index(drop=True)
    if "category_code" not in out.columns:
        categories = sorted(str(value) for value in out["category"].dropna().unique())
        mapping = {value: index for index, value in enumerate(categories)}
        out["category_code"] = out["category"].map(mapping).fillna(-1).astype("int16")
    out = add_lag_features(out)
    out = add_rolling_features(out)
    out = add_price_features(out)
    out = add_calendar_features(out)
    out = add_series_age(out, first_dates=first_dates)
    out = add_category_price_feature(out, category_price_map=category_price_map)
    return out


def feature_columns(frame: pd.DataFrame, mode: str) -> list[str]:
    """Return the deterministic model feature list for a mode."""
    if mode not in MODE_FEATURES:
        raise ValueError(f"Unsupported forecasting mode: {mode}")
    allowed = COMMON_FEATURES | MODE_FEATURES[mode]
    numeric = [
        column
        for column in frame.columns
        if column not in IDENTIFIER_COLUMNS
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    selected = [column for column in numeric if column in allowed]
    return sorted(selected)

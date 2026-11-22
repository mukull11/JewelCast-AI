"""Unit tests for the M3 forecasting feature engineering layer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.forecasting.feature_engineering import (
    COMMON_FEATURES,
    LAGS,
    MODE_FEATURES,
    ROLLING_WINDOWS,
    build_features,
    feature_columns,
)


def _frame() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    series: dict[str, list[float]] = {
        "S1": [2.0, 0.0, 3.0, 1.0, 2.0, 0.0, 4.0, 2.0, 1.0, 0.0],
        "S2": [1.0, 1.0, 0.0, 2.0, 1.0, 1.0, 0.0, 2.0, 2.0, 1.0],
    }
    frames = []
    for sku, values in series.items():
        for day, value in zip(dates, values):
            frames.append(
                {
                    "date": day,
                    "sku": sku,
                    "product_name": f"Product {sku}",
                    "category": "Earrings" if sku == "S1" else "Necklaces",
                    "units_sold": float(value),
                    "selling_price": 300.0,
                    "discount_percentage": 0.0,
                    "revenue": float(value) * 300.0,
                    "festival_flag": 0,
                    "festival_name": None,
                    "promotion_flag": 0,
                    "available_stock": 10.0,
                    "stockout_flag": 0,
                    "event_name": None,
                    "event_flag": 0,
                    "event_code": -1,
                    "date_imputed": 0,
                    "data_source": "synthetic",
                }
            )
    return pd.DataFrame(frames)


def test_lag_features_use_previous_observations_only() -> None:
    features = build_features(_frame(), mode="synthetic")
    series = features.loc[features["sku"] == "S1"].sort_values("date").reset_index(drop=True)
    assert np.isnan(series["lag_1"].iloc[0])
    assert float(series["lag_1"].iloc[1]) == series["units_sold"].iloc[0]
    assert float(series["lag_7"].iloc[7]) == series["units_sold"].iloc[0]
    if len(series) > 13:
        assert float(series["lag_14"].iloc[13]) == series["units_sold"].iloc[0]
    if len(series) > 29:
        assert float(series["lag_30"].iloc[29]) == series["units_sold"].iloc[0]


def test_rolling_features_exclude_the_target_day() -> None:
    features = build_features(_frame(), mode="synthetic")
    series = features.loc[features["sku"] == "S1"].sort_values("date").reset_index(drop=True)
    values = series["units_sold"].to_numpy(dtype=float)
    last = len(values) - 1
    expected_mean = float(np.mean(values[last - 7 : last]))
    assert float(series["rolling_mean_7"].iloc[last]) == pytest.approx(expected_mean)
    expected_std = float(np.std(values[last - 7 : last], ddof=1))
    assert float(series["rolling_std_7"].iloc[last]) == pytest.approx(expected_std)
    window_values = [float(value) for value in values[last - 27 : last] if np.isfinite(value)]
    expected_zero_share = float(np.mean([1.0 if value == 0.0 else 0.0 for value in window_values]))
    assert float(series["rolling_zero_share_28"].iloc[last]) == pytest.approx(expected_zero_share)


def test_calendar_features_are_deterministic() -> None:
    features = build_features(_frame(), mode="synthetic")
    row = features.loc[(features["sku"] == "S1") & (features["date"] == pd.Timestamp("2024-01-07"))].iloc[0]
    assert int(row["day_of_week"]) == 6
    assert int(row["is_weekend"]) == 1
    assert int(row["day_of_month"]) == 7
    assert int(row["month"]) == 1
    assert int(row["quarter"]) == 1
    assert int(row["days_to_month_end"]) == 24


def test_price_features_are_lagged() -> None:
    features = build_features(_frame(), mode="synthetic")
    series = features.loc[features["sku"] == "S1"].sort_values("date").reset_index(drop=True)
    assert np.isnan(series["price_lag1"].iloc[0])
    assert float(series["price_lag1"].iloc[1]) == series["selling_price"].iloc[0]
    assert float(series["discount_lag1"].iloc[0]) == 0.0


def test_feature_columns_are_dataset_specific() -> None:
    synthetic_features = build_features(_frame(), mode="synthetic")
    synthetic_columns = feature_columns(synthetic_features, mode="synthetic")
    assert "lag_1" in synthetic_columns
    assert "festival_flag" in synthetic_columns
    assert "wedding_season_flag" in synthetic_columns
    uci_features = build_features(_frame(), mode="uci")
    uci_columns = feature_columns(uci_features, mode="uci")
    assert "lag_1" in uci_columns
    assert "festival_flag" in uci_columns
    assert "wedding_season_flag" not in uci_columns
    m5_features = build_features(_frame(), mode="m5")
    m5_columns = feature_columns(m5_features, mode="m5")
    assert "lag_1" in m5_columns
    assert "festival_flag" not in m5_columns
    assert "wedding_season_flag" not in m5_columns
    with pytest.raises(ValueError):
        feature_columns(_frame(), mode="unknown")


def test_common_feature_constants() -> None:
    assert LAGS == (1, 7, 14, 28, 30)
    assert ROLLING_WINDOWS == (7, 14, 28)
    assert "rolling_mean_7" in COMMON_FEATURES
    assert "rolling_std_28" in COMMON_FEATURES
    assert MODE_FEATURES["m5"] == set()
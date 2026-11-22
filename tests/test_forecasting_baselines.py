"""Unit tests for the M3 forecasting baselines."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.forecasting.baselines import BASELINE_MODELS, forecast_baselines


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


def test_baseline_registry_contains_minimum_benchmarks() -> None:
    assert set(BASELINE_MODELS) == {"naive", "seasonal_naive", "moving_average"}


def test_naive_baseline_repeats_last_observed_value() -> None:
    predictions = forecast_baselines(_frame(), origin=pd.Timestamp("2024-01-10"), horizon=3, models=("naive",))
    frame = predictions["naive"]
    by_sku = {sku: group for sku, group in frame.groupby("sku")}
    assert float(by_sku["S1"].iloc[0]["predicted_units"]) == 0.0
    assert float(by_sku["S2"].iloc[0]["predicted_units"]) == 1.0
    series_frame = frame.loc[frame["sku"] == "S1"].sort_values("horizon_day")
    assert list(series_frame["horizon_day"]) == [1, 2, 3]
    assert list(series_frame["predicted_units"]) == [0.0, 0.0, 0.0]


def test_seasonal_naive_baseline_uses_previous_week() -> None:
    predictions = forecast_baselines(_frame(), origin=pd.Timestamp("2024-01-10"), horizon=3, models=("seasonal_naive",))
    frame = predictions["seasonal_naive"]
    by_sku = {sku: group for sku, group in frame.groupby("sku")}
    # Seasonal value for day 1 comes from 7 days earlier: 2024-01-03.
    assert float(by_sku["S1"].iloc[0]["predicted_units"]) == 1.0
    assert float(by_sku["S2"].iloc[0]["predicted_units"]) == 2.0


def test_moving_average_baseline_uses_history_only() -> None:
    predictions = forecast_baselines(_frame(), origin=pd.Timestamp("2024-01-10"), horizon=1, models=("moving_average",))
    frame = predictions["moving_average"]
    by_sku = {sku: group for sku, group in frame.groupby("sku")}
    # Mean of the last 7 observed values before the origin.
    assert float(by_sku["S1"].iloc[0]["predicted_units"]) == pytest.approx(10.0 / 7.0)
    assert float(by_sku["S2"].iloc[0]["predicted_units"]) == pytest.approx(9.0 / 7.0)


def test_baselines_use_category_fallback_without_history() -> None:
    predictions = forecast_baselines(_frame(), origin=pd.Timestamp("2023-06-01"), horizon=2, models=("naive",))
    frame = predictions["naive"]
    assert float(frame["predicted_units"].mean()) >= 0.0
    assert frame["predicted_units"].notna().all()


def test_baselines_reject_unknown_model() -> None:
    with pytest.raises(ValueError):
        forecast_baselines(_frame(), origin=pd.Timestamp("2024-01-10"), horizon=1, models=("unknown",))


def test_baselines_reject_non_positive_horizon() -> None:
    with pytest.raises(ValueError):
        forecast_baselines(_frame(), origin=pd.Timestamp("2024-01-10"), horizon=0, models=("naive",))
"""Unit tests for the M3 forecasting, multi-horizon and output artefacts."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.forecasting.data_builder import ForecastingPanel
from src.forecasting.forecasting import forecast_ml, production_forecast
from src.forecasting.feature_engineering import build_features, feature_columns
from src.forecasting.reporting import FORECAST_COLUMNS, write_forecast_files


def _frame() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    series: dict[str, list[float]] = {
        "S1": [2.0, 0.0, 3.0, 1.0, 2.0, 0.0, 4.0, 2.0, 1.0, 0.0, 2.0, 1.0, 0.0, 3.0, 1.0, 2.0, 0.0, 1.0, 2.0, 1.0],
        "S2": [1.0, 1.0, 0.0, 2.0, 1.0, 1.0, 0.0, 2.0, 2.0, 1.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        "S3": [2.0, 1.0, 0.0],
    }
    frames = []
    for sku, values in series.items():
        for day, value in zip(dates, values):
            frames.append(
                {
                    "date": day,
                    "sku": sku,
                    "product_name": f"Product {sku}",
                    "category": "Earrings" if sku in {"S1", "S3"} else "Necklaces",
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


def _panel() -> ForecastingPanel:
    frame = _frame()
    calendar = pd.DataFrame(
        {
            "date": sorted(frame["date"].unique()),
            "festival_flag": 0,
            "festival_name": None,
            "festival_strength": 1.0,
            "wedding_season_flag": 0,
            "wedding_strength": 1.0,
            "event_name": None,
            "event_flag": 0,
            "event_code": -1,
        }
    )
    return ForecastingPanel(
        mode="synthetic",
        frame=frame,
        calendar=calendar,
        category_codes={"Earrings": 0, "Necklaces": 1},
        first_dates=frame.groupby("sku")["date"].min(),
        event_codes={},
        metadata={"max_horizon": 30, "mode": "synthetic"},
    )


class _LagStubModel:
    model_type = "lag_stub"
    feature_names_: list[str] = []

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        values = []
        for _, row in X.iterrows():
            lag = row["lag_1"]
            values.append(float(lag) if pd.notna(lag) else 0.0)
        return np.asarray(values, dtype=float)


def _config() -> dict:
    return {
        "forecasting": {
            "max_horizon": 30,
            "min_history_for_model": 5,
            "target_transform": "log1p",
            "clip_predictions_at_zero": True,
        }
    }


def test_forecast_ml_generates_recursive_multi_horizon_forecast() -> None:
    model = _LagStubModel()
    panel = _panel()
    features = build_features(panel.frame, "synthetic")
    feature_names = feature_columns(features, "synthetic")
    forecast = forecast_ml(
        model,
        panel,
        origin=pd.Timestamp("2024-01-20"),
        horizon=3,
        feature_names=feature_names,
        config=_config(),
    )
    assert forecast["model"].eq("lag_stub").all()
    assert list(forecast["horizon_day"]) == [1, 2, 3, 1, 2, 3, 1, 2, 3]
    by_sku = {sku: group for sku, group in forecast.groupby("sku")}
    s1 = by_sku["S1"].sort_values("horizon_day")
    assert float(s1.iloc[0]["predicted_units"]) == float(s1.iloc[1]["predicted_units"])
    assert float(s1.iloc[1]["predicted_units"]) == float(s1.iloc[2]["predicted_units"])
    assert float(s1.iloc[2]["predicted_units"]) >= 0.0
    s3 = by_sku["S3"].sort_values("horizon_day")
    assert float(s3.iloc[0]["predicted_units"]) >= 0.0
    assert float(s3.iloc[0]["predicted_units"]) == float(s3.iloc[1]["predicted_units"])


def test_forecast_ml_supports_configured_horizons() -> None:
    model = _LagStubModel()
    panel = _panel()
    features = build_features(panel.frame, "synthetic")
    feature_names = feature_columns(features, "synthetic")
    forecast = forecast_ml(
        model,
        panel,
        origin=pd.Timestamp("2024-01-20"),
        horizon=30,
        feature_names=feature_names,
        config=_config(),
    )
    assert forecast["date"].min() == pd.Timestamp("2024-01-21")
    assert forecast["date"].max() == pd.Timestamp("2024-02-19")
    assert forecast["horizon_day"].max() == 30
    assert forecast["predicted_units"].notna().all()


def test_forecast_ml_short_history_uses_fallback() -> None:
    model = _LagStubModel()
    panel = _panel()
    features = build_features(panel.frame, "synthetic")
    feature_names = feature_columns(features, "synthetic")
    forecast = forecast_ml(
        model,
        panel,
        origin=pd.Timestamp("2024-01-20"),
        horizon=2,
        feature_names=feature_names,
        config=_config(),
        min_history_for_model=100,
    )
    assert forecast["predicted_units"].notna().all()
    assert float(forecast["predicted_units"].max()) >= 0.0


def test_production_forecast_uses_origin_only() -> None:
    model = _LagStubModel()
    panel = _panel()
    features = build_features(panel.frame, "synthetic")
    feature_names = feature_columns(features, "synthetic")
    forecast = production_forecast(
        panel,
        model,
        origin=pd.Timestamp("2024-01-20"),
        horizon=7,
        feature_names=feature_names,
        config=_config(),
    )
    assert forecast["origin_date"].eq(pd.Timestamp("2024-01-20")).all()
    assert forecast["date"].min() == pd.Timestamp("2024-01-21")
    assert forecast["date"].max() == pd.Timestamp("2024-01-27")


def test_write_forecast_files_schema_and_names(tmp_path: Any) -> None:
    forecast = production_forecast(
        _panel(),
        _LagStubModel(),
        origin=pd.Timestamp("2024-01-20"),
        horizon=7,
        feature_names=["lag_1"],
        config=_config(),
        model_name="lag_stub",
    )
    paths = write_forecast_files(
        "synthetic",
        forecast,
        horizons=[7],
        output_dir=tmp_path,
        generated_at="2026-01-01T00:00:00",
        model_name="lag_stub",
    )
    assert 7 in paths
    assert paths[7]["parquet"].exists()
    assert paths[7]["summary"].exists()
    assert paths[7]["head"].exists()
    from src.utils.io import read_parquet

    stored = read_parquet(paths[7]["parquet"])
    assert set(FORECAST_COLUMNS).issubset(set(stored.columns))
    assert "dataset" in stored.columns
    assert "forecast_horizon" in stored.columns
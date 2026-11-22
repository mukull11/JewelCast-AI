"""Unit tests for the M3 forecasting evaluation layer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.forecasting.evaluation import (
    comparison_table,
    evaluate_predictions,
    metric_values,
    select_best_model,
)


def test_metric_values_perfect_predictions() -> None:
    values = metric_values([2.0, 0.0, 4.0], [2.0, 0.0, 4.0])
    assert values["n"] == 3
    assert values["mae"] == 0.0
    assert values["rmse"] == 0.0
    assert values["wape"] == 0.0
    assert values["smape"] == 0.0


def test_metric_values_compute_wape_and_smape() -> None:
    values = metric_values([2.0, 4.0], [2.0, 2.0])
    assert values["mae"] == pytest.approx(1.0)
    assert values["wape"] == pytest.approx(100.0 * 2.0 / 6.0)
    assert values["smape"] > 0.0


def test_metric_values_handle_zero_denominators() -> None:
    assert metric_values([0.0, 0.0], [0.0, 0.0])["wape"] == 0.0
    assert metric_values([0.0, 0.0], [1.0, 1.0])["wape"] != metric_values([0.0, 0.0], [1.0, 1.0])["wape"]


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sku": ["S1", "S1", "S1"],
            "category": ["Earrings", "Earrings", "Earrings"],
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "horizon_day": [1, 1, 1],
            "predicted_units": [2.0, 2.0, 2.0],
            "origin_date": pd.Timestamp("2023-12-31"),
            "model": ["lightgbm", "naive", "seasonal_naive"],
        }
    )


def _actuals() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sku": ["S1", "S1", "S1"],
            "category": ["Earrings", "Earrings", "Earrings"],
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "units_sold": [2.0, 2.0, 4.0],
        }
    )


def test_evaluate_predictions_reports_horizon_metrics() -> None:
    result = evaluate_predictions(_predictions(), _actuals(), horizons=[7, 15, 30])
    horizon_metrics = result["horizon_metrics"]
    assert not horizon_metrics.empty
    assert {"mae", "rmse", "wape", "smape"}.issubset(set(horizon_metrics.columns))
    overall = horizon_metrics.loc[horizon_metrics["level"] == "overall"]
    assert len(overall) == 9
    assert set(overall["horizon"]) == {7, 15, 30}


def test_evaluate_predictions_excludes_beyond_horizon_rows() -> None:
    predictions = _predictions().copy()
    predictions.loc[predictions["sku"] == "S1", "horizon_day"] = [1, 2, 3]
    predictions.loc[predictions["sku"] == "S1", "date"] = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    predictions.loc[predictions["sku"] == "S1", "predicted_units"] = [2.0, 2.0, 2.0]
    result = evaluate_predictions(predictions, _actuals(), horizons=[1])
    overall = result["horizon_metrics"].loc[result["horizon_metrics"]["level"] == "overall"]
    assert set(overall["horizon"]) == {1}


def test_select_best_model_picks_lowest_wape() -> None:
    metrics = pd.DataFrame(
        {
            "level": ["overall", "overall", "overall"],
            "model": ["naive", "seasonal_naive", "lightgbm"],
            "horizon": [7, 7, 7],
            "wape": [10.0, 8.0, 12.0],
        }
    )
    selection = select_best_model(metrics, metric="wape", level="overall")
    assert selection["best_model"] == "seasonal_naive"


def test_comparison_table_is_pivoted_by_horizon() -> None:
    metrics = pd.DataFrame(
        {
            "level": ["overall", "overall", "overall"],
            "model": ["naive", "seasonal_naive", "lightgbm"],
            "horizon": [7, 15, 30],
            "wape": [10.0, 8.0, 12.0],
        }
    )
    table = comparison_table(metrics, metric="wape", level="overall")
    assert set(table.columns) == {"model", "h7_wape", "h15_wape", "h30_wape"}
    row = table.loc[table["model"] == "seasonal_naive"].iloc[0]
    assert row["h15_wape"] == 8.0
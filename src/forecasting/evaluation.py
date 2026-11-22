"""Forecast evaluation metrics and aggregation.

MAE, RMSE, WAPE and sMAPE are computed with explicit zero-denominator handling.
WAPE is the primary business metric because retail demand contains many zero
days. Ordinary MAPE is not used as the primary metric.

Metrics are reported at the overall dataset level, category level and SKU level,
for the 7-, 15- and 30-day horizons. Horizon metrics compare the summed actual
and summed predicted demand per series over the horizon window. Daily metrics
are also written for transparency.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

METRIC_NAMES = ("mae", "rmse", "wape", "smape")


def _clean_pair(actual: Iterable[float], predicted: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(list(actual) if not isinstance(actual, np.ndarray) else actual, dtype=float)
    p = np.asarray(list(predicted) if not isinstance(predicted, np.ndarray) else predicted, dtype=float)
    if a.shape != p.shape:
        raise ValueError("actual and predicted must have the same shape")
    mask = np.isfinite(a) & np.isfinite(p)
    return a[mask], p[mask]


def metric_values(actual: Iterable[float], predicted: Iterable[float]) -> dict[str, float]:
    """Compute MAE, RMSE, WAPE and sMAPE for one paired vector."""
    a, p = _clean_pair(actual, predicted)
    n = int(a.size)
    if n == 0:
        return {"n": 0, "actual_sum": 0.0, "pred_sum": 0.0, **{name: float("nan") for name in METRIC_NAMES}}

    errors = a - p
    absolute = np.abs(errors)
    actual_sum = float(np.abs(a).sum())
    pred_sum = float(np.abs(p).sum())

    mae = float(absolute.mean())
    rmse = float(np.sqrt(np.mean(errors**2)))

    if actual_sum == 0:
        wape = 0.0 if pred_sum == 0 else float("nan")
    else:
        wape = float(100.0 * absolute.sum() / actual_sum)

    denominator = np.abs(a) + np.abs(p)
    positive = denominator > 0
    if positive.any():
        smape = float(200.0 * np.mean(absolute[positive] / denominator[positive]))
    else:
        smape = 0.0

    return {
        "n": n,
        "actual_sum": actual_sum,
        "pred_sum": pred_sum,
        "mae": mae,
        "rmse": rmse,
        "wape": wape,
        "smape": smape,
    }


def _metric_rows(
    frame: pd.DataFrame,
    group_columns: list[str],
    level: str,
    horizon: int | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = frame.groupby(group_columns, observed=True, dropna=False) if group_columns else [((), frame)]
    for key, group in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        values = metric_values(group["actual"], group["predicted"])
        row: dict[str, Any] = {
            "level": level,
            "group": " | ".join(str(item) for item in key) if key else "all",
            **values,
        }
        for column, value in zip(group_columns, key):
            row[column] = value
        if horizon is not None:
            row["horizon"] = int(horizon)
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_predictions(
    predictions: pd.DataFrame,
    actuals: pd.DataFrame,
    horizons: Iterable[int],
    model_column: str = "model",
) -> dict[str, pd.DataFrame]:
    """Evaluate long-format predictions against actual observations.

    ``predictions`` must contain ``sku``, ``date``, ``horizon_day``,
    ``predicted_units``, ``origin_date`` and ``model``. ``actuals`` must contain
    ``sku``, ``date`` and ``units_sold``. Actuals are kept separate from
    predictions and merged only for evaluation.
    """
    required_prediction = {"sku", "date", "horizon_day", "predicted_units", "origin_date", model_column}
    missing = required_prediction.difference(predictions.columns)
    if missing:
        raise ValueError(f"predictions are missing columns: {sorted(missing)}")
    required_actual = {"sku", "date", "units_sold"}
    missing_actual = required_actual.difference(actuals.columns)
    if missing_actual:
        raise ValueError(f"actuals are missing columns: {sorted(missing_actual)}")

    pred = predictions.copy()
    act = actuals.copy()
    pred["date"] = pd.to_datetime(pred["date"]).dt.normalize()
    act["date"] = pd.to_datetime(act["date"]).dt.normalize()
    pred["sku"] = pred["sku"].astype(str)
    act["sku"] = act["sku"].astype(str)
    if "category" not in pred.columns:
        pred["category"] = "Unknown"
    if "category" not in act.columns:
        act["category"] = "Unknown"

    merged = pred.merge(
        act.loc[:, ["sku", "date", "units_sold"]],
        on=["sku", "date"],
        how="inner",
    )
    merged = merged.rename(columns={"units_sold": "actual", "predicted_units": "predicted"})
    merged = merged.loc[np.isfinite(merged["actual"]) & np.isfinite(merged["predicted"])].copy()

    horizon_frames: list[pd.DataFrame] = []
    daily_frames: list[pd.DataFrame] = []
    for horizon in sorted({int(value) for value in horizons}):
        window = merged.loc[merged["horizon_day"] <= horizon].copy()
        if window.empty:
            continue
        daily_frames.append(_metric_rows(window, [model_column, "origin_date"], "daily", horizon=None).assign(horizon=horizon))
        daily_frames.append(
            _metric_rows(window, [model_column, "category", "origin_date"], "category_daily", horizon=None).assign(horizon=horizon)
        )

        aggregated = (
            window.groupby(
                ["origin_date", model_column, "sku", "category"],
                observed=True,
                dropna=False,
            )
            .agg(actual=("actual", "sum"), predicted=("predicted", "sum"))
            .reset_index()
        )
        overall = _metric_rows(aggregated, [model_column, "origin_date"], "overall", horizon=horizon)
        category = _metric_rows(aggregated, [model_column, "origin_date", "category"], "category", horizon=horizon)
        sku = _metric_rows(aggregated, [model_column, "origin_date", "sku", "category"], "sku", horizon=horizon)
        horizon_frames.extend([overall, category, sku])

    horizon_metrics = pd.concat(horizon_frames, ignore_index=True) if horizon_frames else pd.DataFrame()
    daily_metrics = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
    return {"horizon_metrics": horizon_metrics, "daily_metrics": daily_metrics}


def select_best_model(
    metrics: pd.DataFrame,
    metric: str = "wape",
    level: str = "overall",
) -> dict[str, Any]:
    """Rank models by mean metric across horizons on the validation period."""
    if metrics.empty:
        raise ValueError("Cannot select a model from empty metrics")
    if metric not in METRIC_NAMES:
        raise ValueError(f"Unsupported selection metric: {metric}")
    view = metrics.loc[metrics["level"] == level].copy()
    if view.empty:
        view = metrics.copy()
    scores = (
        view.groupby("model", observed=True)[metric]
        .agg(["mean", "median", "count"])
        .reset_index()
        .rename(columns={"mean": f"mean_{metric}", "median": f"median_{metric}"})
    )
    scores = scores.sort_values(f"mean_{metric}", na_position="last").reset_index(drop=True)
    best = scores.iloc[0]["model"] if not scores.empty else None
    return {"metric": metric, "level": level, "best_model": best, "scores": scores}


def comparison_table(metrics: pd.DataFrame, metric: str = "wape", level: str = "overall") -> pd.DataFrame:
    """Pivot validation/test metrics into a model x horizon comparison table."""
    view = metrics.loc[metrics["level"] == level]
    if view.empty:
        return pd.DataFrame()
    table = view.pivot_table(index="model", columns="horizon", values=metric, aggfunc="mean")
    table.columns = [f"h{int(column)}_{metric}" for column in table.columns]
    return table.reset_index().sort_values(table.columns[-1])

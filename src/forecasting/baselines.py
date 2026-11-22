"""Simple, strong baselines evaluated with the same chronological protocol.

Baselines are intentionally simple and are never skipped. They provide the
minimum benchmark that a machine-learning model must beat.

- Naive: repeat the most recent observed demand.
- Seasonal naive: repeat the value from seven days earlier.
- Moving average: mean of the last seven observed or predicted values.

For future days, baselines are applied recursively using their own previous
predictions. A category-level fallback is used only when a series has no usable
history at all.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

BASELINE_MODELS: tuple[str, ...] = ("naive", "seasonal_naive", "moving_average")
DEFAULT_WINDOW = 7


def category_fallback_map(
    panel: pd.DataFrame,
    origin: pd.Timestamp,
) -> tuple[dict[str, float], float]:
    """Return category mean daily demand and the global mean from history only."""
    origin = pd.Timestamp(origin).normalize()
    history = panel.loc[panel["date"] <= origin]
    if history.empty:
        return {}, 0.0
    category_means = history.groupby("category", observed=True)["units_sold"].mean()
    global_mean = float(history["units_sold"].mean())
    return {str(key): float(value) for key, value in category_means.items()}, global_mean


def fallback_values_for_series(
    series: pd.Index,
    category_by_series: Mapping[str, str] | pd.Series,
    category_means: Mapping[str, float],
    global_mean: float,
) -> np.ndarray:
    """Align category fallbacks to the requested series order."""
    if isinstance(category_by_series, pd.Series):
        categories = category_by_series.reindex(series).astype(str).to_numpy()
    else:
        categories = np.array([str(category_by_series.get(str(sku), "")) for sku in series], dtype=object)
    values = np.array(
        [float(category_means.get(str(category), global_mean)) for category in categories],
        dtype=float,
    )
    values[~np.isfinite(values)] = max(float(global_mean), 0.0)
    return np.clip(values, 0.0, None)


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    window_values = values[:, -int(window) :]
    counts = np.sum(~np.isnan(window_values), axis=1)
    sums = np.nansum(window_values, axis=1)
    result = np.full(values.shape[0], np.nan, dtype=float)
    valid = counts > 0
    result[valid] = sums[valid] / counts[valid]
    return result


def forecast_baselines(
    panel: pd.DataFrame,
    origin: pd.Timestamp,
    horizon: int,
    category_by_series: Mapping[str, str] | pd.Series | None = None,
    models: Sequence[str] = BASELINE_MODELS,
    max_history: int = 60,
    window: int = DEFAULT_WINDOW,
) -> dict[str, pd.DataFrame]:
    """Generate recursive baseline forecasts for every series in ``panel``.

    Returns one long DataFrame per baseline with columns ``sku``, ``category``,
    ``date``, ``horizon_day``, ``predicted_units``, ``origin_date`` and
    ``model``. Actual values are never included.
    """
    origin = pd.Timestamp(origin).normalize()
    horizon = int(horizon)
    if horizon <= 0:
        raise ValueError("horizon must be a positive integer")

    if category_by_series is None:
        category_by_series = (
            panel.drop_duplicates(subset=["sku"], keep="last").set_index("sku")["category"]
        )
    elif not isinstance(category_by_series, pd.Series):
        category_by_series = pd.Series(category_by_series, dtype="object")

    series_index = pd.Index(sorted(panel["sku"].astype(str).unique()), name="sku")
    history = panel.loc[panel["date"] <= origin, ["sku", "date", "units_sold"]].copy()
    if not history.empty:
        history = history.groupby("sku", sort=False, observed=True).tail(int(max_history))
        history["sku"] = history["sku"].astype(str)
        pivot = history.pivot(index="sku", columns="date", values="units_sold")
        pivot = pivot.reindex(series_index)
    else:
        pivot = pd.DataFrame(index=series_index, columns=[], dtype=float)

    values = pivot.to_numpy(dtype=float)
    if values.size == 0:
        values = np.full((len(series_index), 0), np.nan, dtype=float)
    if values.shape[1] < window:
        padding = np.full((len(series_index), window - values.shape[1]), np.nan, dtype=float)
        values = np.column_stack([padding, values])

    category_means, global_mean = category_fallback_map(panel, origin)
    fallback = fallback_values_for_series(series_index, category_by_series, category_means, global_mean)

    future_dates = pd.date_range(origin + pd.Timedelta(days=1), periods=horizon, freq="D")
    predictions = {name: np.full((len(series_index), horizon), np.nan, dtype=float) for name in models}
    histories = {name: values.copy() for name in models}

    for step in range(horizon):
        for name in models:
            model_history = histories[name]
            last = model_history[:, -1]
            seasonal = model_history[:, -window] if model_history.shape[1] >= window else last
            moving = _moving_average(model_history, window)
            candidates = {
                "naive": last,
                "seasonal_naive": seasonal,
                "moving_average": moving,
            }
            if name not in candidates:
                raise ValueError(f"Unsupported baseline model: {name}")
            pred = np.asarray(candidates[name], dtype=float)
            pred = np.where(np.isnan(pred), fallback, pred)
            pred = np.clip(pred, 0.0, None)
            predictions[name][:, step] = pred
            histories[name] = np.column_stack([model_history, pred])

    result: dict[str, pd.DataFrame] = {}
    categories = category_by_series.reindex(series_index).astype(str).to_numpy()
    sku_values = np.repeat(series_index.to_numpy(), horizon)
    date_values = np.tile(future_dates.to_numpy(), len(series_index))
    horizon_values = np.tile(np.arange(1, horizon + 1), len(series_index))
    category_values = np.repeat(categories, horizon)
    for name in models:
        frame = pd.DataFrame(
            {
                "sku": sku_values,
                "category": category_values,
                "date": pd.to_datetime(date_values),
                "horizon_day": horizon_values.astype("int16"),
                "predicted_units": predictions[name].reshape(-1),
                "origin_date": origin,
                "model": name,
            }
        )
        frame["predicted_units"] = frame["predicted_units"].clip(lower=0.0)
        result[name] = frame
    return result

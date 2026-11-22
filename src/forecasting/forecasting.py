"""Recursive multi-horizon forecasting and the M3 orchestration pipeline.

Strategy
--------
A global one-step-ahead model is trained per dataset. Multi-day forecasts are
generated recursively: after predicting day *t+1*, the prediction is appended to
the history and used for the lag and rolling features of day *t+2*. Actual
future targets are never used. Calendar features are known in advance.

The 7-, 15- and 30-day horizons are derived from the same 30-day recursive path,
so all horizons share one consistent forecasting process.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .baselines import BASELINE_MODELS, category_fallback_map, fallback_values_for_series, forecast_baselines
from .data_builder import EVENT_COLUMNS, ForecastingPanel, load_forecasting_panel
from .evaluation import comparison_table, evaluate_predictions, select_best_model
from .feature_engineering import HISTORY_DAYS, add_category_price_feature, build_features, feature_columns
from .models import GlobalDemandModel, build_model
from .validation import ChronologicalSplit, chronological_split, rolling_origins

LOGGER = logging.getLogger(__name__)


def _category_price_map(panel: ForecastingPanel, origin: pd.Timestamp) -> dict[str, float]:
    history = panel.frame.loc[panel.frame["date"] <= origin]
    prices = history.groupby("category", observed=True)["selling_price"].median().dropna()
    if prices.empty:
        return {}
    global_median = float(history["selling_price"].median())
    if not np.isfinite(global_median):
        global_median = 0.0
    mapping = {str(key): float(value) for key, value in prices.items()}
    for category in panel.category_codes:
        mapping.setdefault(str(category), global_median)
    return mapping


def _training_frame(
    features: pd.DataFrame,
    origin: pd.Timestamp,
    feature_names: Sequence[str],
    max_rows: int | None,
) -> pd.DataFrame:
    train = features.loc[
        (features["date"] <= origin)
        & features["units_sold"].notna()
        & features["lag_1"].notna()
    ].copy()
    if train.empty:
        raise ValueError(f"No training rows available before origin {origin}")
    if max_rows is not None and len(train) > int(max_rows):
        train = train.sort_values(["date", "sku"], kind="mergesort").tail(int(max_rows)).copy()
    missing = [column for column in feature_names if column not in train.columns]
    if missing:
        raise ValueError(f"Training frame is missing features: {missing}")
    return train


def _calendar_scalar(calendar: pd.DataFrame, date: pd.Timestamp, column: str, default: Any) -> Any:
    row = calendar.loc[calendar["date"] == date]
    if row.empty:
        return default
    value = row[column].iloc[0]
    return default if pd.isna(value) and default is not None else value


def _fallback_frame(
    series: pd.Index,
    fallback_lookup: pd.Series,
    category_by_series: pd.Series,
    origin: pd.Timestamp,
    horizon: int,
    model_name: str,
) -> pd.DataFrame:
    if len(series) == 0:
        return pd.DataFrame(
            columns=["sku", "category", "date", "horizon_day", "predicted_units", "origin_date", "model"]
        )
    future_dates = pd.date_range(origin + pd.Timedelta(days=1), periods=horizon, freq="D")
    categories = category_by_series.reindex(series).astype(str).to_numpy()
    values = fallback_lookup.reindex(series).fillna(0.0).to_numpy()
    return pd.DataFrame(
        {
            "sku": np.repeat(series.to_numpy(), horizon),
            "category": np.repeat(categories, horizon),
            "date": np.tile(future_dates.to_numpy(), len(series)),
            "horizon_day": np.tile(np.arange(1, horizon + 1), len(series)).astype("int16"),
            "predicted_units": np.repeat(values, horizon).clip(min=0.0),
            "origin_date": origin,
            "model": model_name,
        }
    )


def forecast_ml(
    model: GlobalDemandModel,
    panel: ForecastingPanel,
    origin: pd.Timestamp,
    horizon: int,
    feature_names: Sequence[str],
    config: Mapping[str, Any],
    category_price_map: Mapping[str, float] | None = None,
    model_name: str | None = None,
    min_history_for_model: int | None = None,
) -> pd.DataFrame:
    """Recursively forecast daily demand with a trained global model.

    Series with fewer than ``min_history_for_model`` observations at the origin,
    or series that do not exist before the origin, receive a documented
    category-level fallback instead of a model prediction. The model is never
    presented as having learned an unseen SKU's history.
    """
    origin = pd.Timestamp(origin).normalize()
    horizon = int(horizon)
    if horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    model_name = model_name or model.model_type
    min_history = int(
        min_history_for_model
        if min_history_for_model is not None
        else (config.get("forecasting", {}) or {}).get("min_history_for_model", 14)
    )
    if category_price_map is None:
        category_price_map = _category_price_map(panel, origin)

    frame = panel.frame
    all_series = pd.Index(sorted(frame["sku"].astype(str).unique()), name="sku")
    category_by_series = frame.drop_duplicates(subset=["sku"], keep="last").set_index("sku")["category"]
    category_by_series.index = category_by_series.index.astype(str)

    history_counts = frame.loc[frame["date"] <= origin].groupby("sku", observed=True).size()
    history_counts.index = history_counts.index.astype(str)
    model_series = pd.Index(sorted(history_counts[history_counts >= min_history].index))
    fallback_series = all_series.difference(model_series)

    category_means, global_mean = category_fallback_map(frame, origin)
    fallback = fallback_values_for_series(all_series, category_by_series, category_means, global_mean)
    fallback_lookup = pd.Series(fallback, index=all_series)

    if len(model_series) == 0:
        return _fallback_frame(fallback_series, fallback_lookup, category_by_series, origin, horizon, model_name)

    history = frame.loc[(frame["date"] <= origin) & (frame["sku"].astype(str).isin(model_series))].copy()
    history = history.groupby("sku", sort=False, observed=True).tail(HISTORY_DAYS).copy()
    history["sku"] = history["sku"].astype(str)
    if history.empty:
        return _fallback_frame(fallback_series, fallback_lookup, category_by_series, origin, horizon, model_name)

    calendar = panel.future_calendar(origin, horizon, config)
    future_dates = pd.date_range(origin + pd.Timedelta(days=1), periods=horizon, freq="D")
    records: list[pd.DataFrame] = []

    working = history
    for step, future_date in enumerate(future_dates, start=1):
        last = working.groupby("sku", sort=False, observed=True).tail(1).copy()
        last["sku"] = last["sku"].astype(str)
        last["date"] = future_date
        last["units_sold"] = np.nan
        last["revenue"] = 0.0
        last["promotion_flag"] = 0
        last["discount_percentage"] = 0.0
        last["available_stock"] = np.nan
        last["stockout_flag"] = pd.NA
        last["date_imputed"] = 0
        last["selling_price"] = last["selling_price"].fillna(
            last["category"].map(category_price_map)
        )
        for column, default in (
            ("festival_flag", 0),
            ("festival_name", None),
            ("festival_strength", 1.0),
            ("wedding_season_flag", 0),
            ("wedding_strength", 1.0),
            ("event_name", None),
            ("event_flag", 0),
            ("event_code", -1),
        ):
            last[column] = _calendar_scalar(calendar, future_date, column, default)

        working = pd.concat([working, last], ignore_index=True)
        built = build_features(
            working,
            panel.mode,
            first_dates=panel.first_dates,
            category_price_map=category_price_map,
        )
        current = built.loc[built["date"] == future_date].copy()
        if current.empty:
            continue
        predictions = model.predict(current)
        prediction_series = pd.Series(
            np.clip(np.asarray(predictions, dtype=float), 0.0, None),
            index=current["sku"].astype(str).to_numpy(),
        )
        mask = working["date"] == future_date
        aligned = working.loc[mask, "sku"].astype(str).map(prediction_series)
        aligned = aligned.fillna(working.loc[mask, "sku"].astype(str).map(fallback_lookup))
        working.loc[mask, "units_sold"] = aligned.to_numpy()

        current["predicted_units"] = prediction_series.reindex(
            current["sku"].astype(str).to_numpy()
        ).to_numpy()
        current["predicted_units"] = current["predicted_units"].fillna(
            current["sku"].astype(str).map(fallback_lookup)
        ).clip(lower=0.0)
        current["horizon_day"] = step
        current["origin_date"] = origin
        current["model"] = model_name
        records.append(current.loc[:, ["sku", "category", "date", "horizon_day", "predicted_units", "origin_date", "model"]])

        working = working.groupby("sku", sort=False, observed=True).tail(HISTORY_DAYS).copy()

    if records:
        model_predictions = pd.concat(records, ignore_index=True)
    else:
        model_predictions = pd.DataFrame(
            columns=["sku", "category", "date", "horizon_day", "predicted_units", "origin_date", "model"]
        )
    fallback_predictions = _fallback_frame(
        fallback_series, fallback_lookup, category_by_series, origin, horizon, model_name
    )
    if fallback_predictions.empty:
        result = model_predictions
    else:
        result = pd.concat([model_predictions, fallback_predictions], ignore_index=True)
    result["predicted_units"] = result["predicted_units"].clip(lower=0.0)
    return result.sort_values(["sku", "date"], kind="mergesort").reset_index(drop=True)


def production_forecast(
    panel: ForecastingPanel,
    model: GlobalDemandModel,
    origin: pd.Timestamp,
    horizon: int,
    feature_names: Sequence[str],
    config: Mapping[str, Any],
    category_price_map: Mapping[str, float] | None = None,
    model_name: str | None = None,
) -> pd.DataFrame:
    """Generate a forecast from a known origin without using future actuals."""
    return forecast_ml(
        model,
        panel,
        origin=origin,
        horizon=horizon,
        feature_names=feature_names,
        config=config,
        category_price_map=category_price_map,
        model_name=model_name,
    )


def _runtime_setting(config: Mapping[str, Any], mode: str, key: str, default: Any) -> Any:
    forecasting = config.get("forecasting", {}) or {}
    runtime = (forecasting.get("runtime", {}) or {}).get(mode, {}) or {}
    if key in runtime:
        return runtime[key]
    return forecasting.get(key, default)


def _max_training_rows(config: Mapping[str, Any], mode: str) -> int | None:
    value = _runtime_setting(config, mode, "max_training_rows", None)
    return None if value in (None, 0) else int(value)


def _fit_for_origin(
    mode: str,
    config: Mapping[str, Any],
    panel: ForecastingPanel,
    features: pd.DataFrame,
    feature_names: Sequence[str],
    origin: pd.Timestamp,
    model_type: str | None,
) -> tuple[GlobalDemandModel, pd.DataFrame]:
    category_price_map = _category_price_map(panel, origin)
    add_category_price_feature(features, category_price_map)
    train = _training_frame(features, origin, feature_names, _max_training_rows(config, mode))
    model = build_model(mode, config, model_type=model_type)
    model.fit(
        train.loc[:, feature_names],
        train["units_sold"],
        feature_names,
        training_metadata={
            "origin": str(pd.Timestamp(origin).date()),
            "training_date_min": str(train["date"].min().date()),
            "training_date_max": str(train["date"].max().date()),
            "training_skus": int(train["sku"].nunique()),
            "category_price_map": {str(key): float(value) for key, value in category_price_map.items()},
        },
    )
    return model, train


@dataclass
class ForecastingRun:
    mode: str
    panel: ForecastingPanel
    split: ChronologicalSplit
    features: pd.DataFrame
    feature_names: list[str]
    validation_origins: list[pd.Timestamp]
    test_origins: list[pd.Timestamp]
    validation_predictions: pd.DataFrame
    validation_actuals: pd.DataFrame
    test_predictions: pd.DataFrame
    test_actuals: pd.DataFrame
    validation_metrics: dict[str, pd.DataFrame]
    test_metrics: dict[str, pd.DataFrame]
    selection: dict[str, Any]
    selected_model_name: str
    selected_model: GlobalDemandModel | None
    feature_importance: pd.DataFrame
    runtime_seconds: float
    metadata: dict[str, Any]


def run_forecasting_pipeline(
    mode: str,
    config: Mapping[str, Any],
    processed_dir: str | None = None,
    sku_limit: int | None = None,
    model_type: str | None = None,
    backtest_origins: int | None = None,
    test_origins: int | None = None,
) -> ForecastingRun:
    """Train, validate, select and test the forecasting engine for one mode."""
    started = time.perf_counter()
    panel = load_forecasting_panel(mode, config, processed_dir=processed_dir, sku_limit=sku_limit)
    features = build_features(panel.frame, mode, first_dates=panel.first_dates)
    feature_names = feature_columns(features, mode)
    if not feature_names:
        raise ValueError(f"[{mode}] no forecasting features were generated")

    split = chronological_split(panel.dates, mode, config)
    horizons = [int(value) for value in (config.get("forecasting", {}) or {}).get("horizons", [7, 15, 30])]
    max_horizon = max(horizons)

    n_backtest = int(
        backtest_origins
        if backtest_origins is not None
        else _runtime_setting(config, mode, "backtest_origins", 2)
    )
    n_test = int(
        test_origins
        if test_origins is not None
        else _runtime_setting(config, mode, "test_origins", 1)
    )
    validation_origins = rolling_origins(split, "validation", max_horizon, n_origins=n_backtest)
    test_origins_list = rolling_origins(split, "test", max_horizon, n_origins=n_test)
    category_by_series = panel.frame.drop_duplicates(subset=["sku"], keep="last").set_index("sku")["category"]
    category_by_series.index = category_by_series.index.astype(str)

    validation_predictions: list[pd.DataFrame] = []
    validation_actuals: list[pd.DataFrame] = []
    test_predictions: list[pd.DataFrame] = []
    test_actuals: list[pd.DataFrame] = []
    last_model: GlobalDemandModel | None = None
    last_origin: pd.Timestamp | None = None

    for origin in validation_origins:
        LOGGER.info("[%s] validation origin %s", mode, origin.date())
        model, _ = _fit_for_origin(mode, config, panel, features, feature_names, origin, model_type)
        category_price_map = _category_price_map(panel, origin)
        ml = forecast_ml(
            model,
            panel,
            origin,
            max_horizon,
            feature_names,
            config,
            category_price_map=category_price_map,
            model_name=model.model_type,
        )
        baselines = forecast_baselines(
            panel.frame,
            origin,
            max_horizon,
            category_by_series=category_by_series,
            models=BASELINE_MODELS,
        )
        validation_predictions.append(pd.concat([ml, *baselines.values()], ignore_index=True))
        validation_actuals.append(
            panel.frame.loc[
                (panel.frame["date"] > origin)
                & (panel.frame["date"] <= origin + pd.Timedelta(days=max_horizon)),
                ["sku", "category", "date", "units_sold"],
            ].copy()
        )

    validation_pred = pd.concat(validation_predictions, ignore_index=True)
    validation_act = pd.concat(validation_actuals, ignore_index=True)
    validation_metric_frames: list[dict[str, pd.DataFrame]] = []
    for origin, pred_frame, act_frame in zip(validation_origins, validation_predictions, validation_actuals):
        origin_metrics = evaluate_predictions(pred_frame, act_frame, horizons)
        validation_metric_frames.append(origin_metrics)
    validation_metrics = (
        {
            "horizon_metrics": pd.concat(
                [frame["horizon_metrics"] for frame in validation_metric_frames],
                ignore_index=True,
            ),
            "daily_metrics": pd.concat(
                [frame["daily_metrics"] for frame in validation_metric_frames],
                ignore_index=True,
            ),
        }
        if validation_metric_frames
        else {"horizon_metrics": pd.DataFrame(), "daily_metrics": pd.DataFrame()}
    )
    selection = select_best_model(validation_metrics["horizon_metrics"], metric="wape", level="overall")
    selected_model_name = str(selection["best_model"])

    for origin in test_origins_list:
        LOGGER.info("[%s] test origin %s", mode, origin.date())
        model, _ = _fit_for_origin(mode, config, panel, features, feature_names, origin, model_type)
        category_price_map = _category_price_map(panel, origin)
        ml = forecast_ml(
            model,
            panel,
            origin,
            max_horizon,
            feature_names,
            config,
            category_price_map=category_price_map,
            model_name=model.model_type,
        )
        baselines = forecast_baselines(
            panel.frame,
            origin,
            max_horizon,
            category_by_series=category_by_series,
            models=BASELINE_MODELS,
        )
        test_predictions.append(pd.concat([ml, *baselines.values()], ignore_index=True))
        test_actuals.append(
            panel.frame.loc[
                (panel.frame["date"] > origin)
                & (panel.frame["date"] <= origin + pd.Timedelta(days=max_horizon)),
                ["sku", "category", "date", "units_sold"],
            ].copy()
        )
        last_model = model
        last_origin = origin

    test_pred = pd.concat(test_predictions, ignore_index=True)
    test_act = pd.concat(test_actuals, ignore_index=True)
    test_metric_frames: list[dict[str, pd.DataFrame]] = []
    for origin, pred_frame, act_frame in zip(test_origins_list, test_predictions, test_actuals):
        origin_metrics = evaluate_predictions(pred_frame, act_frame, horizons)
        test_metric_frames.append(origin_metrics)
    test_metrics = (
        {
            "horizon_metrics": pd.concat(
                [frame["horizon_metrics"] for frame in test_metric_frames],
                ignore_index=True,
            ),
            "daily_metrics": pd.concat(
                [frame["daily_metrics"] for frame in test_metric_frames],
                ignore_index=True,
            ),
        }
        if test_metric_frames
        else {"horizon_metrics": pd.DataFrame(), "daily_metrics": pd.DataFrame()}
    )

    feature_importance = last_model.feature_importance() if last_model is not None else pd.DataFrame()
    runtime = time.perf_counter() - started
    metadata = {
        "mode": mode,
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "panel": panel.metadata,
        "split": split.as_dict(),
        "horizons": horizons,
        "max_horizon": max_horizon,
        "validation_origins": [str(origin.date()) for origin in validation_origins],
        "test_origins": [str(origin.date()) for origin in test_origins_list],
        "backtest_origins_requested": n_backtest,
        "test_origins_requested": n_test,
        "model_type": last_model.model_type if last_model is not None else None,
        "model_label": last_model.label if last_model is not None else None,
        "feature_names": feature_names,
        "max_training_rows": _max_training_rows(config, mode),
        "selected_model": selected_model_name,
        "selection_metric": selection["metric"],
        "runtime_seconds": round(runtime, 2),
        "provenance": {
            "synthetic": "Fully simulated Indian jewellery data; not real company data.",
            "uci": "UCI Online Retail II; UK online retail in GBP; not Indian or jewellery-only data.",
            "m5": "M5 Walmart benchmark; not jewellery data and not Voylla data.",
        }.get(mode, mode),
        "target_definition": "Observed daily units_sold at SKU/series level.",
        "stockout_censoring": (
            "Synthetic sales are censored at zero stock. Forecasts target observed sales, not latent unconstrained demand."
            if mode == "synthetic"
            else "No inventory records exist; stockout-censored demand cannot be identified."
        ),
    }

    return ForecastingRun(
        mode=mode,
        panel=panel,
        split=split,
        features=features,
        feature_names=feature_names,
        validation_origins=validation_origins,
        test_origins=test_origins_list,
        validation_predictions=validation_pred,
        validation_actuals=validation_act,
        test_predictions=test_pred,
        test_actuals=test_act,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        selection=selection,
        selected_model_name=selected_model_name,
        selected_model=last_model,
        feature_importance=feature_importance,
        runtime_seconds=runtime,
        metadata=metadata,
    )

"""Adapters that connect M3 forecasts to inventory decisions.

Look-ahead rules
----------------
- A forecast with origin *o* may use observations up to and including *o* only.
- ``ModelForecastProvider(mode="frozen")`` fits one global model at the first
  simulation origin and reuses it. It never sees data after the simulation start.
- ``ModelForecastProvider(mode="rolling")`` refits every ``refit_every_days`` at
  the then-current origin, still using only prior observations.
- ``TrailingMeanForecastProvider`` is the causal, model-free baseline used by the
  simple reorder-point policy.
- ``ArtifactForecastProvider`` reads persisted M3 forecasts for an existing
  origin. The caller must verify that the model artefact's training cutoff does
  not post-date the forecast origin.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Protocol

import numpy as np
import pandas as pd

from src.forecasting.data_builder import ForecastingPanel
from src.forecasting.feature_engineering import build_features, feature_columns
from src.forecasting.forecasting import production_forecast
from src.forecasting.models import GlobalDemandModel, build_model
from src.utils.io import read_parquet
from src.utils.paths import resolve_path

LOGGER = logging.getLogger(__name__)

FORECAST_OUTPUT_COLUMNS = ("sku", "date", "predicted_units", "forecast_source")


class ForecastProvider(Protocol):
    name: str

    def forecast(self, origin: pd.Timestamp, horizon: int) -> pd.DataFrame:
        ...


def _normalise_forecast(frame: pd.DataFrame, source: str, skus: pd.Index | None = None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=list(FORECAST_OUTPUT_COLUMNS))
    view = frame.copy()
    view["sku"] = view["sku"].astype(str)
    view["date"] = pd.to_datetime(view["date"]).dt.normalize()
    view["predicted_units"] = pd.to_numeric(view["predicted_units"], errors="coerce").fillna(0.0).clip(lower=0.0)
    if "model" in view.columns:
        view["forecast_source"] = view["model"].astype(str)
    else:
        view["forecast_source"] = source
    view = view.sort_values(["sku", "date"]).drop_duplicates(subset=["sku", "date"], keep="last")
    if skus is not None:
        view = view.loc[view["sku"].isin(set(skus.astype(str)))]
    return view.loc[:, list(FORECAST_OUTPUT_COLUMNS)].reset_index(drop=True)


class TrailingMeanForecastProvider:
    """Causal trailing-average forecast used by the simple replenishment policy."""

    name = "trailing_mean"

    def __init__(self, panel: ForecastingPanel, window_days: int = 28, min_history: int = 7) -> None:
        self.panel = panel
        self.window_days = max(1, int(window_days))
        self.min_history = max(1, int(min_history))

    def forecast(self, origin: pd.Timestamp, horizon: int) -> pd.DataFrame:
        origin = pd.Timestamp(origin).normalize()
        horizon = int(horizon)
        history = self.panel.frame.loc[self.panel.frame["date"] <= origin]
        history = history.sort_values(["sku", "date"], kind="mergesort").groupby(
            "sku", observed=True
        ).tail(self.window_days)
        means = history.groupby("sku", observed=True)["units_sold"].mean()
        counts = history.groupby("sku", observed=True)["units_sold"].count()
        category_means = history.groupby("category", observed=True)["units_sold"].mean()
        latest_category = self.panel.frame.drop_duplicates(subset=["sku"], keep="last").set_index("sku")["category"]
        future_dates = pd.date_range(origin + pd.Timedelta(days=1), periods=horizon, freq="D")
        rows: list[pd.DataFrame] = []
        for sku in sorted(self.panel.frame["sku"].astype(str).unique()):
            count = int(counts.get(sku, 0))
            if count >= self.min_history:
                value = float(means.get(sku, 0.0))
                source = self.name
            else:
                category = str(latest_category.get(sku, "Unknown"))
                value = float(category_means.get(category, 0.0))
                source = f"{self.name}_category_fallback"
            rows.append(
                pd.DataFrame(
                    {
                        "sku": sku,
                        "date": future_dates,
                        "predicted_units": max(0.0, value),
                        "forecast_source": source,
                    }
                )
            )
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=list(FORECAST_OUTPUT_COLUMNS))


class ModelForecastProvider:
    """Causal adapter around the M3 global model."""

    name = "m3_model"

    def __init__(
        self,
        panel: ForecastingPanel,
        config: Mapping[str, Any],
        mode: str = "frozen",
        refit_every_days: int = 30,
        max_training_rows: int | None = None,
        model_type: str | None = None,
    ) -> None:
        if mode not in {"frozen", "rolling"}:
            raise ValueError("forecast mode must be 'frozen' or 'rolling'")
        self.panel = panel
        self.config = config
        self.mode = mode
        self.refit_every_days = max(1, int(refit_every_days))
        self.max_training_rows = max_training_rows
        self.model_type = model_type
        self._model: GlobalDemandModel | None = None
        self._fit_origin: pd.Timestamp | None = None
        self._feature_names: list[str] = []
        self.fit_origins: list[pd.Timestamp] = []

    @property
    def name(self) -> str:  # type: ignore[override]
        return f"m3_model_{self.mode}"

    def _category_price_map(self, origin: pd.Timestamp) -> dict[str, float]:
        history = self.panel.frame.loc[self.panel.frame["date"] <= origin]
        prices = history.groupby("category", observed=True)["selling_price"].median().dropna()
        global_median = float(history["selling_price"].median())
        if not np.isfinite(global_median):
            global_median = 0.0
        mapping = {str(key): float(value) for key, value in prices.items()}
        for category in self.panel.category_codes:
            mapping.setdefault(str(category), global_median)
        return mapping

    def _fit(self, origin: pd.Timestamp) -> None:
        origin = pd.Timestamp(origin).normalize()
        price_map = self._category_price_map(origin)
        features = build_features(
            self.panel.frame,
            self.panel.mode,
            first_dates=self.panel.first_dates,
            category_price_map=price_map,
        )
        self._feature_names = feature_columns(features, self.panel.mode)
        train = features.loc[
            (features["date"] <= origin) & features["units_sold"].notna() & features["lag_1"].notna()
        ].copy()
        if train.empty:
            raise ValueError(f"No causal training rows available before origin {origin.date()}")
        if self.max_training_rows is not None and len(train) > int(self.max_training_rows):
            train = train.sort_values(["date", "sku"], kind="mergesort").tail(int(self.max_training_rows)).copy()
        model = build_model(self.panel.mode, self.config, model_type=self.model_type)
        model.fit(
            train.loc[:, self._feature_names],
            train["units_sold"],
            self._feature_names,
            training_metadata={
                "origin": str(origin.date()),
                "training_date_min": str(train["date"].min().date()),
                "training_date_max": str(train["date"].max().date()),
                "training_skus": int(train["sku"].nunique()),
                "inventory_forecast_adapter": True,
                "category_price_map": price_map,
            },
        )
        self._model = model
        self._fit_origin = origin
        self.fit_origins.append(origin)
        LOGGER.info(
            "[inventory] fitted %s model at origin %s on %s rows",
            model.model_type,
            origin.date(),
            len(train),
        )

    def forecast(self, origin: pd.Timestamp, horizon: int) -> pd.DataFrame:
        origin = pd.Timestamp(origin).normalize()
        if self._model is None:
            self._fit(origin)
        elif self.mode == "rolling" and (
            self._fit_origin is None or (origin - self._fit_origin).days >= self.refit_every_days
        ):
            self._fit(origin)
        assert self._model is not None
        frame = production_forecast(
            self.panel,
            self._model,
            origin=origin,
            horizon=int(horizon),
            feature_names=list(self._model.feature_names_),
            config=self.config,
            model_name=self._model.model_type,
        )
        return _normalise_forecast(frame, self.name, skus=self.panel.frame["sku"].unique())


class SavedModelForecastProvider:
    """Use an existing M3 model artefact only if its training cutoff is past-safe.

    The artefact is never refitted. If its recorded ``training_date_max`` is
    later than the requested forecast origin, the provider refuses to forecast,
    preventing accidental look-ahead.
    """

    name = "m3_saved_model"

    def __init__(self, panel: ForecastingPanel, config: Mapping[str, Any], model_dir: str | Path) -> None:
        self.panel = panel
        self.config = config
        self.model = GlobalDemandModel.load(model_dir, config)
        self.model_dir = resolve_path(model_dir)
        cutoff = self.model.metadata.get("training_date_max") or self.model.metadata.get("origin")
        self.training_cutoff = pd.Timestamp(cutoff).normalize() if cutoff else None
        self.fit_origins: list[pd.Timestamp] = [self.training_cutoff] if self.training_cutoff is not None else []

    def forecast(self, origin: pd.Timestamp, horizon: int) -> pd.DataFrame:
        origin = pd.Timestamp(origin).normalize()
        if self.training_cutoff is None:
            raise ValueError(
                f"Model artefact {self.model_dir} has no recorded training cutoff; refusing to use it "
                "without a look-ahead guarantee."
            )
        if self.training_cutoff > origin:
            raise ValueError(
                f"Look-ahead refused: model artefact was trained through {self.training_cutoff.date()} "
                f"but the forecast origin is {origin.date()}."
            )
        frame = production_forecast(
            self.panel,
            self.model,
            origin=origin,
            horizon=int(horizon),
            feature_names=list(self.model.feature_names_),
            config=self.config,
            model_name=self.model.model_type,
        )
        return _normalise_forecast(frame, self.name, skus=self.panel.frame["sku"].unique())


class ArtifactForecastProvider:
    """Read persisted M3 forecast Parquet files for a given origin."""

    def __init__(self, path: str | Path, model_name: str | None = None) -> None:
        self.path = resolve_path(path)
        self.model_name = model_name
        self.name = "m3_artifact"
        if not self.path.exists():
            raise FileNotFoundError(
                f"Forecast artefact not found: {self.path}. Run scripts/generate_forecasts.py first."
            )
        self.frame = read_parquet(self.path)
        required = {"sku", "forecast_date", "predicted_units"}
        missing = required.difference(self.frame.columns)
        if missing:
            raise ValueError(f"forecast artefact is missing columns: {sorted(missing)}")
        self.frame["forecast_date"] = pd.to_datetime(self.frame["forecast_date"]).dt.normalize()
        if "forecast_origin_date" in self.frame.columns:
            self.frame["forecast_origin_date"] = pd.to_datetime(self.frame["forecast_origin_date"]).dt.normalize()
        else:
            raise ValueError("forecast artefact must include forecast_origin_date")

    def forecast(self, origin: pd.Timestamp, horizon: int) -> pd.DataFrame:
        origin = pd.Timestamp(origin).normalize()
        view = self.frame.loc[
            (self.frame["forecast_origin_date"] == origin)
            & (self.frame["forecast_date"] >= origin)
            & (self.frame["forecast_date"] <= origin + pd.Timedelta(days=int(horizon)))
        ].copy()
        if self.model_name is not None and "model" in view.columns:
            view = view.loc[view["model"].astype(str) == str(self.model_name)]
        view = view.rename(columns={"forecast_date": "date"})
        return _normalise_forecast(view, self.name)

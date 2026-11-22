"""CPU-friendly global demand models.

A single global model is trained per dataset mode. The preferred estimator is
LightGBM; XGBoost and scikit-learn's HistGradientBoostingRegressor are
documented fallbacks. The target is transformed with ``log1p`` and predictions
are inverted and clipped at zero.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

from src.utils.io import write_json
from src.utils.paths import resolve_path

LOGGER = logging.getLogger(__name__)

MODEL_LABELS = {
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
    "sklearn_hist": "HistGradientBoosting",
}


def _import_lightgbm():
    try:
        import lightgbm as lgb

        return lgb
    except ImportError:
        return None


def _import_xgboost():
    try:
        import xgboost as xgb

        return xgb
    except ImportError:
        return None


def _import_sklearn():
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor
    except ImportError:
        return None


def available_model_types() -> list[str]:
    available: list[str] = []
    if _import_lightgbm() is not None:
        available.append("lightgbm")
    if _import_xgboost() is not None:
        available.append("xgboost")
    if _import_sklearn() is not None:
        available.append("sklearn_hist")
    return available


def resolve_model_type(config: Mapping[str, Any], requested: str | None = None) -> str:
    """Resolve the preferred model and fall back only when a library is absent."""
    model_config = (config.get("forecasting", {}) or {}).get("model", {}) or {}
    prefer = str(requested or model_config.get("prefer", "lightgbm"))
    order = [prefer] + [str(item) for item in model_config.get("fallback_order", []) if str(item) != prefer]
    available = set(available_model_types())
    for candidate in order:
        if candidate in available:
            return candidate
    raise RuntimeError(
        "No supported gradient-boosting library is installed. Install lightgbm, xgboost or scikit-learn."
    )


@dataclass
class GlobalDemandModel:
    mode: str
    model_type: str
    config: Mapping[str, Any]
    model: Any = None
    feature_names_: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return MODEL_LABELS.get(self.model_type, self.model_type)

    def _params(self) -> dict[str, Any]:
        model_config = (self.config.get("forecasting", {}) or {}).get("model", {}) or {}
        seed = int(model_config.get("random_state", self.config.get("project", {}).get("seed", 42)))
        return {
            "n_estimators": int(model_config.get("n_estimators", 250)),
            "learning_rate": float(model_config.get("learning_rate", 0.05)),
            "num_leaves": int(model_config.get("num_leaves", 31)),
            "min_child_samples": int(model_config.get("min_child_samples", 20)),
            "subsample": float(model_config.get("subsample", 0.9)),
            "colsample_bytree": float(model_config.get("colsample_bytree", 0.9)),
            "reg_lambda": float(model_config.get("reg_lambda", 1.0)),
            "random_state": seed,
        }

    def _build_estimator(self) -> Any:
        params = self._params()
        if self.model_type == "lightgbm":
            lgb = _import_lightgbm()
            if lgb is None:
                raise ImportError("lightgbm is not installed")
            return lgb.LGBMRegressor(
                objective="regression",
                n_estimators=params["n_estimators"],
                learning_rate=params["learning_rate"],
                num_leaves=params["num_leaves"],
                min_child_samples=params["min_child_samples"],
                subsample=params["subsample"],
                colsample_bytree=params["colsample_bytree"],
                reg_lambda=params["reg_lambda"],
                random_state=params["random_state"],
                n_jobs=-1,
                verbose=-1,
                deterministic=True,
                force_row_wise=True,
            )
        if self.model_type == "xgboost":
            xgb = _import_xgboost()
            if xgb is None:
                raise ImportError("xgboost is not installed")
            return xgb.XGBRegressor(
                objective="reg:squarederror",
                tree_method="hist",
                n_estimators=params["n_estimators"],
                learning_rate=params["learning_rate"],
                max_depth=6,
                min_child_weight=params["min_child_samples"] / 10.0,
                subsample=params["subsample"],
                colsample_bytree=params["colsample_bytree"],
                reg_lambda=params["reg_lambda"],
                random_state=params["random_state"],
                n_jobs=-1,
            )
        if self.model_type == "sklearn_hist":
            estimator = _import_sklearn()
            if estimator is None:
                raise ImportError("scikit-learn is not installed")
            return estimator(
                max_iter=params["n_estimators"],
                learning_rate=params["learning_rate"],
                max_leaf_nodes=params["num_leaves"],
                min_samples_leaf=params["min_child_samples"],
                l2_regularization=params["reg_lambda"],
                random_state=params["random_state"],
            )
        raise ValueError(f"Unsupported model type: {self.model_type}")

    def _transform_target(self, y: np.ndarray) -> np.ndarray:
        transform = str((self.config.get("forecasting", {}) or {}).get("target_transform", "log1p"))
        if transform == "log1p":
            return np.log1p(np.clip(y, 0.0, None))
        if transform in {"none", "identity"}:
            return y
        raise ValueError(f"Unsupported target transform: {transform}")

    def _inverse_target(self, values: np.ndarray) -> np.ndarray:
        transform = str((self.config.get("forecasting", {}) or {}).get("target_transform", "log1p"))
        if transform == "log1p":
            return np.expm1(values)
        return values

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        feature_names: list[str] | None = None,
        training_metadata: Mapping[str, Any] | None = None,
    ) -> "GlobalDemandModel":
        self.feature_names_ = list(feature_names or X.columns)
        X_fit = X.loc[:, self.feature_names_]
        y_values = np.asarray(y, dtype=float)
        self.model = self._build_estimator()
        self.model.fit(X_fit, self._transform_target(y_values))
        self.metadata = {
            "mode": self.mode,
            "model_type": self.model_type,
            "label": self.label,
            "features": self.feature_names_,
            "target_transform": (self.config.get("forecasting", {}) or {}).get("target_transform", "log1p"),
            "params": self._params(),
            "training_rows": int(len(X_fit)),
            **dict(training_metadata or {}),
        }
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been fitted")
        raw = self.model.predict(X.loc[:, self.feature_names_])
        values = self._inverse_target(np.asarray(raw, dtype=float))
        clip = bool((self.config.get("forecasting", {}) or {}).get("clip_predictions_at_zero", True))
        if clip:
            values = np.clip(values, 0.0, None)
        return values

    def feature_importance(self) -> pd.DataFrame:
        if self.model is None:
            return pd.DataFrame(columns=["feature", "importance"])
        if self.model_type == "lightgbm":
            importance = self.model.booster_.feature_importance(importance_type="gain")
        elif self.model_type == "xgboost":
            importance = np.asarray(self.model.feature_importances_, dtype=float)
        else:
            return pd.DataFrame(columns=["feature", "importance"])
        frame = pd.DataFrame({"feature": self.feature_names_, "importance": importance})
        total = float(frame["importance"].sum())
        frame["importance_share"] = frame["importance"] / total if total > 0 else 0.0
        return frame.sort_values("importance", ascending=False).reset_index(drop=True)

    def save(self, directory: str | Path) -> dict[str, Path]:
        target = resolve_path(directory)
        target.mkdir(parents=True, exist_ok=True)
        model_path = target / "model.joblib"
        joblib.dump(self.model, model_path)
        paths = {"model": model_path}
        if self.model_type == "lightgbm" and self.model is not None:
            booster_path = target / "model.txt"
            self.model.booster_.save_model(str(booster_path))
            paths["booster"] = booster_path
        metadata_path = write_json(self.metadata, target / "metadata.json")
        paths["metadata"] = metadata_path
        return paths

    @classmethod
    def load(cls, directory: str | Path, config: Mapping[str, Any]) -> "GlobalDemandModel":
        target = resolve_path(directory)
        metadata_path = target / "metadata.json"
        model_path = target / "model.joblib"
        if not metadata_path.exists() or not model_path.exists():
            raise FileNotFoundError(f"Model artefact not found in {target}")
        import json

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        instance = cls(
            mode=str(metadata.get("mode", "")),
            model_type=str(metadata.get("model_type", "lightgbm")),
            config=config,
            model=joblib.load(model_path),
            feature_names_=list(metadata.get("features", [])),
            metadata=metadata,
        )
        return instance


def build_model(mode: str, config: Mapping[str, Any], model_type: str | None = None) -> GlobalDemandModel:
    resolved = resolve_model_type(config, requested=model_type)
    return GlobalDemandModel(mode=mode, model_type=resolved, config=config)

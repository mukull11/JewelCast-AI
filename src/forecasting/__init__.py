"""JewelCast AI M3 forecasting engine.

The package keeps each dataset independent. A global model is trained per
dataset mode; features, validation boundaries and forecasts are never shared
across synthetic, UCI and M5.
"""

from .baselines import BASELINE_MODELS, forecast_baselines
from .data_builder import MODES, ForecastingPanel, load_forecasting_panel
from .evaluation import evaluate_predictions, metric_values, select_best_model
from .feature_engineering import (
    LAGS,
    ROLLING_WINDOWS,
    build_features,
    feature_columns,
)
from .forecasting import (
    forecast_ml,
    production_forecast,
    run_forecasting_pipeline,
)
from .models import GlobalDemandModel, build_model
from .validation import ChronologicalSplit, chronological_split, rolling_origins

__all__ = [
    "BASELINE_MODELS",
    "ChronologicalSplit",
    "ForecastingPanel",
    "GlobalDemandModel",
    "LAGS",
    "MODES",
    "ROLLING_WINDOWS",
    "build_features",
    "build_model",
    "chronological_split",
    "evaluate_predictions",
    "feature_columns",
    "forecast_baselines",
    "forecast_ml",
    "load_forecasting_panel",
    "metric_values",
    "production_forecast",
    "rolling_origins",
    "run_forecasting_pipeline",
    "select_best_model",
]

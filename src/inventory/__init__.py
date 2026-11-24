"""M4 inventory optimization engine.

Public entry points are intentionally kept small; the modules remain importable
directly for testing and scripts.
"""

from src.inventory.data_builder import InventoryData, load_inventory_data
from src.inventory.engine import RECOMMENDATION_COLUMNS, build_recommendations
from src.inventory.evaluation import policy_comparison, simulation_metrics
from src.inventory.forecasting_adapter import (
    ArtifactForecastProvider,
    ModelForecastProvider,
    SavedModelForecastProvider,
    TrailingMeanForecastProvider,
)
from src.inventory.scenarios import run_scenarios
from src.inventory.settings import inventory_settings
from src.inventory.simulation import SimulationResult, simulate_policy

__all__ = [
    "ArtifactForecastProvider",
    "InventoryData",
    "ModelForecastProvider",
    "RECOMMENDATION_COLUMNS",
    "SavedModelForecastProvider",
    "SimulationResult",
    "TrailingMeanForecastProvider",
    "build_recommendations",
    "inventory_settings",
    "load_inventory_data",
    "policy_comparison",
    "run_scenarios",
    "simulate_policy",
    "simulation_metrics",
]

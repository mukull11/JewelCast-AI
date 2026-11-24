"""Reorder-point calculation and transparent reorder statuses.

Reorder Point = Expected Lead-Time Demand + Safety Stock

Status mapping (documented in ``docs/inventory_optimization.md``):

- ``Healthy``: inventory position is at or above the reorder point.
- ``Reorder Required``: inventory position is below the reorder point, but the
  projected cover is long enough that a stockout is not yet imminent.
- ``At Risk``: reorder is required and stock is projected to run out before or
  shortly after replenishment arrives (High stockout risk).
- ``Critical``: an existing stockout, or a projected stockout within the
  configured critical window.
"""

from __future__ import annotations

REORDER_STATUSES = ("Healthy", "Reorder Required", "At Risk", "Critical")
STOCKOUT_RISK_LEVELS = ("Low", "Medium", "High", "Critical")


def reorder_point(lead_time_demand: float, safety_stock: float) -> float:
    """Return the reorder point."""
    return max(0.0, float(lead_time_demand or 0.0) + float(safety_stock or 0.0))


def reorder_required(inventory_position: float, reorder_point_value: float) -> bool:
    return float(inventory_position or 0.0) < float(reorder_point_value or 0.0)


def classify_reorder_status(
    inventory_position: float,
    reorder_point_value: float,
    stockout_risk: str,
) -> str:
    """Map inventory position and stockout risk to a reorder status."""
    risk = str(stockout_risk)
    if risk == "Critical":
        return "Critical"
    if risk == "High":
        return "At Risk"
    if reorder_required(inventory_position, reorder_point_value):
        return "Reorder Required"
    if risk == "Medium":
        return "At Risk"
    return "Healthy"

"""Inventory-position arithmetic.

Inventory Position = Available Stock + Confirmed Incoming Stock - Committed Stock

Only components that genuinely exist in the dataset are used. The synthetic data
has start-of-day stock and scheduled purchase orders. ``committed_qty`` exists
only in weekly snapshots and is excluded from decisions by default because the
generator derived it from future demand.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def inventory_position(
    available_stock: float,
    incoming_stock: float = 0.0,
    committed_stock: float = 0.0,
) -> float:
    """Return the inventory position (never negative)."""
    position = float(available_stock or 0.0) + float(incoming_stock or 0.0) - float(committed_stock or 0.0)
    if not np.isfinite(position):
        return 0.0
    return max(0.0, position)


def add_inventory_position(
    frame: pd.DataFrame,
    stock_column: str = "current_stock",
    incoming_column: str = "incoming_stock",
    committed_column: str | None = None,
    output_column: str = "inventory_position",
) -> pd.DataFrame:
    """Add the inventory-position column to a SKU-level frame."""
    out = frame.copy()
    if stock_column not in out.columns:
        raise ValueError(f"missing stock column: {stock_column}")
    incoming = out[incoming_column] if incoming_column in out.columns else 0.0
    committed = out[committed_column] if committed_column and committed_column in out.columns else 0.0
    position = (
        pd.to_numeric(out[stock_column], errors="coerce").fillna(0.0)
        + pd.to_numeric(incoming, errors="coerce").fillna(0.0)
        - pd.to_numeric(committed, errors="coerce").fillna(0.0)
    )
    out[output_column] = position.clip(lower=0.0)
    return out


def incoming_stock(
    purchase_orders: pd.DataFrame,
    as_of: Any,
    sku: str | None = None,
) -> float | pd.Series:
    """Sum pending incoming purchase orders at ``as_of`` without double counting."""
    from src.inventory.data_builder import pending_purchase_orders

    pending = pending_purchase_orders(purchase_orders, as_of)
    if pending.empty:
        if sku is not None:
            return 0.0
        return pd.Series(dtype="float64")
    totals = pending.groupby("sku", observed=True)["quantity"].sum()
    if sku is not None:
        return float(totals.get(str(sku), 0.0))
    return totals

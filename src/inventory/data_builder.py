"""Inventory and purchase-order preparation for the M4 engine.

Only the synthetic dataset has inventory records. This module never fabricates
inventory for UCI or M5. It aligns three genuinely available inputs:

- ``daily_sales.parquet``: start-of-day ``available_stock`` per SKU-day.
- ``inventory_snapshots.parquet``: weekly end-of-day snapshots with an
  ``incoming_po_qty`` and a simulated ``committed_qty``.
- ``purchase_orders.parquet``: simulated orders with order dates, expected
  arrival dates, quantities and supplier lead times.

Alignment rules (documented in ``docs/inventory_optimization.md``)
-------------------------------------------------------------------
- ``available_stock`` in the daily panel is the stock at the start of the day,
  after orders due that day have been received and before that day's sales.
- A decision at date *d* therefore uses start-of-day stock at *d* and a forecast
  origin of *d - 1*, so same-day sales are never used.
- A purchase order is pending at *d* when it is not cancelled, ``order_date <= d``
  and (``expected_arrival_date > d`` or its recorded status is still ``open``).
  Orders already due at *d* are included in start-of-day stock.
- Pending purchase orders are summed once per ``po_id``. Duplicate PO rows are
  de-duplicated before aggregation, so an incoming order is never counted twice.
- ``committed_qty`` only exists in weekly snapshots and was generated from the
  next seven days of demand. It is therefore context, not a leakage-safe input,
  and is excluded from decisions unless explicitly enabled.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.eda.loading import (
    load_daily,
    load_inventory_snapshots,
    load_purchase_orders,
    load_sku_master,
)
from src.inventory.settings import (
    as_timestamp,
    inventory_settings,
    negative_to_zero,
)

LOGGER = logging.getLogger(__name__)

SUPPORTED_INVENTORY_MODES = ("synthetic",)

CANCELLED_STATUSES = {"cancelled", "canceled", "void", "voided"}


@dataclass
class InventoryData:
    """Cleaned inventory inputs for one dataset mode."""

    mode: str
    daily: pd.DataFrame
    sku_master: pd.DataFrame
    purchase_orders: pd.DataFrame
    snapshots: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def skus(self) -> pd.Index:
        return pd.Index(sorted(self.daily["sku"].astype(str).unique()), name="sku")

    def lead_time_series(self, default_lead_time_days: int | None = None) -> pd.Series:
        return build_lead_time_series(
            self.sku_master,
            default_lead_time_days=default_lead_time_days,
        )[0]


def _clean_daily(daily: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = daily.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["sku"] = frame["sku"].astype(str)
    if frame["date"].isna().any():
        raise ValueError("daily inventory panel contains invalid dates")
    if not frame["available_stock"].notna().any():
        raise ValueError(
            f"[{mode}] available_stock is entirely null. Inventory optimization requires "
            "real or explicitly simulated inventory records; UCI/M5 are not supported."
        )

    duplicates = int(frame.duplicated(subset=["sku", "date"]).sum())
    if duplicates:
        LOGGER.warning("[%s] aggregating %d duplicate daily inventory rows", mode, duplicates)
        frame = (
            frame.sort_values(["sku", "date"], kind="mergesort")
            .groupby(["sku", "date"], as_index=False)
            .agg(
                product_name=("product_name", "first"),
                category=("category", "first"),
                units_sold=("units_sold", "sum"),
                available_stock=("available_stock", "last"),
                stockout_flag=("stockout_flag", "max"),
                data_source=("data_source", "first"),
            )
        )

    stock = pd.to_numeric(frame["available_stock"], errors="coerce")
    null_stock = int(stock.isna().sum())
    negative_stock = int((stock < 0).sum())
    frame["available_stock"] = stock.clip(lower=0.0)
    frame["units_sold"] = pd.to_numeric(frame["units_sold"], errors="coerce").fillna(0.0).clip(lower=0.0)
    frame = frame.sort_values(["sku", "date"], kind="mergesort").reset_index(drop=True)

    return frame, {
        "daily_rows": int(len(frame)),
        "daily_skus": int(frame["sku"].nunique()),
        "null_stock_rows": null_stock,
        "negative_stock_rows_clipped": negative_stock,
        "duplicate_daily_rows_aggregated": duplicates,
    }


def build_lead_time_series(
    sku_master: pd.DataFrame,
    default_lead_time_days: int | None = None,
) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    """Return per-SKU lead times, a ``lead_time_defaulted`` flag and metadata.

    Supplier lead times come from ``sku_master.supplier_lead_time_days`` in the
    synthetic dataset. A missing, non-numeric or non-positive value is replaced
    with the documented configured fallback and flagged; it is never silently
    treated as zero.
    """
    default = int(default_lead_time_days or 14)
    if default <= 0:
        raise ValueError("default_lead_time_days must be positive")
    if sku_master.empty or "sku" not in sku_master.columns:
        return pd.Series(dtype="float64"), pd.Series(dtype="bool"), {"missing_sku_master": True}

    master = sku_master.drop_duplicates(subset=["sku"], keep="last").copy()
    master["sku"] = master["sku"].astype(str)
    master = master.set_index("sku")
    if "supplier_lead_time_days" in master.columns:
        raw = pd.to_numeric(master["supplier_lead_time_days"], errors="coerce")
    else:
        raw = pd.Series(np.nan, index=master.index)
    invalid = raw.isna() | (raw <= 0)
    lead = raw.where(~invalid, float(default)).astype("float64")
    lead = lead.round().clip(lower=1)
    flags = invalid.astype(bool)
    metadata = {
        "lead_time_source": "sku_master.supplier_lead_time_days"
        if "supplier_lead_time_days" in master.columns
        else "configured fallback (no supplier_lead_time_days column)",
        "default_lead_time_days": default,
        "lead_time_missing_or_invalid": int(flags.sum()),
    }
    return lead, flags, metadata


def clean_purchase_orders(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Normalise purchase orders and de-duplicate them by ``po_id``."""
    if frame is None or frame.empty:
        columns = [
            "po_id",
            "sku",
            "order_date",
            "expected_arrival_date",
            "quantity",
            "status",
            "supplier_lead_time_days",
            "data_source",
        ]
        return pd.DataFrame(columns=columns), {"purchase_order_rows": 0}

    orders = frame.copy()
    required = {"po_id", "sku", "order_date", "expected_arrival_date", "quantity"}
    missing = required.difference(orders.columns)
    if missing:
        raise ValueError(f"purchase_orders is missing columns: {sorted(missing)}")

    orders["sku"] = orders["sku"].astype(str)
    orders["po_id"] = orders["po_id"].astype(str)
    orders["order_date"] = pd.to_datetime(orders["order_date"], errors="coerce").dt.normalize()
    orders["expected_arrival_date"] = pd.to_datetime(
        orders["expected_arrival_date"], errors="coerce"
    ).dt.normalize()
    orders["quantity"] = pd.to_numeric(orders["quantity"], errors="coerce")
    if "status" not in orders.columns:
        orders["status"] = "open"
    orders["status"] = orders["status"].astype(str).str.strip().str.lower()

    invalid_dates = int(orders["order_date"].isna().sum() + orders["expected_arrival_date"].isna().sum())
    negative_quantity = int((orders["quantity"] < 0).sum())
    invalid_quantity = int(orders["quantity"].isna().sum())
    cancelled = int(orders["status"].isin(CANCELLED_STATUSES).sum())

    orders = orders.loc[orders["quantity"].notna() & (orders["quantity"] > 0)].copy()
    orders = orders.loc[~orders["status"].isin(CANCELLED_STATUSES)].copy()
    orders = orders.loc[orders["order_date"].notna() & orders["expected_arrival_date"].notna()]

    duplicates = int(orders.duplicated(subset=["po_id"]).sum())
    if duplicates:
        LOGGER.warning("dropping %d duplicate purchase-order rows by po_id", duplicates)
        orders = orders.sort_values(["po_id"]).drop_duplicates(subset=["po_id"], keep="last")

    orders = orders.sort_values(["sku", "order_date", "po_id"], kind="mergesort").reset_index(drop=True)
    return orders, {
        "purchase_order_rows": int(len(orders)),
        "purchase_order_skus": int(orders["sku"].nunique()) if len(orders) else 0,
        "invalid_po_dates": invalid_dates,
        "negative_po_quantities_dropped": negative_quantity,
        "null_po_quantities_dropped": invalid_quantity,
        "cancelled_purchase_orders_dropped": cancelled,
        "duplicate_po_rows_dropped": duplicates,
    }


def clean_snapshots(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """De-duplicate weekly inventory snapshots and clip invalid quantities."""
    if frame is None or frame.empty:
        return pd.DataFrame(), {"inventory_snapshot_rows": 0}
    snapshots = frame.copy()
    snapshots["snapshot_date"] = pd.to_datetime(snapshots["snapshot_date"], errors="coerce").dt.normalize()
    snapshots["sku"] = snapshots["sku"].astype(str)
    duplicates = int(snapshots.duplicated(subset=["sku", "snapshot_date"]).sum())
    snapshots = (
        snapshots.sort_values(["sku", "snapshot_date"], kind="mergesort")
        .drop_duplicates(subset=["sku", "snapshot_date"], keep="last")
        .reset_index(drop=True)
    )
    for column in ("available_stock", "incoming_po_qty", "committed_qty"):
        if column in snapshots.columns:
            snapshots[column] = (
                pd.to_numeric(snapshots[column], errors="coerce").fillna(0.0).clip(lower=0.0)
            )
    return snapshots, {
        "inventory_snapshot_rows": int(len(snapshots)),
        "inventory_snapshot_skus": int(snapshots["sku"].nunique()) if len(snapshots) else 0,
        "duplicate_snapshot_rows_dropped": duplicates,
    }


def load_inventory_data(
    mode: str,
    config: Mapping[str, Any],
    processed_dir: str | Path | None = None,
    sku_limit: int | None = None,
) -> InventoryData:
    """Load and validate the synthetic inventory inputs."""
    if mode not in SUPPORTED_INVENTORY_MODES:
        raise ValueError(
            f"Inventory optimization is currently supported only for the synthetic dataset "
            f"({SUPPORTED_INVENTORY_MODES}); {mode!r} has no inventory or purchase-order records."
        )

    settings = inventory_settings(config)
    daily = load_daily(mode, config, processed_dir=processed_dir, sku_limit=sku_limit)
    sku_master = load_sku_master(mode, config, processed_dir=processed_dir)
    purchase_orders = load_purchase_orders(config, processed_dir=processed_dir)
    snapshots = load_inventory_snapshots(config, processed_dir=processed_dir)

    if purchase_orders.empty:
        raise ValueError(
            "purchase_orders.parquet is missing or empty. Rebuild the synthetic dataset with "
            "scripts/build_datasets.py --modes synthetic."
        )

    daily, daily_meta = _clean_daily(daily, mode)
    orders, orders_meta = clean_purchase_orders(purchase_orders)
    snapshots, snapshot_meta = clean_snapshots(snapshots)

    requested_skus = set(daily["sku"].unique())
    orders = orders.loc[orders["sku"].isin(requested_skus)].copy()
    snapshots = snapshots.loc[snapshots["sku"].isin(requested_skus)].copy() if not snapshots.empty else snapshots
    if not sku_master.empty:
        sku_master = sku_master.loc[sku_master["sku"].astype(str).isin(requested_skus)].copy()

    lead, lead_flags, lead_meta = build_lead_time_series(
        sku_master, default_lead_time_days=int(settings["default_lead_time_days"])
    )
    metadata = {
        "mode": mode,
        "provenance": "Fully simulated Indian jewellery data. Not real company data.",
        "inventory_source": "weekly snapshots and daily start-of-day available_stock (synthetic)",
        "purchase_order_alignment": (
            "pending at d when not cancelled, order_date <= d and "
            "(expected_arrival_date > d or status == 'open'); summed once per po_id"
        ),
        "committed_stock_used_in_decisions": bool(settings.get("use_committed_stock", False)),
        "committed_stock_note": (
            "committed_qty exists only in weekly snapshots and was generated from the following "
            "7 days of demand; it is context, not a leakage-safe decision input"
        ),
        "date_min": str(daily["date"].min().date()),
        "date_max": str(daily["date"].max().date()),
        "skus": int(daily["sku"].nunique()),
        "lead_times": {
            "min": float(lead.min()) if len(lead) else None,
            "max": float(lead.max()) if len(lead) else None,
            "mean": float(lead.mean()) if len(lead) else None,
            **lead_meta,
        },
        **daily_meta,
        **orders_meta,
        **snapshot_meta,
    }
    LOGGER.info(
        "[%s] inventory inputs: days=%s skus=%s pos=%s snapshots=%s",
        mode,
        metadata["daily_rows"],
        metadata["skus"],
        metadata["purchase_order_rows"],
        metadata["inventory_snapshot_rows"],
    )
    return InventoryData(
        mode=mode,
        daily=daily,
        sku_master=sku_master,
        purchase_orders=orders,
        snapshots=snapshots,
        metadata=metadata,
    )


def pending_purchase_orders(purchase_orders: pd.DataFrame, as_of: Any) -> pd.DataFrame:
    """Return purchase orders still pending at ``as_of`` without double counting."""
    if purchase_orders is None or purchase_orders.empty:
        return purchase_orders.copy() if purchase_orders is not None else pd.DataFrame()
    as_of = as_timestamp(as_of)
    orders = purchase_orders
    mask = (
        (orders["order_date"] <= as_of)
        & (
            (orders["expected_arrival_date"] > as_of)
            | (orders["status"].astype(str).str.lower() == "open")
        )
        & (~orders["status"].astype(str).str.lower().isin(CANCELLED_STATUSES))
    )
    pending = orders.loc[mask].copy()
    pending = pending.drop_duplicates(subset=["po_id"], keep="last")
    pending["is_overdue"] = pending["expected_arrival_date"] <= as_of
    return pending.sort_values(["sku", "expected_arrival_date", "po_id"], kind="mergesort").reset_index(drop=True)


def incoming_purchase_order_summary(
    purchase_orders: pd.DataFrame,
    as_of: Any,
    lead_time_by_sku: pd.Series | Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Aggregate pending purchase orders per SKU.

    Columns: ``incoming_stock`` (all pending), ``incoming_due_within_lead_time``
    and ``incoming_overdue_qty``. Every ``po_id`` contributes at most once.
    """
    pending = pending_purchase_orders(purchase_orders, as_of)
    if pending.empty:
        return pd.DataFrame(
            columns=["sku", "incoming_stock", "incoming_due_within_lead_time", "incoming_overdue_qty", "incoming_po_count"]
        )
    lead = pd.Series(lead_time_by_sku, dtype="float64") if lead_time_by_sku is not None else pd.Series(dtype="float64")
    as_of = as_timestamp(as_of)
    pending["lead_time_days"] = pending["sku"].map(lead)
    pending["lead_time_days"] = pd.to_numeric(pending["lead_time_days"], errors="coerce").fillna(
        float(lead.median()) if len(lead.dropna()) else 14.0
    )
    pending["due_within_lead_time"] = pending["expected_arrival_date"] <= (
        as_of + pd.to_timedelta(pending["lead_time_days"], unit="D")
    )
    grouped = pending.groupby("sku", observed=True)
    result = pd.DataFrame(
        {
            "incoming_stock": grouped["quantity"].sum(),
            "incoming_po_count": grouped["po_id"].nunique(),
            "incoming_due_within_lead_time": pending.loc[pending["due_within_lead_time"]]
            .groupby("sku", observed=True)["quantity"]
            .sum(),
            "incoming_overdue_qty": pending.loc[pending["is_overdue"]]
            .groupby("sku", observed=True)["quantity"]
            .sum(),
        }
    ).reset_index()
    for column in ("incoming_stock", "incoming_due_within_lead_time", "incoming_overdue_qty"):
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0.0)
    return result


def _sku_launch_dates(data: InventoryData) -> pd.Series:
    if not data.sku_master.empty and "launch_date" in data.sku_master.columns:
        launch = pd.to_datetime(data.sku_master.set_index("sku")["launch_date"], errors="coerce")
        launch = launch[~launch.isna()]
        if len(launch):
            return launch
    return data.daily.groupby("sku", observed=True)["date"].min()


def build_inventory_state(
    data: InventoryData,
    decision_date: Any,
    use_committed_stock: bool = False,
    default_lead_time_days: int = 14,
) -> pd.DataFrame:
    """Build the per-SKU inventory state available at the start of a date.

    The output contains current stock, pending incoming stock and (optionally)
    snapshot committed stock. Missing stock is never silently treated as an
    ordinary zero: the affected row is flagged ``stock_missing``.
    """
    decision_date = as_timestamp(decision_date)
    lead, lead_flags, _ = build_lead_time_series(data.sku_master, default_lead_time_days)
    launch = _sku_launch_dates(data)

    rows = data.daily.loc[data.daily["date"] <= decision_date].copy()
    exact = rows.loc[rows["date"] == decision_date].set_index("sku")
    non_null = rows.loc[rows["available_stock"].notna()]
    latest = (
        non_null.sort_values(["sku", "date"], kind="mergesort")
        .groupby("sku", observed=True)
        .tail(1)
        .set_index("sku")
    )

    universe = pd.Index(
        sorted(
            sku
            for sku in rows["sku"].unique()
            if pd.Timestamp(launch.get(str(sku), pd.Timestamp.min)) <= decision_date
        ),
        name="sku",
    )
    state = pd.DataFrame(index=universe)
    exact_stock = exact.reindex(universe)["available_stock"]
    latest_stock = latest.reindex(universe)["available_stock"]
    state["current_stock"] = exact_stock.where(exact_stock.notna(), latest_stock)
    state["stock_date"] = pd.NaT
    state.loc[exact_stock.notna(), "stock_date"] = decision_date
    stale_mask = state["stock_date"].isna() & latest_stock.notna()
    state.loc[stale_mask, "stock_date"] = latest.reindex(universe).loc[stale_mask, "date"]
    state["stock_missing"] = state["current_stock"].isna()
    state["stock_stale"] = stale_mask
    state["current_stock"] = state["current_stock"].clip(lower=0.0).fillna(0.0)

    category = pd.Series(dtype="object")
    if not data.sku_master.empty:
        master = data.sku_master.drop_duplicates(subset=["sku"], keep="last").set_index("sku")
        category = master["category"] if "category" in master.columns else category
    daily_meta = data.daily.drop_duplicates(subset=["sku"], keep="last").set_index("sku")
    state["category"] = daily_meta["category"].reindex(universe).where(
        daily_meta["category"].reindex(universe).notna(), category.reindex(universe)
    )
    state["lead_time_days"] = lead.reindex(universe).fillna(float(default_lead_time_days)).round().clip(lower=1)
    state["lead_time_defaulted"] = lead_flags.reindex(universe).fillna(True).astype(bool)

    incoming = incoming_purchase_order_summary(data.purchase_orders, decision_date, lead)
    state = state.reset_index().merge(incoming, on="sku", how="left")
    for column in ("incoming_stock", "incoming_due_within_lead_time", "incoming_overdue_qty", "incoming_po_count"):
        if column not in state.columns:
            state[column] = 0.0
        state[column] = pd.to_numeric(state[column], errors="coerce").fillna(0.0)
    state["incoming_po_count"] = state["incoming_po_count"].astype(int)

    committed = pd.Series(0.0, index=universe)
    committed_source = pd.Series("unavailable", index=universe, dtype="object")
    if use_committed_stock and not data.snapshots.empty:
        snapshots = data.snapshots.loc[data.snapshots["snapshot_date"] <= decision_date]
        snapshots = (
            snapshots.sort_values(["sku", "snapshot_date"], kind="mergesort")
            .groupby("sku", observed=True)
            .tail(1)
            .set_index("sku")
        )
        committed = snapshots["committed_qty"].reindex(universe).fillna(0.0)
        committed_source = pd.Series("weekly_snapshot", index=universe, dtype="object")
    state["committed_stock"] = committed.reindex(state["sku"]).fillna(0.0).to_numpy()
    state["committed_stock_source"] = committed_source.reindex(state["sku"]).fillna("unavailable").to_numpy()
    state["committed_stock_used"] = bool(use_committed_stock)
    state["inventory_position"] = (
        state["current_stock"] + state["incoming_stock"] - state["committed_stock"]
    ).clip(lower=0.0)

    flags = []
    for _, row in state.iterrows():
        items = []
        if row["stock_missing"]:
            items.append("stock_missing")
        if row["stock_stale"]:
            items.append("stock_from_earlier_date")
        if row["lead_time_defaulted"]:
            items.append("lead_time_defaulted")
        if row["incoming_overdue_qty"] > 0:
            items.append("overdue_purchase_order_pending")
        flags.append(";".join(items))
    state["state_flags"] = flags
    state["decision_date"] = decision_date
    return state.sort_values("sku").reset_index(drop=True)


def daily_demand_history(
    daily: pd.DataFrame,
    origin: Any,
    window_days: int,
    skus: pd.Index | None = None,
) -> pd.DataFrame:
    """Return trailing daily units up to ``origin`` for every SKU.

    Each row is one SKU with ``history`` (a float array) and ``history_days``.
    Only dates strictly at or before the origin are used.
    """
    origin = as_timestamp(origin)
    window = int(window_days)
    if window <= 0:
        raise ValueError("window_days must be positive")
    history = daily.loc[daily["date"] <= origin, ["sku", "date", "units_sold"]].copy()
    history = history.sort_values(["sku", "date"], kind="mergesort")
    history = history.groupby("sku", observed=True).tail(window)
    records: list[dict[str, Any]] = []
    for sku, group in history.groupby("sku", sort=True, observed=True):
        values = pd.to_numeric(group["units_sold"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        records.append({"sku": str(sku), "history": values, "history_days": int(len(values))})
    if records:
        frame = pd.DataFrame(records)
    else:
        frame = pd.DataFrame(columns=["sku", "history", "history_days"])
    if skus is not None:
        frame = pd.DataFrame({"sku": list(skus)}).merge(frame, on="sku", how="left")
        frame["history"] = frame["history"].apply(lambda value: value if isinstance(value, np.ndarray) else np.array([], dtype=float))
        frame["history_days"] = pd.to_numeric(frame["history_days"], errors="coerce").fillna(0).astype(int)
    return frame

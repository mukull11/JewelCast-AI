"""Lead-time demand from the M3 daily forecast sequence.

The preferred calculation sums the daily forecast over the supplier lead time
rather than multiplying a single average by the lead time. This preserves
weekday, festival and seasonal shape in the forecast.

If the available forecast horizon is shorter than the requested lead time, the
shortfall is filled from the mean of the available *forecast* days (never actual
future demand) and the result is flagged. If a SKU has no forecast at all, a
documented causal fallback (trailing average demand) is used and flagged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class HorizonDemand:
    units: float
    days_requested: int
    days_forecast: int
    days_extrapolated: int
    daily_mean: float
    fallback_used: bool
    flags: list[str] = field(default_factory=list)

    @property
    def flag_string(self) -> str:
        return ";".join(self.flags)


def _clean_forecast_values(values: object) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return np.clip(array, 0.0, None)


def sum_demand_over_horizon(
    forecast_values: object,
    days: int,
    fallback_daily: float,
    horizon_label: str = "lead_time",
) -> HorizonDemand:
    """Sum forecast demand over ``days`` with an explicitly flagged fallback."""
    days = int(round(float(days)))
    if days <= 0:
        return HorizonDemand(
            units=0.0,
            days_requested=max(0, days),
            days_forecast=0,
            days_extrapolated=0,
            daily_mean=0.0,
            fallback_used=False,
            flags=[f"invalid_{horizon_label}_days"],
        )
    values = _clean_forecast_values(forecast_values)
    fallback = max(0.0, float(fallback_daily or 0.0))
    flags: list[str] = []
    if len(values) == 0:
        flags.append("forecast_missing_fallback")
        daily_mean = fallback
        return HorizonDemand(
            units=fallback * days,
            days_requested=days,
            days_forecast=0,
            days_extrapolated=days,
            daily_mean=daily_mean,
            fallback_used=True,
            flags=flags,
        )

    used = values[:days]
    daily_mean = float(used.mean())
    units = float(used.sum())
    extrapolated = 0
    if len(used) < days:
        extrapolated = days - len(used)
        units += daily_mean * extrapolated
        flags.append(f"{horizon_label}_exceeds_forecast_horizon")
    return HorizonDemand(
        units=max(0.0, units),
        days_requested=days,
        days_forecast=int(len(used)),
        days_extrapolated=int(extrapolated),
        daily_mean=daily_mean,
        fallback_used=False,
        flags=flags,
    )


def lead_time_demand(
    forecast_daily: object,
    lead_time_days: float,
    fallback_daily: float,
) -> HorizonDemand:
    """Expected demand during supplier lead time."""
    return sum_demand_over_horizon(
        forecast_daily,
        lead_time_days,
        fallback_daily,
        horizon_label="lead_time",
    )


def protection_period_demand(
    forecast_daily: object,
    protection_days: float,
    fallback_daily: float,
) -> HorizonDemand:
    """Expected demand during the protection period (lead time + review)."""
    return sum_demand_over_horizon(
        forecast_daily,
        protection_days,
        fallback_daily,
        horizon_label="protection_period",
    )


def forecast_series_for_sku(
    forecast: pd.DataFrame,
    sku: str,
    start_date: pd.Timestamp,
) -> pd.Series:
    """Return a sorted daily forecast Series for one SKU from ``start_date``."""
    if forecast is None or forecast.empty:
        return pd.Series(dtype="float64")
    view = forecast.loc[forecast["sku"].astype(str) == str(sku), ["date", "predicted_units"]].copy()
    if view.empty:
        return pd.Series(dtype="float64")
    view["date"] = pd.to_datetime(view["date"]).dt.normalize()
    view = view.loc[view["date"] >= pd.Timestamp(start_date).normalize()]
    view = view.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    return pd.to_numeric(view.set_index("date")["predicted_units"], errors="coerce").fillna(0.0).clip(lower=0.0)

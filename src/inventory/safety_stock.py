"""Safety-stock estimation with documented assumptions and fallbacks.

Default method (``demand_volatility``)
--------------------------------------
sigma(lead-time demand) = std(trailing daily units) * sqrt(lead time)
Safety stock = z(service level) * sigma(lead-time demand)

This is the conventional independent-and-identically-distributed daily demand
formula. It assumes stationary daily demand over the lead time, which is only an
approximation for fashion jewellery with festivals, promotions and lifecycle
effects. Observed synthetic sales are also stockout-censored, which can
understate true demand variability.

Alternative methods
-------------------
- ``forecast_error``: std of historical lead-time forecast errors, if supplied.
- ``empirical``: a historical percentile of rolling lead-time demand minus the
  mean, which makes fewer distributional assumptions but needs more history.

A configured service level is a *target*, not proof that the service level is
achieved. SKUs with too little history use a category-level daily volatility
fallback and are flagged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import NormalDist

import numpy as np


@dataclass
class SafetyStockResult:
    safety_stock: float
    z_value: float
    sigma_lead_time_demand: float
    sigma_daily: float
    method: str
    fallback_used: bool
    flags: list[str] = field(default_factory=list)

    @property
    def flag_string(self) -> str:
        return ";".join(self.flags)


def service_level_z(service_level: float) -> float:
    """Return the standard-normal service factor for a target service level."""
    level = float(service_level)
    if not (0.0 < level < 1.0):
        raise ValueError("service_level must be strictly between 0 and 1")
    return float(NormalDist().inv_cdf(level))


def _daily_sigma(values: np.ndarray, min_history: int) -> tuple[float | None, int]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if len(clean) < max(2, int(min_history)):
        return None, int(len(clean))
    sigma = float(np.std(clean, ddof=1))
    if not np.isfinite(sigma):
        return None, int(len(clean))
    return sigma, int(len(clean))


def estimate_lead_time_demand_sigma(
    history: np.ndarray,
    lead_time_days: float,
    method: str = "demand_volatility",
    service_level: float = 0.95,
    min_history: int = 14,
    fallback_sigma_daily: float = 0.0,
    forecast_error_std_daily: float | None = None,
) -> tuple[float, float, bool, list[str]]:
    """Return ``(sigma_lead_time_demand, sigma_daily, fallback_used, flags)``."""
    lead = max(1.0, float(lead_time_days))
    values = np.asarray(history, dtype=float)
    flags: list[str] = []
    fallback = max(0.0, float(fallback_sigma_daily or 0.0))
    sigma_daily, n = _daily_sigma(values, min_history)

    if method == "forecast_error":
        if forecast_error_std_daily is None or not np.isfinite(float(forecast_error_std_daily)):
            flags.append("forecast_error_unavailable_fallback_to_demand_volatility")
            method = "demand_volatility"
            if sigma_daily is None:
                sigma_daily = fallback
                flags.append("short_history_category_volatility_fallback")
            return sigma_daily * np.sqrt(lead), sigma_daily, bool(flags), flags
        sigma_daily = max(0.0, float(forecast_error_std_daily))
        return sigma_daily * np.sqrt(lead), sigma_daily, False, flags

    if method == "empirical":
        clean = values[np.isfinite(values)]
        if len(clean) >= max(2 * int(round(lead)), int(min_history) + int(round(lead)), 10):
            length = int(round(lead))
            rolling = np.convolve(clean, np.ones(length), mode="valid")
            percentile = float(np.percentile(rolling, float(service_level) * 100.0))
            mean = float(rolling.mean())
            sigma_ltd = max(0.0, percentile - mean)
            sigma_daily = sigma_ltd / np.sqrt(lead)
            return sigma_ltd, sigma_daily, False, flags + ["empirical_percentile_method"]
        flags.append("insufficient_history_for_empirical_fallback_to_demand_volatility")
        if sigma_daily is None:
            sigma_daily = fallback
            flags.append("short_history_category_volatility_fallback")
        return sigma_daily * np.sqrt(lead), sigma_daily, bool(flags), flags

    if method != "demand_volatility":
        raise ValueError(f"Unsupported safety-stock method: {method}")

    if sigma_daily is None:
        sigma_daily = fallback
        flags.append("short_history_category_volatility_fallback")
        return sigma_daily * np.sqrt(lead), sigma_daily, True, flags
    if n < int(min_history):
        flags.append("below_min_history")
    return sigma_daily * np.sqrt(lead), sigma_daily, False, flags


def safety_stock(
    service_level: float,
    sigma_lead_time_demand: float,
) -> tuple[float, float]:
    """Return ``(safety_stock, z_value)`` using z * sigma_LTD."""
    z = service_level_z(service_level)
    sigma = max(0.0, float(sigma_lead_time_demand or 0.0))
    return z * sigma, z


def category_volatility_fallback(
    daily_history: dict[str, np.ndarray],
    categories: dict[str, str],
    min_history: int = 14,
) -> tuple[dict[str, float], dict[str, float]]:
    """Estimate SKU and category daily volatility fallbacks.

    Returns ``(sku_sigma, category_sigma)``. SKUs with enough history get their
    own standard deviation; categories use the median of member SKU sigmas.
    """
    sku_sigma = {}
    for sku, values in daily_history.items():
        sigma, _ = _daily_sigma(values, min_history)
        if sigma is not None:
            sku_sigma[str(sku)] = sigma
    category_values: dict[str, list[float]] = {}
    for sku, sigma in sku_sigma.items():
        category = str(categories.get(sku, "Unknown"))
        category_values.setdefault(category, []).append(sigma)
    category_sigma = {
        category: float(np.median(values)) for category, values in category_values.items() if values
    }
    overall = float(np.median(list(sku_sigma.values()))) if sku_sigma else 0.0
    category_sigma.setdefault("__overall__", overall)
    return sku_sigma, category_sigma

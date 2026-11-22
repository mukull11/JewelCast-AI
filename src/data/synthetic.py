"""Synthetic Indian fashion-jewellery business dataset.

Everything produced here is explicitly simulated. It is never presented as real
company data. The generator is fully reproducible from a fixed seed and writes a
configuration file recording every assumption.

Generated artefacts
-------------------
- products.parquet            SKU master (attributes, price, lead time)
- daily_sales.parquet         SKU-day observed sales, inventory and flags
- inventory_snapshots.parquet weekly stock positions
- purchase_orders.parquet     simulated replenishment orders
- festivals.parquet           expanded Indian festival / wedding-season flags
- festival_calendar.csv       festival assumptions in human-readable form
- synthetic_config.json       seed and assumptions used
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.data.festivals import build_festival_calendar, build_wedding_calendar, expand_daily_event_flags
from src.utils.io import save_parquet, write_json
from src.utils.paths import resolve_path

CATEGORY_SPECS: dict[str, dict[str, Any]] = {
    "Earrings": {"code": "EAR", "price": (249, 1499), "base_demand": 4.2},
    "Necklaces": {"code": "NEC", "price": (499, 3999), "base_demand": 2.3},
    "Rings": {"code": "RIN", "price": (399, 2999), "base_demand": 2.1},
    "Bracelets": {"code": "BRA", "price": (399, 2499), "base_demand": 2.0},
    "Bangles": {"code": "BAN", "price": (299, 1999), "base_demand": 2.6},
    "Anklets": {"code": "ANK", "price": (199, 999), "base_demand": 2.2},
    "Pendants": {"code": "PEN", "price": (299, 1999), "base_demand": 2.0},
    "Jewellery Sets": {"code": "SET", "price": (999, 4999), "base_demand": 1.2},
}

MATERIALS = (
    "Gold Plated",
    "Silver",
    "Oxidised",
    "Kundan",
    "Pearl",
    "American Diamond",
    "Brass",
    "Alloy",
)
DESIGN_TYPES = (
    "Floral",
    "Minimal",
    "Traditional",
    "Contemporary",
    "Temple",
    "Geometric",
    "Vintage",
    "Boho",
    "Classic",
    "Fusion",
)

WEEKDAY_FACTORS = np.array([0.92, 0.92, 0.96, 1.02, 1.16, 1.28, 1.18])
MONTH_FACTORS = np.array([1.00, 1.12, 0.98, 1.02, 0.94, 0.88, 0.92, 1.06, 1.00, 1.16, 1.30, 1.20])
SERVICE_Z = {0.90: 1.2816, 0.95: 1.6449, 0.975: 1.9600, 0.99: 2.3263}


@dataclass
class SyntheticResult:
    products: pd.DataFrame
    daily_sales: pd.DataFrame
    inventory_snapshots: pd.DataFrame
    purchase_orders: pd.DataFrame
    festivals: pd.DataFrame
    settings: dict

    def save(self, output_dir: str | Path) -> dict[str, Path]:
        output = resolve_path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        paths = {
            "products": save_parquet(self.products, output / "products.parquet"),
            "daily_sales": save_parquet(self.daily_sales, output / "daily_sales.parquet"),
            "inventory_snapshots": save_parquet(self.inventory_snapshots, output / "inventory_snapshots.parquet"),
            "purchase_orders": save_parquet(self.purchase_orders, output / "purchase_orders.parquet"),
            "festivals": save_parquet(self.festivals, output / "festivals.parquet"),
        }
        self.festivals.to_csv(output / "festival_calendar.csv", index=False)
        write_json(self.settings, output / "synthetic_config.json")
        return paths


def _merged_settings(config: Mapping[str, Any], overrides: Mapping[str, Any] | None) -> dict:
    settings = copy.deepcopy(dict(config.get("data", {}).get("synthetic", {}) or {}))
    if overrides:
        settings.update(dict(overrides))
    settings.setdefault("seed", int(config.get("project", {}).get("seed", 42)))
    settings["n_skus"] = int(settings.get("n_skus", 300))
    settings["start_date"] = pd.Timestamp(settings.get("start_date", "2023-01-01")).normalize()
    settings["end_date"] = pd.Timestamp(settings.get("end_date", "2024-12-31")).normalize()
    if settings["end_date"] <= settings["start_date"]:
        raise ValueError("synthetic.end_date must be after synthetic.start_date")
    return settings


def _build_products(settings: dict, rng: np.random.Generator) -> pd.DataFrame:
    n_skus = settings["n_skus"]
    start = settings["start_date"]
    end = settings["end_date"]
    total_days = (end - start).days + 1
    cold_fraction = float(settings.get("cold_start_fraction", 0.12))

    categories = list(CATEGORY_SPECS)
    assignments: list[str] = []
    while len(assignments) < n_skus:
        assignments.extend(rng.permutation(categories).tolist())
    assignments = assignments[:n_skus]

    rows: list[dict[str, Any]] = []
    for index, category in enumerate(assignments):
        spec = CATEGORY_SPECS[category]
        low, high = spec["price"]
        list_price = int(round(rng.uniform(low, high) / 10.0) * 10 + 9)

        if rng.random() < cold_fraction:
            cold_window = min(120, max(30, total_days // 3))
            offset = total_days - int(rng.integers(30, cold_window + 1))
        else:
            offset = int(rng.integers(0, max(1, min(180, total_days // 2))))
        launch = start + pd.Timedelta(days=max(0, min(offset, total_days - 8)))

        rows.append(
            {
                "sku": f"JWL-{spec['code']}-{index + 1:04d}",
                "product_name": f"{rng.choice(DESIGN_TYPES)} {rng.choice(MATERIALS)} {category[:-1]} #{index + 1:03d}",
                "category": category,
                "material": str(rng.choice(MATERIALS)),
                "design_type": str(rng.choice(DESIGN_TYPES)),
                "list_price": list_price,
                "launch_date": launch,
                "supplier_lead_time_days": int(
                    rng.integers(
                        int(settings.get("lead_time_days_min", 7)),
                        int(settings.get("lead_time_days_max", 30)) + 1,
                    )
                ),
                "initial_inventory": 0,  # filled after demand simulation
                "popularity_score": float(np.clip(rng.lognormal(0.0, 0.55), 0.15, 6.0)),
                "festival_sensitivity": float(rng.uniform(0.4, 1.7)),
                "price_sensitivity": float(rng.uniform(0.15, 0.35)),
                "discount_elasticity": float(rng.uniform(1.2, 2.8)),
                "data_source": "synthetic",
            }
        )

    products = pd.DataFrame(rows)
    products["category_median_price"] = products.groupby("category")["list_price"].transform("median")
    return products


def _promotion_arrays(
    n_days: int,
    settings: dict,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    promotion = np.zeros(n_days, dtype=np.int8)
    discount = np.zeros(n_days, dtype=np.float64)
    probability = float(settings.get("promotion_probability", 0.06))
    max_discount = float(settings.get("max_discount_percentage", 40))
    expected_windows = max(1, int(round(n_days * probability / 5)))
    for _ in range(expected_windows):
        length = int(rng.integers(3, 11))
        start = int(rng.integers(0, max(1, n_days - length)))
        level = float(rng.uniform(10, max_discount))
        promotion[start : start + length] = 1
        discount[start : start + length] = level
    return promotion, discount


def _simulate_sku(
    product: pd.Series,
    settings: dict,
    event_flags: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    start = settings["start_date"]
    end = settings["end_date"]
    launch = pd.Timestamp(product["launch_date"]).normalize()
    dates = pd.date_range(launch, end, freq="D")
    n_days = len(dates)
    if n_days == 0:
        empty = pd.DataFrame()
        return empty, empty, empty, 0

    flags = event_flags.set_index("date").reindex(dates).reset_index()
    flags["festival_flag"] = flags["festival_flag"].fillna(0).astype("int8")
    flags["festival_strength"] = flags["festival_strength"].fillna(1.0)
    flags["wedding_season_flag"] = flags["wedding_season_flag"].fillna(0).astype("int8")
    flags["wedding_strength"] = flags["wedding_strength"].fillna(1.0)

    promotion, discount = _promotion_arrays(n_days, settings, rng)

    age = (dates - launch).days.to_numpy()
    ramp = np.clip(age / 21.0, 0.0, 1.0)
    decline = np.where(age < 365, 1.0, 0.72 + 0.28 * np.exp(-(age - 365) / 240.0))
    lifecycle = ramp * decline

    weekday = WEEKDAY_FACTORS[dates.dayofweek.to_numpy()]
    monthly = MONTH_FACTORS[dates.month.to_numpy() - 1]
    price_factor = (product["category_median_price"] / product["list_price"]) ** product["price_sensitivity"]

    sensitivity = product["festival_sensitivity"]
    festival_mult = 1.0 + (flags["festival_strength"].to_numpy() - 1.0) * sensitivity
    festival_mult = np.where(flags["festival_flag"].to_numpy() == 1, festival_mult, 1.0)
    wedding_mult = 1.0 + (flags["wedding_strength"].to_numpy() - 1.0) * sensitivity * 0.6
    wedding_mult = np.where(flags["wedding_season_flag"].to_numpy() == 1, wedding_mult, 1.0)
    discount_mult = 1.0 + product["discount_elasticity"] * discount / 100.0

    base = CATEGORY_SPECS[product["category"]]["base_demand"] * product["popularity_score"]
    lam = (
        base
        * weekday
        * monthly
        * price_factor
        * lifecycle
        * festival_mult
        * wedding_mult
        * discount_mult
    )
    lam = np.clip(lam, 0.0, 80.0)
    overdispersion = 6.0
    demand = rng.negative_binomial(overdispersion, overdispersion / (overdispersion + lam)).astype(np.int64)

    lead_time = int(product["supplier_lead_time_days"])
    mean_daily = float(np.mean(lam[: min(30, n_days)])) or 1.0
    std_daily = float(np.std(lam[: min(60, n_days)])) or mean_daily * 0.5
    service_level = float(settings.get("target_service_level", 0.95))
    z_value = SERVICE_Z.get(round(service_level, 3), 1.6449)
    safety_stock = z_value * std_daily * np.sqrt(lead_time)
    reorder_point = mean_daily * lead_time + safety_stock
    target_stock = reorder_point + mean_daily * lead_time

    initial_stock = int(max(5, round(mean_daily * rng.uniform(
        float(settings.get("initial_stock_days_min", 18)),
        float(settings.get("initial_stock_days_max", 55)),
    ))))

    stock = initial_stock
    open_orders: list[dict[str, Any]] = []
    sales_rows: list[dict[str, Any]] = []
    order_rows: list[dict[str, Any]] = []
    snapshot_rows: list[dict[str, Any]] = []
    order_counter = 0

    for index, day in enumerate(dates):
        due = [order for order in open_orders if order["expected_arrival_date"] <= day]
        for order in due:
            stock += order["quantity"]
            order["status"] = "received"
            open_orders.remove(order)

        open_quantity = int(sum(order["quantity"] for order in open_orders))
        inventory_position = stock + open_quantity
        if inventory_position < reorder_point and open_quantity == 0:
            quantity = int(round(max(0.0, target_stock - inventory_position)))
            if quantity > 0:
                order_counter += 1
                order = {
                    "po_id": f"PO-{product['sku']}-{order_counter:04d}",
                    "sku": product["sku"],
                    "order_date": day,
                    "expected_arrival_date": day + pd.Timedelta(days=lead_time),
                    "quantity": quantity,
                    "status": "open",
                    "supplier_lead_time_days": lead_time,
                    "data_source": "synthetic",
                }
                open_orders.append(order)
                order_rows.append(order)

        available = stock
        sold = int(min(demand[index], stock))
        stockout = int(demand[index] > stock)
        stock -= sold
        selling_price = round(float(product["list_price"]) * (1.0 - discount[index] / 100.0), 2)

        sales_rows.append(
            {
                "date": day,
                "sku": product["sku"],
                "product_name": product["product_name"],
                "category": product["category"],
                "units_sold": sold,
                "selling_price": selling_price,
                "discount_percentage": round(float(discount[index]), 2),
                "revenue": round(sold * selling_price, 2),
                "festival_flag": int(flags.loc[index, "festival_flag"]),
                "festival_name": flags.loc[index, "festival_name"],
                "promotion_flag": int(promotion[index]),
                "available_stock": available,
                "stockout_flag": stockout,
                "data_source": "synthetic",
            }
        )

        if day.dayofweek == 0:  # Monday snapshot
            future = demand[index + 1 : index + 8]
            committed = int(round(float(future.sum()) * 0.10)) if len(future) else 0
            snapshot_rows.append(
                {
                    "snapshot_date": day,
                    "sku": product["sku"],
                    "available_stock": int(stock),
                    "incoming_po_qty": int(sum(order["quantity"] for order in open_orders)),
                    "committed_qty": committed,
                    "data_source": "synthetic",
                }
            )

    for order in order_rows:
        if order["expected_arrival_date"] > end:
            order["status"] = "open"

    return pd.DataFrame(sales_rows), pd.DataFrame(snapshot_rows), pd.DataFrame(order_rows), initial_stock


def generate_synthetic_dataset(
    config: Mapping[str, Any],
    output_dir: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> SyntheticResult:
    """Generate the full synthetic jewellery dataset.

    Parameters
    ----------
    config:
        Parsed ``config.yaml`` mapping.
    output_dir:
        Where to write Parquet/CSV artefacts. Defaults to ``data/synthetic``.
    overrides:
        Optional overrides for ``data.synthetic`` settings (useful in tests).
    """
    settings = _merged_settings(config, overrides)
    rng = np.random.default_rng(int(settings["seed"]))

    products = _build_products(settings, rng)
    all_dates = pd.date_range(settings["start_date"], settings["end_date"], freq="D")
    event_flags = expand_daily_event_flags(all_dates, config)

    sales_frames: list[pd.DataFrame] = []
    snapshot_frames: list[pd.DataFrame] = []
    order_frames: list[pd.DataFrame] = []

    for index, product in products.iterrows():
        sales, snapshots, orders, initial_stock = _simulate_sku(product, settings, event_flags, rng)
        products.loc[index, "initial_inventory"] = initial_stock
        if not sales.empty:
            sales_frames.append(sales)
        if not snapshots.empty:
            snapshot_frames.append(snapshots)
        if not orders.empty:
            order_frames.append(orders)

    daily_sales = pd.concat(sales_frames, ignore_index=True)
    inventory_snapshots = pd.concat(snapshot_frames, ignore_index=True)
    purchase_orders = pd.concat(order_frames, ignore_index=True)

    end = settings["end_date"]
    products["product_age_days"] = (end - pd.to_datetime(products["launch_date"])).dt.days
    products["is_cold_start"] = products["product_age_days"] < 120

    festivals = build_festival_calendar(config)
    weddings = build_wedding_calendar(config)

    settings_out = {
        "dataset": "synthetic_indian_jewellery",
        "provenance": "Fully simulated. Not real company data.",
        "seed": int(settings["seed"]),
        "n_skus": int(settings["n_skus"]),
        "start_date": str(settings["start_date"].date()),
        "end_date": str(settings["end_date"].date()),
        "cold_start_fraction": float(settings.get("cold_start_fraction", 0.12)),
        "target_service_level": float(settings.get("target_service_level", 0.95)),
        "festivals": festivals.to_dict(orient="records"),
        "wedding_seasons": weddings.to_dict(orient="records"),
        "assumptions": [
            "Daily demand is a negative-binomial draw around a category/price/seasonality/lifecycle mean.",
            "Festival effects use the configured Indian festival calendar with SKU-specific sensitivity.",
            "Promotions are random windows; discount uplift follows a SKU-specific elasticity.",
            "Inventory is simulated with (s, S)-style reordering and lost sales when stock reaches zero.",
            "Committed quantities and purchase orders are simulated, not observed.",
        ],
    }

    result = SyntheticResult(
        products=products,
        daily_sales=daily_sales,
        inventory_snapshots=inventory_snapshots,
        purchase_orders=purchase_orders,
        festivals=festivals,
        settings=settings_out,
    )
    if output_dir is not None:
        result.save(output_dir)
    return result

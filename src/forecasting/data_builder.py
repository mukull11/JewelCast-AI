"""Build leakage-safe daily panels for the forecasting engine.

Inputs are the processed SKU-day Parquet files created in M0-M2. The builder
never regenerates source data and never combines modes. It normalises dates,
verifies that duplicate SKU-days are absent, completes any calendar gaps within
each series, and attaches the mode-specific calendar information that is known
at forecast time.

The primary target is ``units_sold``: observed historical sales. For the
synthetic dataset this is stockout-censored sales, not latent unconstrained
demand. That limitation is documented in ``docs/forecasting.md``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.data.festivals import expand_daily_event_flags, festival_flag_for_dates
from src.eda.loading import load_daily
from src.utils.paths import resolve_path

LOGGER = logging.getLogger(__name__)

MODES = ("synthetic", "uci", "m5")

PANEL_COLUMNS = (
    "date",
    "sku",
    "product_name",
    "category",
    "units_sold",
    "selling_price",
    "discount_percentage",
    "revenue",
    "festival_flag",
    "festival_name",
    "promotion_flag",
    "available_stock",
    "stockout_flag",
    "event_name",
    "date_imputed",
    "data_source",
)

EVENT_COLUMNS = (
    "festival_flag",
    "festival_name",
    "festival_strength",
    "wedding_season_flag",
    "wedding_strength",
    "event_name",
    "event_flag",
    "event_code",
)


@dataclass
class ForecastingPanel:
    """A single mode's daily panel plus calendar and encoding metadata."""

    mode: str
    frame: pd.DataFrame
    calendar: pd.DataFrame
    category_codes: dict[str, int]
    first_dates: pd.Series
    event_codes: dict[str, int]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(sorted(self.frame["date"].unique()))

    @property
    def max_horizon(self) -> int:
        return int(self.metadata.get("max_horizon", 30))

    def future_calendar(self, origin: pd.Timestamp, horizon: int, config: Mapping[str, Any]) -> pd.DataFrame:
        """Return calendar rows for ``origin + 1 .. origin + horizon``."""
        origin = pd.Timestamp(origin).normalize()
        end = origin + pd.Timedelta(days=int(horizon))
        if end > self.calendar["date"].max():
            extension = build_event_calendar(
                self.mode,
                self.calendar["date"].min(),
                end,
                config,
                panel=self.frame,
            )
            self.calendar = extension
            self.event_codes = _event_code_map(self.calendar)
            self.calendar["event_code"] = _apply_event_codes(self.calendar["event_name"], self.event_codes)
        dates = pd.date_range(origin + pd.Timedelta(days=1), end, freq="D")
        result = self.calendar.loc[self.calendar["date"].isin(dates)].copy()
        missing = dates.difference(pd.DatetimeIndex(result["date"]))
        if len(missing):
            filler = pd.DataFrame({"date": missing})
            filler["festival_flag"] = 0
            filler["festival_name"] = None
            filler["festival_strength"] = 1.0
            filler["wedding_season_flag"] = 0
            filler["wedding_strength"] = 1.0
            filler["event_name"] = None
            filler["event_flag"] = 0
            filler["event_code"] = -1
            result = pd.concat([result, filler], ignore_index=True)
        return result.sort_values("date").reset_index(drop=True)


def _coerce_panel(raw: pd.DataFrame, mode: str) -> pd.DataFrame:
    required = ("date", "sku", "category", "units_sold")
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise ValueError(f"[{mode}] forecasting panel is missing columns: {missing}")

    frame = raw.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["sku"] = frame["sku"].astype("string").str.strip()
    frame["category"] = frame["category"].fillna("Unknown").astype("string")
    frame["product_name"] = frame.get("product_name", frame["sku"]).astype("string")
    frame["units_sold"] = pd.to_numeric(frame["units_sold"], errors="coerce")

    invalid_dates = int(frame["date"].isna().sum())
    invalid_skus = int((frame["sku"].isna() | (frame["sku"].str.len() == 0)).sum())
    invalid_targets = int(frame["units_sold"].isna().sum())
    if invalid_dates or invalid_skus or invalid_targets:
        raise ValueError(
            f"[{mode}] invalid panel rows: dates={invalid_dates}, skus={invalid_skus}, targets={invalid_targets}"
        )
    if (frame["units_sold"] < 0).any():
        raise ValueError(f"[{mode}] units_sold contains negative values")

    duplicates = int(frame.duplicated(subset=["sku", "date"]).sum())
    if duplicates:
        LOGGER.warning("[%s] aggregating %s duplicate SKU-day rows", mode, duplicates)
        frame = (
            frame.groupby(["sku", "date"], as_index=False)
            .agg(
                product_name=("product_name", "first"),
                category=("category", "first"),
                units_sold=("units_sold", "sum"),
                selling_price=("selling_price", "mean"),
                discount_percentage=("discount_percentage", "mean"),
                revenue=("revenue", "sum"),
                festival_flag=("festival_flag", "max"),
                festival_name=("festival_name", "first"),
                promotion_flag=("promotion_flag", "max"),
                available_stock=("available_stock", "first"),
                stockout_flag=("stockout_flag", "max"),
                event_name=("event_name", "first"),
                date_imputed=("date_imputed", "max"),
                data_source=("data_source", "first"),
            )
        )

    for column in PANEL_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    frame = frame.loc[:, list(PANEL_COLUMNS)].copy()

    for column in ("selling_price", "discount_percentage", "available_stock"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
    for column in ("festival_flag", "promotion_flag", "date_imputed"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0).astype("int8")
    frame["units_sold"] = frame["units_sold"].astype("float64")
    frame["revenue"] = pd.to_numeric(frame["revenue"], errors="coerce").fillna(0.0).astype("float64")
    frame["data_source"] = frame["data_source"].fillna(mode).astype("string")
    frame["stockout_flag"] = pd.to_numeric(frame["stockout_flag"], errors="coerce").astype("Int8")

    sources = set(frame["data_source"].dropna().astype(str).unique())
    if sources and sources != {mode}:
        raise ValueError(f"[{mode}] mixed data sources found: {sorted(sources)}")

    return frame.sort_values(["sku", "date"], kind="mergesort").reset_index(drop=True)


def _complete_calendar_gaps(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    bounds = frame.groupby("sku", observed=True)["date"].agg(["min", "max"])
    expected = int(((bounds["max"] - bounds["min"]).dt.days + 1).sum())
    missing = expected - len(frame)
    if missing <= 0:
        return frame, 0

    lengths = (bounds["max"] - bounds["min"]).dt.days.to_numpy() + 1
    sku_repeat = np.repeat(bounds.index.to_numpy(), lengths)
    dates = np.concatenate(
        [pd.date_range(start, end, freq="D").to_numpy() for start, end in zip(bounds["min"], bounds["max"])]
    )
    skeleton = pd.DataFrame({"sku": sku_repeat, "date": dates})
    skeleton["sku"] = skeleton["sku"].astype(frame["sku"].dtype)
    merged = skeleton.merge(frame, on=["sku", "date"], how="left", indicator=True)
    merged["date_imputed"] = np.where(merged["_merge"] == "left_only", 1, merged["date_imputed"].fillna(0)).astype("int8")
    merged = merged.drop(columns=["_merge"]).sort_values(["sku", "date"], kind="mergesort").reset_index(drop=True)
    for column in ("product_name", "category", "selling_price", "discount_percentage", "available_stock"):
        merged[column] = merged.groupby("sku", observed=True)[column].ffill()
    for column in ("product_name", "category"):
        merged[column] = merged.groupby("sku", observed=True)[column].bfill()
    merged["units_sold"] = pd.to_numeric(merged["units_sold"], errors="coerce").fillna(0.0)
    merged["revenue"] = pd.to_numeric(merged["revenue"], errors="coerce").fillna(0.0)
    for column in ("festival_flag", "promotion_flag"):
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0).astype("int8")
    merged["data_source"] = merged["data_source"].fillna(frame["data_source"].iloc[0]).astype("string")
    return merged, int(missing)


def _event_code_map(calendar: pd.DataFrame) -> dict[str, int]:
    names = [str(name) for name in calendar["event_name"].dropna().unique() if str(name) not in {"", "None", "nan"}]
    return {name: index for index, name in enumerate(sorted(names))}


def _apply_event_codes(names: pd.Series, mapping: Mapping[str, int]) -> pd.Series:
    return names.map(lambda value: mapping.get(str(value), -1) if pd.notna(value) else -1).astype("int16")


def _empty_calendar(dates: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": dates,
            "festival_flag": 0,
            "festival_name": None,
            "festival_strength": 1.0,
            "wedding_season_flag": 0,
            "wedding_strength": 1.0,
            "event_name": None,
            "event_flag": 0,
            "event_code": -1,
        }
    )


def _m5_calendar_from_raw(config: Mapping[str, Any], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame | None:
    raw_dir = resolve_path(config["paths"]["data_raw"])
    path = raw_dir / str(config.get("data", {}).get("m5", {}).get("calendar_file", "calendar.csv"))
    if not path.exists():
        return None
    try:
        calendar = pd.read_csv(path, usecols=["date", "event_name_1", "event_name_2"])
    except (ValueError, pd.errors.ParserError):
        return None
    calendar["date"] = pd.to_datetime(calendar["date"], errors="coerce").dt.normalize()
    calendar = calendar.dropna(subset=["date"])
    calendar["event_name"] = calendar["event_name_1"].where(
        calendar["event_name_1"].notna(), calendar["event_name_2"]
    )
    calendar = calendar.loc[:, ["date", "event_name"]].drop_duplicates(subset=["date"])
    return calendar.loc[(calendar["date"] >= start) & (calendar["date"] <= end)]


def build_event_calendar(
    mode: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    config: Mapping[str, Any],
    panel: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a mode-specific daily event calendar.

    Synthetic uses the configured Indian festival and wedding calendar. UCI uses
    only documented UK bank holidays. M5 uses its own US calendar events and
    never receives Indian festival labels.
    """
    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()
    dates = pd.date_range(start, end, freq="D")

    if mode == "synthetic":
        calendar = expand_daily_event_flags(dates, config)
        calendar = calendar.rename(columns={"festival_name": "festival_name"})
        calendar["event_name"] = calendar["festival_name"]
        calendar["event_flag"] = calendar["festival_flag"].astype("int8")
    elif mode == "uci":
        flags = festival_flag_for_dates(pd.Series(dates))
        calendar = _empty_calendar(dates)
        calendar["festival_flag"] = flags.to_numpy()
        calendar["festival_name"] = np.where(calendar["festival_flag"] == 1, "UK Bank Holiday", None)
        calendar["event_name"] = calendar["festival_name"]
        calendar["event_flag"] = calendar["festival_flag"].astype("int8")
    elif mode == "m5":
        calendar = _empty_calendar(dates)
        raw_calendar = _m5_calendar_from_raw(config, start, end)
        if raw_calendar is not None and not raw_calendar.empty:
            calendar = calendar.drop(columns=["event_name"]).merge(raw_calendar, on="date", how="left")
        elif panel is not None and "event_name" in panel.columns:
            events = (
                panel.loc[:, ["date", "event_name"]]
                .dropna(subset=["event_name"])
                .drop_duplicates(subset=["date"])
            )
            calendar = calendar.drop(columns=["event_name"]).merge(events, on="date", how="left")
        else:
            calendar["event_name"] = None
        calendar["event_flag"] = calendar["event_name"].notna().astype("int8")
    else:
        raise ValueError(f"Unsupported forecasting mode: {mode}")

    for column, default in (
        ("festival_flag", 0),
        ("festival_strength", 1.0),
        ("wedding_season_flag", 0),
        ("wedding_strength", 1.0),
        ("event_flag", 0),
        ("event_code", -1),
    ):
        if column not in calendar.columns:
            calendar[column] = default
    calendar["festival_flag"] = pd.to_numeric(calendar["festival_flag"], errors="coerce").fillna(0).astype("int8")
    calendar["wedding_season_flag"] = pd.to_numeric(
        calendar["wedding_season_flag"], errors="coerce"
    ).fillna(0).astype("int8")
    calendar["festival_strength"] = pd.to_numeric(calendar["festival_strength"], errors="coerce").fillna(1.0)
    calendar["wedding_strength"] = pd.to_numeric(calendar["wedding_strength"], errors="coerce").fillna(1.0)
    calendar["event_flag"] = pd.to_numeric(calendar["event_flag"], errors="coerce").fillna(0).astype("int8")
    calendar = calendar.loc[:, ["date", *EVENT_COLUMNS]].sort_values("date").reset_index(drop=True)
    event_codes = _event_code_map(calendar)
    calendar["event_code"] = _apply_event_codes(calendar["event_name"], event_codes)
    return calendar


def _attach_calendar(frame: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    base = frame.drop(columns=[column for column in EVENT_COLUMNS if column in frame.columns])
    merged = base.merge(calendar, on="date", how="left")
    merged["festival_flag"] = merged["festival_flag"].fillna(0).astype("int8")
    merged["wedding_season_flag"] = merged["wedding_season_flag"].fillna(0).astype("int8")
    merged["festival_strength"] = merged["festival_strength"].fillna(1.0)
    merged["wedding_strength"] = merged["wedding_strength"].fillna(1.0)
    merged["event_flag"] = merged["event_flag"].fillna(0).astype("int8")
    merged["event_code"] = merged["event_code"].fillna(-1).astype("int16")
    return merged.sort_values(["sku", "date"], kind="mergesort").reset_index(drop=True)


def load_forecasting_panel(
    mode: str,
    config: Mapping[str, Any],
    processed_dir: str | Path | None = None,
    sku_limit: int | None = None,
    exclude_non_products: bool = True,
) -> ForecastingPanel:
    """Load one processed mode and return a validated forecasting panel."""
    if mode not in MODES:
        raise ValueError(f"Unsupported forecasting mode: {mode}. Choose from {MODES}")

    raw, load_meta = load_daily(
        mode,
        config,
        processed_dir=processed_dir,
        sku_limit=sku_limit,
        exclude_non_products=exclude_non_products,
        return_meta=True,
    )
    frame = _coerce_panel(raw, mode)
    frame, filled = _complete_calendar_gaps(frame)

    max_horizon = int(config.get("forecasting", {}).get("max_horizon", 30))
    start = frame["date"].min()
    end = frame["date"].max() + pd.Timedelta(days=max_horizon)
    calendar = build_event_calendar(mode, start, end, config, panel=frame)
    frame = _attach_calendar(frame, calendar)

    category_values = sorted(str(value) for value in frame["category"].dropna().unique())
    category_codes = {value: index for index, value in enumerate(category_values)}
    frame["category_code"] = frame["category"].map(category_codes).fillna(-1).astype("int16")

    first_dates = frame.groupby("sku", observed=True)["date"].min()
    frame["series_age_days"] = (
        frame["date"] - frame["sku"].map(first_dates)
    ).dt.days.astype("int32")

    event_codes = _event_code_map(calendar)
    metadata = {
        "mode": mode,
        "rows": int(len(frame)),
        "skus": int(frame["sku"].nunique()),
        "categories": int(frame["category"].nunique()),
        "date_min": str(frame["date"].min().date()),
        "date_max": str(frame["date"].max().date()),
        "days": int(frame["date"].nunique()),
        "calendar_gap_rows_filled": int(filled),
        "max_horizon": max_horizon,
        "category_codes": category_codes,
        "event_codes": event_codes,
        "load_meta": load_meta,
        "target": "units_sold (observed historical sales)",
        "loaded_at": datetime.now(timezone.utc).isoformat(),
    }
    LOGGER.info(
        "[%s] forecasting panel rows=%s skus=%s dates=%s..%s gap_rows_filled=%s",
        mode,
        metadata["rows"],
        metadata["skus"],
        metadata["date_min"],
        metadata["date_max"],
        filled,
    )
    return ForecastingPanel(
        mode=mode,
        frame=frame,
        calendar=calendar,
        category_codes=category_codes,
        first_dates=first_dates,
        event_codes=event_codes,
        metadata=metadata,
    )

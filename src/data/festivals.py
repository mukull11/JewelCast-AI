"""Festival calendars and event flags.

The Indian festival calendar is an explicit, configurable assumption used only by
the synthetic dataset. It is not presented as observed company data.

UCI Online Retail II transactions are UK-based, so this module also provides a
small documented UK bank-holiday list for 2009-2011 rather than applying Indian
festivals to UK data.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Mapping

import pandas as pd


def _festival_rows(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in (config.get("festivals", {}) or {}).get("indian", []) or []:
        rows.append(
            {
                "date": pd.Timestamp(item["date"]).normalize(),
                "festival_name": item["name"],
                "festival_type": item.get("type", "festival"),
                "strength": float(item.get("strength", 1.0)),
                "window_days": int(item.get("window_days", 0)),
            }
        )
    return rows


def _wedding_rows(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in (config.get("festivals", {}) or {}).get("wedding_seasons", []) or []:
        rows.append(
            {
                "start": pd.Timestamp(item["start"]).normalize(),
                "end": pd.Timestamp(item["end"]).normalize(),
                "wedding_name": item["name"],
                "strength": float(item.get("strength", 1.0)),
            }
        )
    return rows


def build_festival_calendar(config: Mapping[str, Any]) -> pd.DataFrame:
    """Return one row per configured Indian festival date."""
    rows = _festival_rows(config)
    columns = ["date", "festival_name", "festival_type", "strength", "window_days"]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values("date").reset_index(drop=True)


def build_wedding_calendar(config: Mapping[str, Any]) -> pd.DataFrame:
    rows = _wedding_rows(config)
    columns = ["start", "end", "wedding_name", "strength"]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values("start").reset_index(drop=True)


def expand_daily_event_flags(dates: pd.Series | pd.DatetimeIndex, config: Mapping[str, Any]) -> pd.DataFrame:
    """Expand festivals/wedding seasons onto a daily date index.

    For each date the strongest active festival is retained in ``festival_name``.
    ``festival_strength`` is the raw configured multiplier (1.0 when inactive).
    """
    index = pd.DatetimeIndex(pd.to_datetime(pd.Series(dates).drop_duplicates())).normalize().sort_values()
    result = pd.DataFrame(
        {
            "date": index,
            "festival_flag": 0,
            "festival_name": None,
            "festival_type": None,
            "festival_strength": 1.0,
            "wedding_season_flag": 0,
            "wedding_name": None,
            "wedding_strength": 1.0,
        }
    )

    for row in _festival_rows(config):
        start = row["date"] - pd.Timedelta(days=row["window_days"])
        end = row["date"] + pd.Timedelta(days=row["window_days"])
        mask = (result["date"] >= start) & (result["date"] <= end)
        stronger = mask & (row["strength"] > result["festival_strength"].to_numpy())
        result.loc[stronger, "festival_flag"] = 1
        result.loc[stronger, "festival_name"] = row["festival_name"]
        result.loc[stronger, "festival_type"] = row["festival_type"]
        result.loc[stronger, "festival_strength"] = row["strength"]

    for row in _wedding_rows(config):
        mask = (result["date"] >= row["start"]) & (result["date"] <= row["end"])
        result.loc[mask, "wedding_season_flag"] = 1
        result.loc[mask, "wedding_name"] = row["wedding_name"]
        result.loc[mask, "wedding_strength"] = row["strength"]

    return result


# Documented UK bank holidays for the UCI Online Retail II date range (2009-2011).
# These are calendar assumptions used for a UK dataset, not Indian festivals.
UK_BANK_HOLIDAYS: dict[int, tuple[str, ...]] = {
    2009: (
        "2009-01-01",
        "2009-04-10",
        "2009-04-13",
        "2009-05-04",
        "2009-05-25",
        "2009-08-31",
        "2009-12-25",
        "2009-12-28",
    ),
    2010: (
        "2010-01-01",
        "2010-04-02",
        "2010-04-05",
        "2010-05-03",
        "2010-05-31",
        "2010-08-30",
        "2010-12-27",
        "2010-12-28",
    ),
    2011: (
        "2011-01-03",
        "2011-04-22",
        "2011-04-25",
        "2011-05-02",
        "2011-05-30",
        "2011-08-29",
        "2011-12-26",
        "2011-12-27",
    ),
}


def uk_bank_holiday_set() -> set[pd.Timestamp]:
    return {
        pd.Timestamp(day).normalize()
        for days in UK_BANK_HOLIDAYS.values()
        for day in days
    }


def festival_flag_for_dates(dates: pd.Series) -> pd.Series:
    """Return 1/0 flags for UK bank holidays (used by the UCI mode)."""
    normalized = pd.to_datetime(dates).dt.normalize()
    holidays = uk_bank_holiday_set()
    return normalized.isin(holidays).astype("int8")


def is_within_wedding_season(day: date | pd.Timestamp, config: Mapping[str, Any]) -> bool:
    ts = pd.Timestamp(day).normalize()
    return any(row["start"] <= ts <= row["end"] for row in _wedding_rows(config))

"""Chronological splits and rolling forecast origins.

Random shuffling is never used. All series in a dataset share the same date
boundaries, so no SKU is trained on a date that is later than another SKU's
training data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ChronologicalSplit:
    mode: str
    dates: pd.DatetimeIndex
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    fractions: tuple[float, float, float]

    @property
    def train_days(self) -> int:
        return int((self.dates <= self.train_end).sum())

    @property
    def validation_days(self) -> int:
        return int(((self.dates > self.train_end) & (self.dates <= self.validation_end)).sum())

    @property
    def test_days(self) -> int:
        return int(((self.dates > self.validation_end) & (self.dates <= self.test_end)).sum())

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "train_start": str(self.train_start.date()),
            "train_end": str(self.train_end.date()),
            "validation_start": str(self.validation_start.date()),
            "validation_end": str(self.validation_end.date()),
            "test_start": str(self.test_start.date()),
            "test_end": str(self.test_end.date()),
            "train_days": self.train_days,
            "validation_days": self.validation_days,
            "test_days": self.test_days,
            "fractions": list(self.fractions),
        }


def _resolve_fractions(config: Mapping[str, Any]) -> tuple[float, float, float]:
    split = (config.get("forecasting", {}) or {}).get("split", {}) or {}
    train = float(split.get("train_fraction", 0.70))
    validation = float(split.get("validation_fraction", 0.15))
    test = float(split.get("test_fraction", 0.15))
    total = train + validation + test
    if total <= 0:
        raise ValueError("Split fractions must be positive")
    return train / total, validation / total, test / total


def chronological_split(
    dates: pd.Series | pd.DatetimeIndex | Sequence[pd.Timestamp],
    mode: str,
    config: Mapping[str, Any],
) -> ChronologicalSplit:
    """Split unique dates into chronological train/validation/test periods.

    The configured 70/15/15 fractions are used when there is enough history.
    If the resulting validation or test period is shorter than the configured
    minimum window, the boundaries are moved earlier while preserving the
    chronological order.
    """
    unique = pd.DatetimeIndex(sorted(pd.to_datetime(pd.Series(dates)).dt.normalize().unique()))
    if len(unique) < 3:
        raise ValueError("At least three distinct dates are required for a chronological split")

    train_fraction, validation_fraction, test_fraction = _resolve_fractions(config)
    n = len(unique)
    min_validation = int((config.get("forecasting", {}) or {}).get("validation_days", 30))
    min_test = int((config.get("forecasting", {}) or {}).get("test_days", 30))

    train_end_index = max(0, int(np.floor(n * train_fraction)) - 1)
    validation_end_index = max(train_end_index + 1, int(np.floor(n * (train_fraction + validation_fraction))) - 1)
    validation_end_index = min(validation_end_index, n - 2)

    if n >= (min_validation + min_test + 2):
        train_end_index = min(train_end_index, n - min_validation - min_test - 1)
        validation_end_index = min(max(validation_end_index, train_end_index + min_validation), n - min_test - 1)
    else:
        train_end_index = min(train_end_index, n - 3)
        validation_end_index = min(max(validation_end_index, train_end_index + 1), n - 2)

    train_end = unique[train_end_index]
    validation_start = unique[train_end_index + 1]
    validation_end = unique[validation_end_index]
    test_start = unique[validation_end_index + 1]
    test_end = unique[-1]

    return ChronologicalSplit(
        mode=mode,
        dates=unique,
        train_start=unique[0],
        train_end=train_end,
        validation_start=validation_start,
        validation_end=validation_end,
        test_start=test_start,
        test_end=test_end,
        fractions=(train_fraction, validation_fraction, test_fraction),
    )


def rolling_origins(
    split: ChronologicalSplit,
    period: str,
    horizon: int,
    n_origins: int = 1,
) -> list[pd.Timestamp]:
    """Return forecast origins for a validation or test period.

    An origin is the last observed date before a forecast window. Every origin
    leaves at least ``horizon`` future dates inside the requested period.
    """
    if period == "validation":
        period_start = split.train_end
        period_end = split.validation_end
    elif period == "test":
        period_start = split.validation_end
        period_end = split.test_end
    elif period == "all":
        period_start = split.dates[0]
        period_end = split.dates[-1]
    else:
        raise ValueError(f"Unsupported period: {period}")

    candidates = split.dates[(split.dates >= period_start) & (split.dates <= period_end)]
    if len(candidates) == 0:
        return [period_start]
    max_origin = period_end - pd.Timedelta(days=int(horizon))
    eligible = [date for date in candidates if date <= max_origin]
    if not eligible:
        return [candidates[0]]

    count = max(1, int(n_origins))
    if count == 1 or len(eligible) == 1:
        return [eligible[0]]
    positions = np.linspace(0, len(eligible) - 1, num=count, dtype=int)
    return sorted({eligible[index] for index in positions})

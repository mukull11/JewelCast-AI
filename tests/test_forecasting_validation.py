"""Unit tests for the M3 forecasting validation layer."""

from __future__ import annotations

import pandas as pd
import pytest

from src.forecasting.validation import ChronologicalSplit, chronological_split, rolling_origins


def _config() -> dict:
    return {
        "forecasting": {
            "split": {
                "train_fraction": 0.70,
                "validation_fraction": 0.15,
                "test_fraction": 0.15,
            },
            "validation_days": 30,
            "test_days": 30,
        }
    }


def test_chronological_split_is_ordered_and_disjoint() -> None:
    dates = pd.date_range("2023-01-01", periods=200, freq="D")
    split = chronological_split(dates, "synthetic", _config())
    assert isinstance(split, ChronologicalSplit)
    assert split.train_end < split.validation_start
    assert split.validation_end < split.test_start
    assert split.test_end == dates[-1]
    assert split.train_start == dates[0]
    assert split.train_days + split.validation_days + split.test_days == len(split.dates)
    assert split.validation_days > 0
    assert split.test_days > 0


def test_chronological_split_never_uses_randomness() -> None:
    dates = pd.date_range("2023-01-01", periods=120, freq="D")
    split = chronological_split(dates, "uci", _config())
    assert split.train_end < split.validation_start
    assert split.validation_end < split.test_start
    assert split.train_end < split.validation_end
    assert split.validation_end < split.test_end


def test_chronological_split_short_history_uses_configured_windows() -> None:
    config = {
        "forecasting": {
            "split": {
                "train_fraction": 0.70,
                "validation_fraction": 0.15,
                "test_fraction": 0.15,
            },
            "validation_days": 30,
            "test_days": 30,
        }
    }
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    split = chronological_split(dates, "m5", config)
    assert split.validation_end < split.test_start
    assert split.test_end == dates[-1]


def test_rolling_origins_are_sorted_and_within_period() -> None:
    dates = pd.date_range("2023-01-01", periods=200, freq="D")
    split = chronological_split(dates, "synthetic", _config())
    origins = rolling_origins(split, "validation", horizon=7, n_origins=2)
    assert origins == sorted(origins)
    for origin in origins:
        assert split.train_end <= origin <= split.validation_end
        assert origin <= split.validation_end - pd.Timedelta(days=7)
    assert len(origins) == 2
    test_origins = rolling_origins(split, "test", horizon=7, n_origins=1)
    assert len(test_origins) == 1


def test_rolling_origins_rejects_unknown_period() -> None:
    split = chronological_split(pd.date_range("2023-01-01", periods=60, freq="D"), "synthetic", _config())
    with pytest.raises(ValueError):
        rolling_origins(split, "unknown", horizon=7, n_origins=1)
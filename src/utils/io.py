"""I/O helpers: hashing, Parquet and JSON serialisation."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .paths import resolve_path


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute the SHA-256 checksum of a file without loading it fully."""
    file_path = resolve_path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataframe_fingerprint(frame: pd.DataFrame) -> str:
    """Deterministic fingerprint of a DataFrame's values and index."""
    hashed = pd.util.hash_pandas_object(frame, index=True).to_numpy(dtype="uint64")
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def save_parquet(frame: pd.DataFrame, path: str | Path) -> Path:
    """Write a DataFrame to Parquet, creating parent directories."""
    file_path = resolve_path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(file_path, index=False)
    return file_path


def read_parquet(path: str | Path) -> pd.DataFrame:
    file_path = resolve_path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {file_path}")
    return pd.read_parquet(file_path)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if math.isnan(float(value)) or math.isinf(float(value)):
            return None
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, tuple)):
        return list(value)
    if pd.isna(value):
        return None
    raise TypeError(f"Object of type {type(value)!r} is not JSON serialisable")


def write_json(data: Any, path: str | Path, indent: int = 2) -> Path:
    """Write JSON with numpy/pandas-aware serialisation."""
    file_path = resolve_path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=indent, default=_json_default, ensure_ascii=False)
        handle.write("\n")
    return file_path

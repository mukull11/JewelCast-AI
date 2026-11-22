"""Shared utilities: configuration, paths, logging and I/O helpers."""

from .paths import PROJECT_ROOT, resolve_path
from .config import load_config, get_config_value, ensure_project_dirs
from .logging import get_logger
from .io import (
    sha256_file,
    dataframe_fingerprint,
    save_parquet,
    read_parquet,
    write_json,
)

__all__ = [
    "PROJECT_ROOT",
    "resolve_path",
    "load_config",
    "get_config_value",
    "ensure_project_dirs",
    "get_logger",
    "sha256_file",
    "dataframe_fingerprint",
    "save_parquet",
    "read_parquet",
    "write_json",
]

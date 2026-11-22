"""Configuration loading and validation."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import yaml

from .paths import PROJECT_ROOT, resolve_path

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def _deep_merge(base: dict, override: Mapping[str, Any]) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(
    path: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict:
    """Load ``config.yaml`` and optionally apply a nested override mapping."""
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if overrides:
        config = _deep_merge(config, overrides)
    return config


def get_config_value(config: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    """Read a nested value using a dotted key, e.g. ``data.m5.subset.n_items``."""
    node: Any = config
    for part in dotted_key.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return default
        node = node[part]
    return node


def ensure_project_dirs(config: Mapping[str, Any]) -> None:
    """Create every directory declared under ``paths``."""
    for value in (config.get("paths") or {}).values():
        ensure_path = resolve_path(value)
        ensure_path.mkdir(parents=True, exist_ok=True)

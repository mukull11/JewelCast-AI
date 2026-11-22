"""Path helpers.

All relative paths in ``config.yaml`` are resolved against the repository root,
so scripts and the dashboard behave the same regardless of the working directory.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(value: str | Path) -> Path:
    """Resolve a config path relative to the repository root."""
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def ensure_dir(value: str | Path) -> Path:
    """Create a directory (and parents) if it does not exist."""
    path = resolve_path(value)
    path.mkdir(parents=True, exist_ok=True)
    return path

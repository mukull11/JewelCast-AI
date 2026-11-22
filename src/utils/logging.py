"""Project logging configuration."""

from __future__ import annotations

import logging
from pathlib import Path

from .paths import resolve_path

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_CONFIGURED: set[str] = set()


def get_logger(name: str, log_file: str | Path = "logs/jewelcast.log", level: int | str = "INFO") -> logging.Logger:
    """Return a logger that writes to the console and a rotating-free file handler.

    Handlers are attached only once per logger to avoid duplicate lines.
    """
    logger = logging.getLogger(name)
    if name in _CONFIGURED:
        return logger

    logger.setLevel(level if isinstance(level, int) else getattr(logging, str(level).upper(), logging.INFO))
    logger.propagate = False

    formatter = logging.Formatter(_FORMAT)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    path = resolve_path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    _CONFIGURED.add(name)
    return logger

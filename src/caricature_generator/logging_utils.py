"""Logging utilities for the caricature generator."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from loguru import logger


def configure_logging(level: str, log_dir: Path) -> None:
    """Configure loguru sinks for console and optional file logging."""
    logger.remove()
    logger.add(
        sink=lambda msg: print(msg, end=""),  # loguru uses lazy writer
        level=level.upper(),
        colorize=True,
        backtrace=False,
        diagnose=False,
    )

    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(log_dir / "pipeline.log", level=level.upper(), rotation="10 MB", retention=5)


def get_logger(name: Optional[str] = None):
    """Return a child logger for module-scoped logging."""
    return logger.bind(module=name or __name__)


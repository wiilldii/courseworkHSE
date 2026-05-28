"""
Logging utilities and shared helpers.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

from src.config import OUT_LOGS, MISSING_MARKERS


def setup_logger(name: str = "coursework") -> logging.Logger:
    OUT_LOGS.mkdir(parents=True, exist_ok=True)
    log_path = OUT_LOGS / "run_log.txt"

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)

    import io
    safe_stream = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
    )
    ch = logging.StreamHandler(safe_stream)
    ch.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(f"Run started: {datetime.now().isoformat()}")
    return logger


def is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip() in MISSING_MARKERS:
        return True
    return False


def to_numeric(value):
    """Convert a cell value to float; return None for missing markers."""
    if is_missing(value):
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def safe_divide(numerator, denominator):
    """Return numerator/denominator or None if denominator is 0/None."""
    if numerator is None or denominator is None:
        return None
    if denominator == 0:
        return None
    return numerator / denominator

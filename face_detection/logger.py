"""Centralised logging configuration.

Every module obtains its logger via :func:`get_logger`, and the root logger is
configured exactly once through :func:`setup_logging`. This gives the whole
application consistent, structured, timestamped output and satisfies the
"no ``print()``" requirement -- the console reporter and all diagnostics flow
through the standard :mod:`logging` framework.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

from .config import LoggingConfig

_CONFIGURED = False


def setup_logging(config: LoggingConfig) -> None:
    """Configure the root logger from :class:`~config.LoggingConfig`.

    Idempotent: calling it more than once will not attach duplicate handlers,
    which matters when the application is embedded (e.g. behind a future GUI).
    """

    global _CONFIGURED
    if _CONFIGURED:
        return

    level = getattr(logging, config.level.upper(), logging.INFO)
    formatter = logging.Formatter(fmt=config.fmt, datefmt=config.datefmt)

    root = logging.getLogger()
    root.setLevel(level)

    # Console handler (stdout) -- this is where FPS / device stats appear.
    console = logging.StreamHandler(stream=sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    # Optional rotating-free file handler for post-mortem analysis.
    if config.file_path:
        file_handler = logging.FileHandler(config.file_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    _CONFIGURED = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a module-scoped logger (thin wrapper around :func:`logging.getLogger`)."""

    return logging.getLogger(name if name else "face_detection")

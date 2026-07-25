"""Logging helpers shared by command-line workflows."""

from __future__ import annotations

import logging


def configure_logging(verbose: bool = False) -> None:
    """Configure concise, timestamped console logging."""

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

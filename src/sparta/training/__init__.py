"""Two-stage training utilities for SPARTA."""

from __future__ import annotations

from typing import Any

__all__ = ["SpartaTrainer"]


def __getattr__(name: str) -> Any:
    """Import the trainer lazily so loss utilities remain lightweight."""

    if name == "SpartaTrainer":
        from sparta.training.trainer import SpartaTrainer

        return SpartaTrainer
    raise AttributeError(name)

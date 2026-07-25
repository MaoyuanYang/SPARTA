"""Numerical smoke tests for SPARTA loss functions."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from sparta.training.losses import pearson_correlation  # noqa: E402


def test_pearson_correlation_matches_perfect_linear_relationship() -> None:
    values = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert pearson_correlation(values, values * 2.0).item() == pytest.approx(1.0)

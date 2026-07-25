"""Lightweight tests for configuration loading and validation."""

from __future__ import annotations

import copy
from pathlib import Path

from sparta.config import SpartaConfig, load_config, validate_config

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CONFIG = REPOSITORY_ROOT / "configs" / "k562.example.yaml"


def test_example_configuration_is_structurally_valid() -> None:
    config = load_config(EXAMPLE_CONFIG)
    report = validate_config(config, check_files=False)
    assert report.errors == ()


def test_relative_paths_are_resolved_from_configuration_directory() -> None:
    config = load_config(EXAMPLE_CONFIG)
    expected = (EXAMPLE_CONFIG.parent / "../data/raw/reference/histone_genes.tsv").resolve()
    assert config.resource_path("histone_genes") == expected


def test_overlapping_chromosome_splits_are_rejected() -> None:
    original = load_config(EXAMPLE_CONFIG)
    values = copy.deepcopy(original.values)
    values["training"]["validation_chromosomes"] = ["chr1"]
    config = SpartaConfig(source_path=original.source_path, values=values)
    report = validate_config(config)
    assert any("overlap" in error.lower() for error in report.errors)

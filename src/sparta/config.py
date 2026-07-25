"""Configuration loading, path resolution, and validation."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_MISSING = object()
REQUIRED_RESOURCES = (
    "histone_genes",
    "gencode_gtf",
    "cage_peaks_bed",
    "cage_peak_names",
    "cage_annotations",
    "rna_seq",
    "atac_peaks",
    "reference_fasta",
    "hic",
)
MANUSCRIPT_TRACKS = (
    "ATAC",
    "CTCF",
    "RAD21",
    "H3K27ac",
    "H3K4me3",
    "H3K4me1",
    "H3K27me3",
    "H3K9me3",
)


@dataclass(frozen=True)
class ValidationReport:
    """Structured configuration validation result."""

    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def raise_for_errors(self) -> None:
        if self.errors:
            details = "\n".join(f"- {message}" for message in self.errors)
            raise ValueError(f"Invalid SPARTA configuration:\n{details}")


@dataclass(frozen=True)
class SpartaConfig:
    """Loaded YAML configuration with paths relative to the YAML file."""

    source_path: Path
    values: dict[str, Any]

    @property
    def base_dir(self) -> Path:
        return self.source_path.parent

    def get(self, dotted_key: str, default: Any = _MISSING) -> Any:
        value: Any = self.values
        for key in dotted_key.split("."):
            if not isinstance(value, dict) or key not in value:
                if default is _MISSING:
                    raise KeyError(f"Configuration key not found: {dotted_key}")
                return default
            value = value[key]
        return value

    def section(self, name: str) -> dict[str, Any]:
        value = self.get(name)
        if not isinstance(value, dict):
            raise TypeError(f"Configuration section '{name}' must be a mapping.")
        return value

    def resolve_path(self, value: str | Path) -> Path:
        expanded = os.path.expandvars(os.path.expanduser(str(value)))
        path = Path(expanded)
        if not path.is_absolute():
            path = self.base_dir / path
        return path.resolve()

    def path(self, dotted_key: str) -> Path:
        value = self.get(dotted_key)
        if isinstance(value, dict):
            value = value.get("path")
        if not isinstance(value, str | Path):
            raise TypeError(f"Configuration key '{dotted_key}' must define a path.")
        return self.resolve_path(value)

    def resource(self, name: str) -> dict[str, Any]:
        resource = self.get(f"resources.{name}")
        if not isinstance(resource, dict):
            raise TypeError(f"Resource '{name}' must be a mapping.")
        return resource

    def resource_path(self, name: str) -> Path:
        return self.path(f"resources.{name}")

    def output_path(self, name: str) -> Path:
        return self.path(f"outputs.{name}")

    def epigenomic_track_paths(self) -> dict[str, Path]:
        tracks = self.section("epigenomic_tracks")
        return {name: self.resolve_path(entry["path"]) for name, entry in tracks.items()}

    def canonical_json(self) -> str:
        return json.dumps(self.values, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def serializable_copy(self, resolve_paths: bool = False) -> dict[str, Any]:
        result = copy.deepcopy(self.values)
        if not resolve_paths:
            return result

        for section_name in ("resources", "epigenomic_tracks"):
            for entry in result.get(section_name, {}).values():
                if isinstance(entry, dict) and "path" in entry:
                    entry["path"] = str(self.resolve_path(entry["path"]))
        for key, value in result.get("outputs", {}).items():
            result["outputs"][key] = str(self.resolve_path(value))
        return result


def load_config(path: str | Path) -> SpartaConfig:
    """Load a UTF-8 YAML configuration file."""

    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {source_path}")
    with source_path.open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle)
    if not isinstance(values, dict):
        raise ValueError("The configuration root must be a YAML mapping.")
    return SpartaConfig(source_path=source_path, values=values)


def validate_config(config: SpartaConfig, check_files: bool = False) -> ValidationReport:
    """Validate structure, scientific defaults, and optionally input files."""

    errors: list[str] = []
    warnings: list[str] = []

    for section in ("project", "resources", "epigenomic_tracks", "outputs", "data", "dataset", "model", "training"):
        if not isinstance(config.get(section, None), dict):
            errors.append(f"Missing or invalid '{section}' section.")

    resources = config.get("resources", {})
    for name in REQUIRED_RESOURCES:
        entry = resources.get(name)
        if not isinstance(entry, dict) or not entry.get("path"):
            errors.append(f"Resource '{name}' must define a path.")
            continue
        if check_files:
            path = config.resolve_path(entry["path"])
            if not path.is_file():
                if entry.get("url"):
                    warnings.append(f"Resource '{name}' is missing locally but has a download URL: {path}")
                else:
                    errors.append(f"Resource '{name}' does not exist: {path}")

    tracks = config.get("epigenomic_tracks", {})
    if not isinstance(tracks, dict) or not tracks:
        errors.append("At least one epigenomic track is required.")
    else:
        missing_manuscript_tracks = [name for name in MANUSCRIPT_TRACKS if name not in tracks]
        if missing_manuscript_tracks:
            warnings.append(
                "The configuration does not reproduce the manuscript's eight-track input set; "
                f"missing: {', '.join(missing_manuscript_tracks)}"
            )
        for name, entry in tracks.items():
            if not isinstance(entry, dict) or not entry.get("path"):
                errors.append(f"Epigenomic track '{name}' must define a path.")
            elif check_files:
                path = config.resolve_path(entry["path"])
                if not path.is_file():
                    if entry.get("url"):
                        warnings.append(f"Track '{name}' is missing locally but has a download URL: {path}")
                    else:
                        errors.append(f"Epigenomic track '{name}' does not exist: {path}")

    for output_name in ("tss_anchors", "gene_labels", "dataset_dir", "training_dir"):
        if not config.get(f"outputs.{output_name}", None):
            errors.append(f"Output '{output_name}' is required.")

    dataset = config.get("dataset", {})
    model = config.get("model", {})
    training = config.get("training", {})

    max_nodes = dataset.get("max_nodes", 0)
    window_bp = dataset.get("window_bp", 0)
    regulatory_window_bp = dataset.get("regulatory_window_bp", 0)
    if not isinstance(max_nodes, int) or max_nodes < 2:
        errors.append("dataset.max_nodes must be an integer of at least 2.")
    if not isinstance(window_bp, int) or window_bp <= 0 or window_bp % 2:
        errors.append("dataset.window_bp must be a positive even integer.")
    if not isinstance(regulatory_window_bp, int) or regulatory_window_bp <= 0 or regulatory_window_bp % 2:
        errors.append("dataset.regulatory_window_bp must be a positive even integer.")
    if dataset.get("peak_selection") not in {"nearest", "score"}:
        errors.append("dataset.peak_selection must be 'nearest' or 'score'.")
    if dataset.get("hic_resolution_bp", 0) <= 0:
        errors.append("dataset.hic_resolution_bp must be positive.")
    if dataset.get("n_jobs", 0) < 1:
        errors.append("dataset.n_jobs must be at least 1.")
    if dataset.get("batch_size", 0) < 1:
        errors.append("dataset.batch_size must be at least 1.")
    if isinstance(window_bp, int) and isinstance(regulatory_window_bp, int) and regulatory_window_bp < window_bp:
        errors.append("dataset.regulatory_window_bp must be at least dataset.window_bp.")
    if dataset.get("expression_zscore_ddof", 1) not in {0, 1}:
        errors.append("dataset.expression_zscore_ddof must be 0 or 1.")

    hidden_dim = model.get("hidden_dim", 0)
    physics_heads = model.get("physics_heads", 0)
    if not isinstance(hidden_dim, int) or hidden_dim <= 0:
        errors.append("model.hidden_dim must be a positive integer.")
    if not isinstance(physics_heads, int) or physics_heads <= 0:
        errors.append("model.physics_heads must be a positive integer.")
    elif isinstance(hidden_dim, int) and hidden_dim % physics_heads:
        errors.append("model.hidden_dim must be divisible by model.physics_heads.")
    if not isinstance(model.get("physics_layers"), int) or model.get("physics_layers", 0) < 1:
        errors.append("model.physics_layers must be a positive integer.")
    dropout = model.get("dropout", 0.0)
    if not isinstance(dropout, int | float) or not 0.0 <= dropout < 1.0:
        errors.append("model.dropout must be in the interval [0, 1).")

    split_keys = ("train_chromosomes", "validation_chromosomes", "test_chromosomes")
    splits: dict[str, set[str]] = {}
    for key in split_keys:
        value = training.get(key)
        if not isinstance(value, list) or not value:
            errors.append(f"training.{key} must be a non-empty list.")
            splits[key] = set()
        else:
            splits[key] = set(value)
    if splits["train_chromosomes"] & splits["validation_chromosomes"]:
        errors.append("Training and validation chromosome sets overlap.")
    if splits["train_chromosomes"] & splits["test_chromosomes"]:
        errors.append("Training and test chromosome sets overlap.")
    if splits["validation_chromosomes"] & splits["test_chromosomes"]:
        errors.append("Validation and test chromosome sets overlap.")

    configured_chromosomes = set(config.get("data.chromosomes", []))
    split_chromosomes = set().union(*splits.values())
    unavailable = sorted(split_chromosomes - configured_chromosomes)
    if unavailable:
        errors.append(f"Split chromosomes are absent from data.chromosomes: {', '.join(unavailable)}")

    for stage_name in ("stage1", "stage2"):
        stage = training.get(stage_name)
        if not isinstance(stage, dict):
            errors.append(f"training.{stage_name} must be a mapping.")
        elif not isinstance(stage.get("epochs"), int) or stage["epochs"] < 0:
            errors.append(f"training.{stage_name}.epochs must be a non-negative integer.")
    if training.get("batch_size", 0) < 1:
        errors.append("training.batch_size must be at least 1.")
    if training.get("num_workers", -1) < 0:
        errors.append("training.num_workers must be non-negative.")
    if training.get("validation_interval", 0) < 1:
        errors.append("training.validation_interval must be at least 1.")

    return ValidationReport(errors=tuple(errors), warnings=tuple(warnings))

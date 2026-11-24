"""End-to-end SPARTA workflow orchestration."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from sparta.acquire import acquire_resources
from sparta.config import SpartaConfig, validate_config

LOGGER = logging.getLogger(__name__)
DATA_STAGES = ("anchors", "labels", "dataset")
ALL_STAGES = ("download", *DATA_STAGES, "train")


def _summary_path(output_path: Path) -> Path:
    return output_path.with_suffix(f"{output_path.suffix}.summary.json")


def _write_summary(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _run_anchors(config: SpartaConfig, force: bool) -> None:
    output = config.output_path("tss_anchors")
    if output.exists() and not force:
        LOGGER.info("Skipping TSS anchor curation; output exists: %s", output)
        return
    from sparta.data.anchors import curate_tss_anchors

    summary = curate_tss_anchors(config)
    _write_summary(_summary_path(output), summary)


def _run_labels(config: SpartaConfig, force: bool) -> None:
    output = config.output_path("gene_labels")
    if output.exists() and not force:
        LOGGER.info("Skipping expression-label integration; output exists: %s", output)
        return
    from sparta.data.labels import build_expression_labels

    summary = build_expression_labels(config)
    _write_summary(_summary_path(output), summary)


def _run_dataset(config: SpartaConfig, force: bool) -> None:
    output_dir = config.output_path("dataset_dir")
    marker = output_dir / "_SUCCESS.json"
    if marker.exists() and not force:
        summary = json.loads(marker.read_text(encoding="utf-8"))
        if "peak_center_method" not in summary:
            raise ValueError(
                f"{marker} predates peak-center tracking. Use a new dataset directory or rebuild with --force."
            )
        if summary.get("configuration_sha256") == config.digest():
            LOGGER.info("Skipping HDF5 dataset construction; completion marker matches: %s", marker)
            return
    from sparta.data.builder import build_hdf5_dataset

    if force:
        # An interrupted rebuild may leave chromosomes from two different runs.
        marker.unlink(missing_ok=True)
    summary = build_hdf5_dataset(config, force=force)
    _write_summary(marker, summary)


def _run_training(config: SpartaConfig, force: bool) -> None:
    output_dir = config.output_path("training_dir")
    marker = output_dir / "run_complete.json"
    if marker.exists() and not force:
        LOGGER.info("Skipping training; completion marker exists: %s", marker)
        return
    from sparta.training.trainer import SpartaTrainer

    trainer = SpartaTrainer(config)
    summary = trainer.run(force=force)
    _write_summary(marker, summary)


def run_pipeline(config: SpartaConfig, stage: str = "all", force: bool = False) -> None:
    """Run one stage or the complete configuration-driven workflow."""

    report = validate_config(config, check_files=False)
    for warning in report.warnings:
        LOGGER.warning(warning)
    report.raise_for_errors()

    runners: dict[str, Callable[[SpartaConfig, bool], None]] = {
        "download": lambda cfg, overwrite: acquire_resources(cfg, overwrite),
        "anchors": _run_anchors,
        "labels": _run_labels,
        "dataset": _run_dataset,
        "train": _run_training,
    }

    if stage != "all" and stage not in runners:
        raise ValueError(f"Unknown pipeline stage: {stage}")

    if stage == "all":
        stages = [*DATA_STAGES, "train"]
        if bool(config.get("pipeline.download_missing", False)):
            stages.insert(0, "download")
    else:
        stages = [stage]

    started = datetime.now(timezone.utc)
    LOGGER.info("Starting SPARTA pipeline stages: %s", ", ".join(stages))
    for current_stage in stages:
        LOGGER.info("Stage started: %s", current_stage)
        runners[current_stage](config, force)
        LOGGER.info("Stage completed: %s", current_stage)
    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    LOGGER.info("SPARTA pipeline completed in %.1f seconds.", elapsed)

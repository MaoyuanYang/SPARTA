"""Reproducible two-stage training and held-out evaluation for SPARTA."""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import random
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from sparta import __version__
from sparta.config import SpartaConfig
from sparta.data.hdf5_dataset import create_dataloader
from sparta.models import SpartaModel
from sparta.training.losses import (
    ExpressionPredictionLoss,
    GlobalHiCPearsonLoss,
    StratifiedHiCPearsonLoss,
    pearson_correlation,
)

LOGGER = logging.getLogger(__name__)
READOUT_MODULE_NAMES = (
    "regulatory_potential_head",
    "biochemical_affinity_head",
    "basal_expression_head",
)


def set_reproducibility(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch and select deterministic cuDNN behavior."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        LOGGER.warning("CUDA was requested but is unavailable; using CPU.")
        return torch.device("cpu")
    return torch.device(requested)


class SpartaTrainer:
    """Train SPARTA with physics pre-training and joint fine-tuning."""

    def __init__(self, config: SpartaConfig) -> None:
        self.config = config
        self.training_config = config.section("training")
        self.model_config = config.section("model")
        self.output_dir = config.output_path("training_dir")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.seed = int(self.training_config.get("seed", 3407))
        set_reproducibility(self.seed)
        self.device = _resolve_device(str(self.training_config.get("device", "auto")))
        self.use_amp = bool(self.training_config.get("use_amp", True)) and (self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        dataset_dir = config.output_path("dataset_dir")
        train_paths = self._chromosome_paths(
            dataset_dir,
            self.training_config["train_chromosomes"],
        )
        validation_paths = self._chromosome_paths(
            dataset_dir,
            self.training_config["validation_chromosomes"],
        )
        test_paths = self._chromosome_paths(
            dataset_dir,
            self.training_config["test_chromosomes"],
        )
        selected_tracks = self.training_config.get("selected_epigenomic_tracks")
        self.data_generators = {
            name: torch.Generator().manual_seed(self.seed + index)
            for index, name in enumerate(("train", "validation", "test"))
        }
        common_loader_arguments = {
            "batch_size": int(self.training_config.get("batch_size", 12)),
            "num_workers": int(self.training_config.get("num_workers", 4)),
            "pin_memory": self.device.type == "cuda",
            "coordinate_scale": float(self.training_config.get("coordinate_scale", 1.0)),
            "epigenomic_tracks": selected_tracks,
        }
        self.train_loader = create_dataloader(
            train_paths,
            shuffle=True,
            return_hic=True,
            return_metadata=False,
            generator=self.data_generators["train"],
            **common_loader_arguments,
        )
        self.validation_loader = create_dataloader(
            validation_paths,
            shuffle=False,
            return_hic=True,
            return_metadata=False,
            generator=self.data_generators["validation"],
            **common_loader_arguments,
        )
        self.test_loader = create_dataloader(
            test_paths,
            shuffle=False,
            return_hic=False,
            return_metadata=True,
            generator=self.data_generators["test"],
            **common_loader_arguments,
        )

        self.train_loader.dataset.ensure_compatible(self.validation_loader.dataset, "validation split")
        self.train_loader.dataset.ensure_compatible(self.test_loader.dataset, "test split")
        self.input_metadata = self.train_loader.dataset.input_metadata
        epigenomic_channels = self.train_loader.dataset.number_of_epigenomic_tracks
        self.model = SpartaModel.from_config(
            self.model_config,
            epigenomic_channels=epigenomic_channels,
        ).to(self.device)
        loss_config = self.training_config.get("hic_loss", {})
        self.stratified_hic_loss = StratifiedHiCPearsonLoss(
            distance_bins_bp=loss_config.get(
                "distance_bins_bp",
                [20000, 100000],
            ),
            minimum_pairs_per_bin=int(loss_config.get("minimum_pairs_per_bin", 30)),
        )
        self.global_hic_loss = GlobalHiCPearsonLoss(minimum_pairs=int(loss_config.get("global_minimum_pairs", 100)))
        self.expression_loss = ExpressionPredictionLoss()
        self.best_scores = {
            "stage1": -math.inf,
            "stage2": -math.inf,
        }
        self.metrics_path = self.output_dir / "metrics.jsonl"
        LOGGER.info(
            "Initialized SPARTA on %s with %d training, %d validation, and %d test genes.",
            self.device,
            len(self.train_loader.dataset),
            len(self.validation_loader.dataset),
            len(self.test_loader.dataset),
        )

    @staticmethod
    def _chromosome_paths(dataset_dir: Path, chromosomes: list[str]) -> list[Path]:
        paths = [dataset_dir / f"{chromosome}.h5" for chromosome in chromosomes]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Chromosome HDF5 files are missing: {missing[:3]}. Run the dataset stage first.")
        return paths

    def _set_trainable_modules(self, stage: int) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad = True
        if stage == 1:
            for name, parameter in self.model.named_parameters():
                if any(module_name in name for module_name in READOUT_MODULE_NAMES):
                    parameter.requires_grad = False

    def _optimizer(self, stage: int) -> optim.Optimizer:
        stage_config = self.training_config[f"stage{stage}"]
        readout_parameters: list[torch.nn.Parameter] = []
        backbone_parameters: list[torch.nn.Parameter] = []
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            destination = (
                readout_parameters
                if any(module_name in name for module_name in READOUT_MODULE_NAMES)
                else backbone_parameters
            )
            destination.append(parameter)

        if stage == 1:
            return optim.AdamW(
                backbone_parameters,
                lr=float(stage_config["learning_rate"]),
                weight_decay=float(stage_config.get("weight_decay", 0.0)),
            )
        return optim.AdamW(
            [
                {
                    "params": backbone_parameters,
                    "lr": float(stage_config["backbone_learning_rate"]),
                },
                {
                    "params": readout_parameters,
                    "lr": float(stage_config["readout_learning_rate"]),
                },
            ],
            weight_decay=float(stage_config.get("weight_decay", 0.01)),
        )

    def _scheduler(
        self,
        optimizer: optim.Optimizer,
        epochs: int,
    ) -> torch.optim.lr_scheduler.LRScheduler:
        scheduler_config = self.training_config.get("scheduler", {})
        minimum_learning_rate = float(scheduler_config.get("minimum_learning_rate", 1e-6))
        if epochs < 2:
            return CosineAnnealingLR(
                optimizer,
                T_max=max(1, epochs),
                eta_min=minimum_learning_rate,
            )
        warmup_epochs = max(
            1,
            int(epochs * float(scheduler_config.get("warmup_fraction", 0.1))),
        )
        warmup_epochs = min(warmup_epochs, epochs - 1)
        decay_epochs = epochs - warmup_epochs
        warmup = LinearLR(
            optimizer,
            start_factor=float(scheduler_config.get("warmup_start_factor", 0.01)),
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        decay = CosineAnnealingLR(
            optimizer,
            T_max=max(1, decay_epochs),
            eta_min=minimum_learning_rate,
        )
        return SequentialLR(
            optimizer,
            schedulers=[warmup, decay],
            milestones=[warmup_epochs],
        )

    def _calculate_loss(
        self,
        outputs: dict[str, torch.Tensor | None],
        batch: dict[str, Any],
        stage: int,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        interaction_energy = outputs["interaction_energy"]
        if interaction_energy is None:
            raise RuntimeError("Interaction energy is required for training.")
        stage_config = self.training_config[f"stage{stage}"]
        node_mask = batch["node_mask"].to(self.device, non_blocking=True)
        genomic_coordinates = batch["genomic_coordinates"].to(
            self.device,
            non_blocking=True,
        )
        stratified_loss, stratified_metrics = self.stratified_hic_loss(
            interaction_energy,
            batch["hic_oe"].to(self.device, non_blocking=True),
            genomic_coordinates,
            node_mask,
        )
        global_loss, global_metrics = self.global_hic_loss(
            interaction_energy,
            batch["hic_scale"].to(self.device, non_blocking=True),
            node_mask,
        )
        total_loss = (
            float(stage_config["stratified_hic_weight"]) * stratified_loss
            + float(stage_config["global_hic_weight"]) * global_loss
        )
        metrics = {**stratified_metrics, **global_metrics}

        if stage == 2:
            prediction = outputs["predicted_expression"]
            if prediction is None:
                raise RuntimeError("Predicted expression is unavailable.")
            expression_loss, expression_metrics = self.expression_loss(
                prediction,
                batch["expression_target"].to(self.device, non_blocking=True),
            )
            total_loss = total_loss + float(stage_config["expression_weight"]) * expression_loss
            metrics.update(expression_metrics)
        metrics["loss"] = float(total_loss.detach().item())
        return total_loss, metrics

    def _run_epoch(
        self,
        *,
        stage: int,
        training: bool,
        optimizer: optim.Optimizer | None,
    ) -> dict[str, float]:
        loader = self.train_loader if training else self.validation_loader
        self.model.train(training)
        totals: dict[str, float] = {}
        number_of_batches = 0
        predictions: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        description = f"{'train' if training else 'validation'}-stage{stage}"
        iterator = tqdm(loader, desc=description, leave=False)
        gradient_context = torch.enable_grad() if training else torch.no_grad()
        with gradient_context:
            for batch in iterator:
                if training:
                    if optimizer is None:
                        raise RuntimeError("Training requires an optimizer.")
                    optimizer.zero_grad(set_to_none=True)
                autocast_context = (
                    torch.autocast(device_type="cuda", dtype=torch.float16) if self.use_amp else nullcontext()
                )
                with autocast_context:
                    outputs = self.model(
                        dna_sequence=batch["dna_sequence"].to(
                            self.device,
                            non_blocking=True,
                        ),
                        epigenomic_signals=batch["epigenomic_signals"].to(
                            self.device,
                            non_blocking=True,
                        ),
                        genomic_coordinates=batch["genomic_coordinates"].to(
                            self.device,
                            non_blocking=True,
                        ),
                        promoter_mask=batch["promoter_mask"].to(
                            self.device,
                            non_blocking=True,
                        ),
                        node_mask=batch["node_mask"].to(
                            self.device,
                            non_blocking=True,
                        ),
                        return_intermediates=True,
                    )
                    loss, batch_metrics = self._calculate_loss(
                        outputs,
                        batch,
                        stage,
                    )
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite loss during {description}. Check source data and numerical settings."
                    )
                if training:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        float(
                            self.training_config.get(
                                "gradient_clip_norm",
                                1.0,
                            )
                        ),
                    )
                    self.scaler.step(optimizer)
                    self.scaler.update()

                for name, value in batch_metrics.items():
                    totals[name] = totals.get(name, 0.0) + float(value)
                number_of_batches += 1
                if not training and stage == 2:
                    prediction = outputs["predicted_expression"]
                    if prediction is not None:
                        predictions.append(prediction.detach().cpu().flatten())
                        targets.append(batch["expression_target"].cpu().flatten())
                iterator.set_postfix(loss=f"{batch_metrics['loss']:.4f}")

        if number_of_batches == 0:
            raise RuntimeError(f"No batches were produced by {description}.")
        metrics = {name: value / number_of_batches for name, value in totals.items()}
        if predictions:
            metrics["global_expression_r"] = float(
                pearson_correlation(
                    torch.cat(predictions),
                    torch.cat(targets),
                ).item()
            )
        return metrics

    def _checkpoint_path(self, stage: int, kind: str) -> Path:
        if kind == "best":
            suffix = "physics" if stage == 1 else "expression"
            return self.output_dir / f"best_stage{stage}_{suffix}.pt"
        return self.output_dir / f"last_stage{stage}.pt"

    def _save_checkpoint(
        self,
        path: Path,
        *,
        stage: int,
        epoch: int,
        optimizer: optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        best_score: float,
        metrics: dict[str, Any],
    ) -> None:
        checkpoint = {
            "sparta_version": __version__,
            "training_state_version": 1,
            "configuration_sha256": self.config.digest(),
            "input_metadata": self.input_metadata,
            "stage": stage,
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else None,
                "device_type": self.device.type,
            },
            "data_generator_states": {name: generator.get_state() for name, generator in self.data_generators.items()},
            "worker_generator_states": {name: loader.generator.get_state() for name, loader in self._loaders().items()},
            "best_score": best_score,
            "metrics": metrics,
        }
        temporary = path.with_name(f"{path.name}.part")
        try:
            torch.save(checkpoint, temporary)
            temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _loaders(self) -> dict[str, Any]:
        return {"train": self.train_loader, "validation": self.validation_loader, "test": self.test_loader}

    def _restore_training_state(self, checkpoint: dict[str, Any]) -> None:
        required = ("rng_state", "scaler_state", "data_generator_states", "worker_generator_states")
        if checkpoint.get("training_state_version") != 1 or any(key not in checkpoint for key in required):
            raise ValueError("Cannot resume a legacy checkpoint without complete training state. Start a new run.")
        state = checkpoint["rng_state"]
        if state["device_type"] != self.device.type:
            raise ValueError("Exact checkpoint resume requires the same device type.")
        cuda_states = state["cuda"]
        if cuda_states is not None and len(cuda_states) != torch.cuda.device_count():
            raise ValueError("Exact checkpoint resume requires the same CUDA device count.")
        self.scaler.load_state_dict(checkpoint["scaler_state"])
        for name, generator in self.data_generators.items():
            generator.set_state(checkpoint["data_generator_states"][name].cpu())
        for name, loader in self._loaders().items():
            loader.generator.set_state(checkpoint["worker_generator_states"][name].cpu())
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"].cpu())
        if cuda_states is not None:
            torch.cuda.set_rng_state_all([value.cpu() for value in cuda_states])

    def _load_model_checkpoint(self, path: Path) -> dict[str, Any]:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if checkpoint.get("input_metadata") != self.input_metadata:
            raise ValueError(
                f"Checkpoint input metadata/track order is missing or incompatible: {path}. "
                "Use matching data and a checkpoint containing verified input metadata."
            )
        self.model.load_state_dict(checkpoint["model_state"])
        return checkpoint

    def _append_metrics(self, record: dict[str, Any]) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            json.dump(record, handle, sort_keys=True)
            handle.write("\n")

    def train_stage(self, stage: int) -> None:
        """Train or resume one stage and retain its best validation checkpoint."""

        stage_config = self.training_config[f"stage{stage}"]
        epochs = int(stage_config["epochs"])
        if epochs <= 0:
            LOGGER.info("Skipping stage %d because epochs is zero.", stage)
            return
        self._set_trainable_modules(stage)
        optimizer = self._optimizer(stage)
        scheduler = self._scheduler(optimizer, epochs)
        start_epoch = 1
        best_score = -math.inf
        last_path = self._checkpoint_path(stage, "last")
        if bool(self.training_config.get("resume", True)) and last_path.is_file():
            checkpoint = self._load_model_checkpoint(last_path)
            if checkpoint.get("configuration_sha256") != self.config.digest():
                raise ValueError(f"Cannot resume stage {stage}: configuration checksum differs from {last_path}")
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            self._restore_training_state(checkpoint)
            start_epoch = int(checkpoint["epoch"]) + 1
            best_score = float(checkpoint.get("best_score", -math.inf))
            LOGGER.info("Resuming stage %d at epoch %d.", stage, start_epoch)
        self.best_scores[f"stage{stage}"] = best_score

        monitor = "loss" if stage == 1 else "global_expression_r"
        for epoch in range(start_epoch, epochs + 1):
            training_metrics = self._run_epoch(
                stage=stage,
                training=True,
                optimizer=optimizer,
            )
            scheduler.step()
            validation_metrics: dict[str, float] | None = None
            if epoch % int(self.training_config.get("validation_interval", 1)) == 0:
                validation_metrics = self._run_epoch(
                    stage=stage,
                    training=False,
                    optimizer=None,
                )
                monitored_value = validation_metrics.get(
                    monitor,
                    math.inf if stage == 1 else -math.inf,
                )
                score = -monitored_value if stage == 1 else monitored_value
                if score > best_score:
                    best_score = score
                    self.best_scores[f"stage{stage}"] = score
                    self._save_checkpoint(
                        self._checkpoint_path(stage, "best"),
                        stage=stage,
                        epoch=epoch,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        best_score=best_score,
                        metrics=validation_metrics,
                    )
                    LOGGER.info(
                        "Saved best stage %d checkpoint at epoch %d (%s=%.5f).",
                        stage,
                        epoch,
                        monitor,
                        monitored_value,
                    )
            record = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "stage": stage,
                "epoch": epoch,
                "learning_rates": [float(group["lr"]) for group in optimizer.param_groups],
                "training": training_metrics,
                "validation": validation_metrics,
            }
            self._append_metrics(record)
            self._save_checkpoint(
                last_path,
                stage=stage,
                epoch=epoch,
                optimizer=optimizer,
                scheduler=scheduler,
                best_score=best_score,
                metrics=record,
            )
            LOGGER.info(
                "Stage %d epoch %d/%d: training loss %.5f%s",
                stage,
                epoch,
                epochs,
                training_metrics["loss"],
                (
                    f", validation {monitor}={validation_metrics.get(monitor, float('nan')):.5f}"
                    if validation_metrics is not None
                    else ""
                ),
            )

    def evaluate_test_set(self) -> dict[str, float | int | str]:
        """Evaluate expression prediction on held-out test chromosomes."""

        self.model.eval()
        predictions: list[float] = []
        targets: list[float] = []
        metadata: list[tuple[str, str, str, str]] = []
        with torch.inference_mode():
            for batch in tqdm(self.test_loader, desc="held-out-test", leave=False):
                outputs = self.model(
                    dna_sequence=batch["dna_sequence"].to(
                        self.device,
                        non_blocking=True,
                    ),
                    epigenomic_signals=batch["epigenomic_signals"].to(
                        self.device,
                        non_blocking=True,
                    ),
                    genomic_coordinates=batch["genomic_coordinates"].to(
                        self.device,
                        non_blocking=True,
                    ),
                    promoter_mask=batch["promoter_mask"].to(
                        self.device,
                        non_blocking=True,
                    ),
                    node_mask=batch["node_mask"].to(
                        self.device,
                        non_blocking=True,
                    ),
                    return_intermediates=False,
                )
                predicted = outputs["predicted_expression"]
                if predicted is None:
                    raise RuntimeError("Predicted expression is unavailable.")
                predictions.extend(predicted.detach().cpu().flatten().tolist())
                targets.extend(batch["expression_target"].flatten().tolist())
                metadata.extend(
                    zip(
                        batch["gene_id"],
                        batch["gene_name"],
                        batch["chromosome"],
                        batch["strand"],
                        strict=True,
                    )
                )

        prediction_tensor = torch.tensor(predictions, dtype=torch.float32)
        target_tensor = torch.tensor(targets, dtype=torch.float32)
        correlation = float(pearson_correlation(prediction_tensor, target_tensor).item())
        smooth_l1 = float(
            torch.nn.functional.smooth_l1_loss(
                prediction_tensor,
                target_tensor,
            ).item()
        )
        output_path = self.output_dir / "test_predictions.tsv"
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(
                [
                    "gene_id",
                    "gene_name",
                    "chromosome",
                    "strand",
                    "observed_expression",
                    "predicted_expression",
                ]
            )
            for values, observed, predicted in zip(
                metadata,
                targets,
                predictions,
                strict=True,
            ):
                writer.writerow([*values, observed, predicted])
        return {
            "genes": len(predictions),
            "pearson_r": correlation,
            "smooth_l1": smooth_l1,
            "predictions": str(output_path),
        }

    def _write_run_metadata(self) -> None:
        resolved_config = self.config.serializable_copy(resolve_paths=True)
        with (self.output_dir / "resolved_config.yaml").open(
            "w",
            encoding="utf-8",
        ) as handle:
            yaml.safe_dump(
                resolved_config,
                handle,
                sort_keys=False,
                allow_unicode=False,
            )
        metadata = {
            "sparta_version": __version__,
            "configuration_sha256": self.config.digest(),
            "configuration_source": str(self.config.source_path),
            "device": str(self.device),
            "cuda_available": torch.cuda.is_available(),
            "torch_version": torch.__version__,
            "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
            "started_utc": datetime.now(timezone.utc).isoformat(),
        }
        with (self.output_dir / "run_metadata.json").open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
            handle.write("\n")

    def run(self, force: bool = False) -> dict[str, Any]:
        """Run both training stages and final held-out evaluation."""

        if force:
            reset_paths = [
                self.metrics_path,
                self._checkpoint_path(1, "last"),
                self._checkpoint_path(1, "best"),
                self._checkpoint_path(2, "last"),
                self._checkpoint_path(2, "best"),
                self.output_dir / "test_predictions.tsv",
            ]
            for path in reset_paths:
                if path.exists():
                    path.unlink()
        self._write_run_metadata()
        self.train_stage(1)
        best_stage1 = self._checkpoint_path(1, "best")
        if best_stage1.is_file():
            self._load_model_checkpoint(best_stage1)
        self.train_stage(2)
        best_stage2 = self._checkpoint_path(2, "best")
        if best_stage2.is_file():
            self._load_model_checkpoint(best_stage2)
        test_metrics = self.evaluate_test_set()
        LOGGER.info(
            "Held-out test expression Pearson r: %.5f",
            test_metrics["pearson_r"],
        )
        return {
            "sparta_version": __version__,
            "configuration_sha256": self.config.digest(),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "best_scores": self.best_scores,
            "test": test_metrics,
        }

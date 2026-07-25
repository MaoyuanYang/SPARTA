"""End-to-end smoke test for HDF5 loading and two-stage training."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
h5py = pytest.importorskip("h5py")
np = pytest.importorskip("numpy")

from sparta.config import SpartaConfig  # noqa: E402
from sparta.data.hdf5_dataset import SpartaHDF5Dataset  # noqa: E402
from sparta.training.trainer import SpartaTrainer  # noqa: E402


def _write_chromosome(path: Path, chromosome: str, offset: float) -> None:
    number_of_genes = 2
    number_of_nodes = 4
    window_bp = 32
    generator = np.random.default_rng(10 + int(offset))
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "dna_sequence",
            data=generator.integers(
                0,
                2,
                size=(number_of_genes, number_of_nodes, 4, window_bp),
                dtype=np.uint8,
            ),
        )
        handle.create_dataset(
            "epigenomic_signals",
            data=generator.random(
                (number_of_genes, number_of_nodes, 2, window_bp),
                dtype=np.float32,
            ),
        )
        hic_scale = generator.random(
            (number_of_genes, number_of_nodes, number_of_nodes),
            dtype=np.float32,
        )
        hic_oe = generator.random(
            (number_of_genes, number_of_nodes, number_of_nodes),
            dtype=np.float32,
        )
        handle.create_dataset("hic_scale", data=hic_scale + hic_scale.transpose(0, 2, 1))
        handle.create_dataset("hic_oe", data=hic_oe + hic_oe.transpose(0, 2, 1))
        handle.create_dataset(
            "genomic_coordinates",
            data=np.asarray(
                [
                    [1000, 2000, 4000, 0],
                    [2000, 3000, 5000, 0],
                ],
                dtype=np.int64,
            ),
        )
        handle.create_dataset(
            "expression_target",
            data=np.asarray([offset, offset + 1.0], dtype=np.float32),
        )
        handle.create_dataset(
            "number_of_nodes",
            data=np.asarray([3, 3], dtype=np.int16),
        )
        handle.create_dataset(
            "promoter_mask",
            data=np.asarray(
                [
                    [False, True, False, False],
                    [False, True, False, False],
                ]
            ),
        )
        handle.create_dataset(
            "node_mask",
            data=np.asarray(
                [
                    [True, True, True, False],
                    [True, True, True, False],
                ]
            ),
        )
        string_dtype = h5py.string_dtype(encoding="utf-8")
        handle.create_dataset(
            "gene_id",
            data=np.asarray([f"{chromosome}_gene1", f"{chromosome}_gene2"], dtype=object),
            dtype=string_dtype,
        )
        handle.create_dataset(
            "gene_name",
            data=np.asarray(["GENE1", "GENE2"], dtype=object),
            dtype=string_dtype,
        )
        handle.create_dataset(
            "chromosome",
            data=np.asarray([chromosome, chromosome], dtype=object),
            dtype=string_dtype,
        )
        handle.create_dataset(
            "strand",
            data=np.asarray(["+", "-"], dtype=object),
            dtype=string_dtype,
        )
        handle.attrs["max_nodes"] = number_of_nodes
        handle.attrs["window_bp"] = window_bp
        handle.attrs["number_of_epigenomic_tracks"] = 2
        handle.attrs["epigenomic_track_names"] = json.dumps(["ATAC", "CTCF"])


def test_two_stage_training_smoke(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    training_dir = tmp_path / "training"
    dataset_dir.mkdir()
    _write_chromosome(dataset_dir / "chr1.h5", "chr1", 0.0)
    _write_chromosome(dataset_dir / "chr16.h5", "chr16", 1.0)
    _write_chromosome(dataset_dir / "chr8.h5", "chr8", 2.0)

    dataset = SpartaHDF5Dataset(dataset_dir / "chr1.h5", return_metadata=True)
    sample = dataset[0]
    assert sample["genomic_coordinates"].tolist() == [-1000.0, 0.0, 2000.0, 0.0]
    assert sample["gene_id"] == "chr1_gene1"
    dataset.close()

    values = {
        "outputs": {
            "dataset_dir": str(dataset_dir),
            "training_dir": str(training_dir),
        },
        "model": {
            "sequence_channels": 4,
            "hidden_dim": 32,
            "physics_layers": 1,
            "physics_heads": 4,
            "dropout": 0.0,
            "insulation_projection_dim": 8,
            "initial_distance_reference_bp": 10000.0,
            "initial_distance_decay": 1.0,
            "initial_temperature": 1.0,
            "learnable_temperature": True,
        },
        "training": {
            "seed": 7,
            "device": "cpu",
            "batch_size": 2,
            "num_workers": 0,
            "use_amp": False,
            "gradient_clip_norm": 1.0,
            "validation_interval": 1,
            "resume": True,
            "selected_epigenomic_tracks": None,
            "train_chromosomes": ["chr1"],
            "validation_chromosomes": ["chr16"],
            "test_chromosomes": ["chr8"],
            "stage1": {
                "epochs": 1,
                "learning_rate": 0.0001,
                "weight_decay": 0.0,
                "stratified_hic_weight": 1.0,
                "global_hic_weight": 1.0,
            },
            "stage2": {
                "epochs": 1,
                "backbone_learning_rate": 0.00001,
                "readout_learning_rate": 0.0001,
                "weight_decay": 0.01,
                "stratified_hic_weight": 0.03,
                "global_hic_weight": 0.03,
                "expression_weight": 1.0,
            },
            "scheduler": {
                "warmup_fraction": 0.1,
                "warmup_start_factor": 0.01,
                "minimum_learning_rate": 0.000001,
            },
            "hic_loss": {
                "distance_bins_bp": [20000, 100000],
                "minimum_pairs_per_bin": 1,
                "global_minimum_pairs": 1,
            },
        },
    }
    config = SpartaConfig(source_path=tmp_path / "config.yaml", values=values)
    summary = SpartaTrainer(config).run(force=True)

    assert summary["test"]["genes"] == 2
    assert (training_dir / "best_stage1_physics.pt").is_file()
    assert (training_dir / "best_stage2_expression.pt").is_file()
    assert (training_dir / "test_predictions.tsv").is_file()

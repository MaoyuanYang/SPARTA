"""Regression coverage for data precision, masking, and training continuity."""

from __future__ import annotations

import copy
import gzip
import hashlib
import importlib
import json
import random
import sys
import types
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from scipy import sparse

from sparta import pipeline
from sparta.config import SpartaConfig, load_config
from sparta.data.hdf5_dataset import SpartaHDF5Dataset
from sparta.models import SpartaModel
from sparta.training.trainer import SpartaTrainer


def _write_hdf5(path: Path, **overrides) -> None:
    generator = np.random.default_rng(17)
    with h5py.File(path, "w") as handle:
        arrays = {
            "dna_sequence": generator.random((4, 4, 4, 32), dtype=np.float32),
            "epigenomic_signals": generator.random((4, 4, 2, 32), dtype=np.float32),
            "hic_scale": generator.random((4, 4, 4), dtype=np.float32),
            "hic_oe": generator.random((4, 4, 4), dtype=np.float32),
            "genomic_coordinates": np.tile([1000, 5000, 30000, 0], (4, 1)),
            "promoter_mask": np.tile([False, True, False, False], (4, 1)),
            "node_mask": np.tile([True, True, True, False], (4, 1)),
            "expression_target": np.arange(4, dtype=np.float32),
        }
        for name, values in arrays.items():
            handle.create_dataset(name, data=values)
        for key in ("gene_id", "gene_name", "chromosome", "strand"):
            handle.create_dataset(key, data=[f"{key}_{i}" for i in range(4)], dtype=h5py.string_dtype())
        attributes = {
            "schema_version": "1.0",
            "max_nodes": 4,
            "window_bp": 32,
            "number_of_epigenomic_tracks": 2,
            "epigenomic_track_names": '["ATAC", "CTCF"]',
            "cell_line": "K562",
            "genome_assembly": "hg38",
            "regulatory_window_bp": 4000000,
            "hic_resolution_bp": 5000,
            "hic_normalization": "SCALE",
            "epigenomic_log1p": True,
            "expression_log2p1": True,
            "expression_zscore": True,
            "expression_mean": 2.0,
            "expression_std": 1.0,
            "configuration_sha256": "fixture",
        }
        handle.attrs.update(attributes | overrides)


def _config(tmp_path: Path, name: str) -> SpartaConfig:
    values = copy.deepcopy(load_config(Path(__file__).parents[1] / "configs/k562.example.yaml").values)
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir(exist_ok=True)
    for chromosome in ("chr1", "chr16", "chr8"):
        path = dataset_dir / f"{chromosome}.h5"
        if not path.exists():
            _write_hdf5(path)
    values["outputs"]["dataset_dir"] = str(dataset_dir)
    values["outputs"]["training_dir"] = str(tmp_path / name)
    values["epigenomic_tracks"] = {key: {"path": key + ".bw"} for key in ("ATAC", "CTCF")}
    values["model"].update(hidden_dim=32, physics_layers=2, physics_heads=4, dropout=0.1)
    values["training"].update(
        device="cpu",
        use_amp=False,
        num_workers=0,
        batch_size=2,
        train_chromosomes=["chr1"],
        validation_chromosomes=["chr16"],
        test_chromosomes=["chr8"],
    )
    values["training"]["stage1"]["epochs"] = 2
    values["training"]["stage2"]["epochs"] = 2
    values["training"]["hic_loss"].update(minimum_pairs_per_bin=1, global_minimum_pairs=1)
    return SpartaConfig(tmp_path / "config.yaml", values)


@pytest.fixture
def builder(monkeypatch):
    """Replace optional genomic file readers, not SPARTA preprocessing."""

    class Fasta:
        def __init__(self, *args):
            pass

        def __contains__(self, key):
            return True

        def __getitem__(self, key):
            return self

        seq = "A" * 32

        def close(self):
            pass

    class BigWig:
        def chroms(self):
            return {"chr1": 32}

        def values(self, *args, **kwargs):
            return np.full(32, 70000, dtype=np.float32)

        def close(self):
            pass

    monkeypatch.setitem(sys.modules, "pyfaidx", types.SimpleNamespace(Fasta=Fasta))
    monkeypatch.setitem(sys.modules, "pyBigWig", types.SimpleNamespace(open=lambda _: BigWig()))
    module = importlib.import_module("sparta.data.builder")
    monkeypatch.setattr(module, "Fasta", Fasta)
    monkeypatch.setattr(module, "pyBigWig", types.SimpleNamespace(open=lambda _: BigWig()))
    return module


def _peak_config(path: Path, peak_format: str | None = None, blacklist: Path | None = None) -> SpartaConfig:
    if blacklist is None:
        blacklist = path.parent / "empty_blacklist.bed"
        blacklist.touch()
    peaks = {"path": str(path)}
    if peak_format:
        peaks["format"] = peak_format
    resources = {"atac_peaks": peaks, "atac_blacklist": {"path": str(blacklist)}}
    return SpartaConfig(path.parent / "config.yaml", {"resources": resources})


def test_narrowpeak_uses_summits_before_deduplicating_centers(builder, tmp_path):
    path = tmp_path / "peaks.bed.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(
            "chr1\t817227\t818384\ta\t20\t.\t1\t1\t1\t267\n"
            "chr1\t817227\t818384\tb\t20\t.\t1\t1\t1\t861\n"
            "chr1\t817494\t817520\tc\t20\t.\t1\t1\t1\t0\n"
            "chr1\t100\t121\td\t20\t.\t1\t1\t1\t-1\n"
            "chr2\t817494\t817520\te\t20\t.\t1\t1\t1\t0\n"
        )
    peaks = builder._prepare_peak_table(_peak_config(path))
    assert list(zip(peaks["chrom"], peaks["center"], strict=True)) == [
        ("chr1", 110),
        ("chr1", 817494),
        ("chr1", 818088),
        ("chr2", 817494),
    ]


@pytest.mark.parametrize(
    "format_override,extra_columns",
    [
        (None, ""),
        ("bed", "\t+\t10\t30\t0\t2"),
        (None, "\t+\t10\t30\t0\t2\t4,4\t0,16"),
    ],
    ids=["processed-bed5", "explicit-bed10", "bed12"],
)
def test_bed_peak_windows_keep_their_interval_midpoints(builder, tmp_path, format_override, extra_columns):
    path = tmp_path / "peaks.bed"
    path.write_text(f"chr1\t10\t30\tp\t20{extra_columns}\n", encoding="utf-8")
    assert builder._prepare_peak_table(_peak_config(path, format_override))["center"].tolist() == [20]


def test_narrowpeak_spacing_uses_signal_per_chromosome_and_keeps_exactly_512_bp(builder, tmp_path):
    path = tmp_path / "peaks.narrowPeak"
    path.write_text(
        "chr1\t1511\t1513\tbetween\t999\t.\t20\t1\t1\t0\n"
        "chr1\t1000\t1002\tstrongest\t5\t.\t30\t1\t1\t0\n"
        "chr1\t1512\t1514\tboundary\t3\t.\t10\t1\t1\t0\n"
        "chr2\t1511\t1513\tother_chrom\t1\t.\t20\t1\t1\t0\n",
        encoding="utf-8",
    )
    peaks = builder._prepare_peak_table(_peak_config(path))
    assert list(peaks[["chrom", "center", "name", "score"]].itertuples(index=False, name=None)) == [
        ("chr1", 1000, "strongest", 5),
        ("chr1", 1512, "boundary", 3),
        ("chr2", 1511, "other_chrom", 1),
    ]


def test_bed_spacing_prefers_score_then_coordinate_then_input_order(builder, tmp_path):
    path = tmp_path / "peaks.bed"
    path.write_text(
        "chr1\t1000\t1002\tweak\t5\n"
        "chr1\t1200\t1202\tstrong\t10\n"
        "chr1\t3100\t3102\tright\t30\n"
        "chr1\t3000\t3002\tfirst_left\t30\n"
        "chr1\t3000\t3002\tsecond_left\t30\n",
        encoding="utf-8",
    )
    peaks = builder._prepare_peak_table(_peak_config(path))
    assert list(peaks[["center", "name"]].itertuples(index=False, name=None)) == [
        (1201, "strong"),
        (3001, "first_left"),
    ]


@pytest.mark.parametrize("offset", ["-2", "20", "1.5", "."])
def test_narrowpeak_rejects_invalid_summit_offset(builder, tmp_path, offset):
    path = tmp_path / "peaks.bed"
    path.write_text(f"chr1\t10\t30\tp\t20\t.\t1\t1\t1\t{offset}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="offset"):
        builder._prepare_peak_table(_peak_config(path, "narrowPeak"))


def test_blacklist_excludes_summits_before_spacing(builder, tmp_path):
    path = tmp_path / "peaks.narrowPeak"
    path.write_text(
        "chr1\t900\t1020\tstart\t5\t.\t90\t1\t1\t100\n"
        "chr1\t1190\t1220\tinside\t5\t.\t80\t1\t1\t10\n"
        "chr1\t1390\t1420\tlast_base\t5\t.\t70\t1\t1\t9\n"
        "chr1\t1300\t1520\tend\t5\t.\t1\t1\t1\t100\n"
        "chr2\t900\t1020\tother_chrom\t5\t.\t90\t1\t1\t100\n",
        encoding="utf-8",
    )
    blacklist = tmp_path / "blacklist.bed.gz"
    with gzip.open(blacklist, "wt", encoding="utf-8") as handle:
        handle.write("1\t1250\t1400\n1\t1000\t1300\n1\t1050\t1100\n")
    peaks = builder._prepare_peak_table(_peak_config(path, blacklist=blacklist))
    assert list(peaks[["chrom", "center"]].itertuples(index=False, name=None)) == [("chr1", 1400), ("chr2", 1000)]


@pytest.mark.parametrize("extra", ["", "\t.\t5\t1\t1\t-1"], ids=["bed", "missing-summit"])
def test_blacklist_uses_midpoint_for_interval_centers(builder, tmp_path, extra):
    path = tmp_path / "peaks.bed"
    path.write_text(f"chr1\t1000\t2000\tpeak\t9{extra}\n", encoding="utf-8")
    blacklist = tmp_path / "blacklist.bed"
    blacklist.write_text("chr1\t1500\t1501\n", encoding="utf-8")
    assert builder._prepare_peak_table(_peak_config(path, blacklist=blacklist)).empty


@pytest.mark.parametrize("interval", ["chr1\t20\t10", "chr1\t-1\t10", "chr1\t1.5\t10"])
def test_invalid_blacklist_interval_is_rejected(builder, tmp_path, interval):
    path = tmp_path / "peaks.bed"
    path.write_text("chr1\t10\t30\tp\t20\n", encoding="utf-8")
    blacklist = tmp_path / "blacklist.bed"
    blacklist.write_text(interval + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        builder._prepare_peak_table(_peak_config(path, blacklist=blacklist))


def test_building_peaks_requires_the_blacklist(builder, tmp_path):
    path = tmp_path / "peaks.bed"
    path.write_text("chr1\t10\t30\tp\t20\n", encoding="utf-8")
    config = _peak_config(path, blacklist=tmp_path / "missing.bed")
    with pytest.raises(FileNotFoundError):
        builder._prepare_peak_table(config)
    del config.values["resources"]["atac_blacklist"]
    with pytest.raises(ValueError, match="atac_blacklist"):
        builder._prepare_peak_table(config)


def test_large_finite_values_survive_cache_and_hic_extraction(builder, monkeypatch):
    contacts = sparse.csr_matrix(np.array([[70000, 2], [2, 3]], dtype=np.float32))
    monkeypatch.setattr(builder.ChromosomeDataCache, "_load_hic_matrices", lambda *args: (contacts, contacts))
    cache = builder.ChromosomeDataCache(
        "chr1", Path("fixture.fa"), {"ATAC": Path("fixture.bw")}, Path("fixture.hic"), 16, "SCALE"
    )
    signals = cache.epigenomic_window("ATAC", 0, 16)
    np.testing.assert_array_equal(signals, np.full(16, 70000, dtype=np.float32))
    assert np.isfinite(np.log1p(signals)).all()
    hic, _ = cache.hic_submatrices(np.array([0, 1, 16]))
    assert hic[0, 1] == 70000
    assert np.isfinite(hic).all()


def test_hdf5_storage_preserves_large_finite_values(builder, tmp_path):
    with h5py.File(tmp_path / "large.h5", "w") as handle:
        datasets = builder._create_hdf5_datasets(handle, 1, 4, 2, 32, {})
        for name in ("epigenomic_signals", "hic_scale", "hic_oe"):
            datasets[name][:] = 70000
            assert np.isfinite(datasets[name][:]).all()
            assert (datasets[name][:] == 70000).all()


def test_padding_does_not_change_valid_encoding_or_batchnorm():
    torch.manual_seed(11)
    model = SpartaModel(epigenomic_channels=2, hidden_dim=32, physics_layers=2, physics_heads=4, dropout=0.0)
    other = copy.deepcopy(model)
    dna = torch.rand(2, 3, 4, 32, requires_grad=True)
    epi = torch.rand(2, 3, 2, 32, requires_grad=True)
    padded_dna = torch.cat([dna.detach(), torch.full((2, 2, 4, 32), 100.0)], dim=1).requires_grad_()
    padded_epi = torch.cat([epi.detach(), torch.full((2, 2, 2, 32), 100.0)], dim=1).requires_grad_()
    coords = torch.tensor([[0.0, 10000.0, 30000.0]]).repeat(2, 1)
    promoter = torch.tensor([[False, True, False]]).repeat(2, 1)
    mask = torch.ones(2, 3, dtype=torch.bool)
    out = model(dna, epi, coords, promoter, mask, True)
    padded = other(
        padded_dna,
        padded_epi,
        torch.nn.functional.pad(coords, (0, 2)),
        torch.nn.functional.pad(promoter, (0, 2)),
        torch.nn.functional.pad(mask, (0, 2)),
        True,
    )
    torch.testing.assert_close(out["predicted_expression"], padded["predicted_expression"])
    out["predicted_expression"].sum().backward()
    padded["predicted_expression"].sum().backward()
    torch.testing.assert_close(dna.grad, padded_dna.grad[:, :3])
    assert torch.count_nonzero(padded_dna.grad[:, 3:]) == 0
    for key, value in model.state_dict().items():
        if "running_" in key:
            torch.testing.assert_close(value, other.state_dict()[key], rtol=0, atol=0)


def test_channel_order_changes_digest_but_other_mapping_order_does_not(tmp_path):
    config = _config(tmp_path, "config_test")
    changed = copy.deepcopy(config.values)
    changed["epigenomic_tracks"] = dict(reversed(list(changed["epigenomic_tracks"].items())))
    assert config.digest() != SpartaConfig(config.source_path, changed).digest()
    changed = dict(reversed(list(config.values.items())))
    assert config.digest() == SpartaConfig(config.source_path, changed).digest()


@pytest.mark.parametrize(
    "field,value",
    [
        ("hic_normalization", "KR"),
        ("expression_mean", 10.0),
        ("expression_std", 2.0),
        ("genome_assembly", "hg19"),
        ("cell_line", "other"),
        ("epigenomic_log1p", False),
        ("expression_log2p1", False),
        ("expression_zscore", False),
        ("hic_resolution_bp", 10000),
        ("regulatory_window_bp", 2000000),
    ],
)
def test_rejects_mixed_preprocessing(tmp_path, field, value):
    first, second = tmp_path / "first.h5", tmp_path / "second.h5"
    _write_hdf5(first)
    _write_hdf5(second, **{field: value})
    with pytest.raises(ValueError, match=field):
        SpartaHDF5Dataset([first, second])


def test_missing_preprocessing_metadata_is_not_silently_accepted(tmp_path):
    path = tmp_path / "missing.h5"
    _write_hdf5(path)
    with h5py.File(path, "a") as handle:
        del handle.attrs["hic_normalization"]
    with pytest.raises(ValueError, match="hic_normalization"):
        SpartaHDF5Dataset(path)


def test_training_only_config_changes_do_not_make_hdf5_incompatible(tmp_path):
    first, second = tmp_path / "first.h5", tmp_path / "second.h5"
    _write_hdf5(first, configuration_sha256="learning_rate_one")
    _write_hdf5(second, configuration_sha256="learning_rate_two")
    dataset = SpartaHDF5Dataset([first, second])
    assert len(dataset) == 8
    dataset.close()


def test_validation_split_must_match_training_preprocessing(tmp_path):
    config = _config(tmp_path, "mixed_splits")
    _write_hdf5(config.output_path("dataset_dir") / "chr16.h5", hic_normalization="KR")
    with pytest.raises(ValueError, match="hic_normalization"):
        SpartaTrainer(config)


@pytest.mark.parametrize("field", ["epigenomic_signals", "hic_scale", "hic_oe", "expression_target"])
def test_nonfinite_stored_values_fail_before_training(tmp_path, field):
    path = tmp_path / "invalid.h5"
    _write_hdf5(path)
    with h5py.File(path, "a") as handle:
        handle[field][0] = float("inf")
    dataset = SpartaHDF5Dataset(path)
    try:
        with pytest.raises(ValueError, match=field):
            dataset[0]
    finally:
        dataset.close()


def _close_trainer(trainer):
    for loader in (trainer.train_loader, trainer.validation_loader, trainer.test_loader):
        iterator = getattr(loader, "_iterator", None)
        if iterator is not None:
            iterator._shutdown_workers()
        loader.dataset.close()


class _Interruption(Exception):
    pass


class _InterruptTrainer(SpartaTrainer):
    stop_stage = 1

    def _run_epoch(self, **kwargs):
        if kwargs["training"] and kwargs["stage"] == self.stop_stage:
            self._audit_calls = getattr(self, "_audit_calls", 0) + 1
            if self._audit_calls == 2:
                raise _Interruption
        return super()._run_epoch(**kwargs)


def _assert_nested_equal(first, second):
    if isinstance(first, torch.Tensor):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            _assert_nested_equal(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for a, b in zip(first, second, strict=True):
            _assert_nested_equal(a, b)
    else:
        assert first == second


@pytest.mark.parametrize("stop_stage,workers", [(1, 0), (2, 0), (1, 1)])
def test_epoch_resume_matches_uninterrupted_training(tmp_path, stop_stage, workers):
    continuous_config = _config(tmp_path, "continuous")
    interrupted_config = _config(tmp_path, "interrupted")
    for config in (continuous_config, interrupted_config):
        config.values["training"]["num_workers"] = workers
    continuous = SpartaTrainer(continuous_config)
    try:
        continuous.run()
    finally:
        _close_trainer(continuous)
    interrupted = _InterruptTrainer(interrupted_config)
    interrupted.stop_stage = stop_stage
    try:
        with pytest.raises(_Interruption):
            interrupted.run()
    finally:
        _close_trainer(interrupted)
    resumed = SpartaTrainer(interrupted_config)
    try:
        resumed.run()
    finally:
        _close_trainer(resumed)
    for stage in (1, 2):
        first = torch.load(continuous._checkpoint_path(stage, "last"), weights_only=False)
        second = torch.load(resumed._checkpoint_path(stage, "last"), weights_only=False)
        for key in ("model_state", "optimizer_state", "scheduler_state", "scaler_state"):
            _assert_nested_equal(first[key], second[key])
    assert (continuous.output_dir / "test_predictions.tsv").read_bytes() == (
        resumed.output_dir / "test_predictions.tsv"
    ).read_bytes()


def test_checkpoint_rejects_changed_input_channel_order(tmp_path):
    trainer = SpartaTrainer(_config(tmp_path, "checkpoint"))
    try:
        trainer.train_stage(1)
        path = trainer._checkpoint_path(1, "last")
        checkpoint = torch.load(path, weights_only=False)
        assert checkpoint["input_metadata"]["epigenomic_tracks"] == ["ATAC", "CTCF"]
        checkpoint["input_metadata"]["epigenomic_tracks"].reverse()
        altered = tmp_path / "swapped.pt"
        torch.save(checkpoint, altered)
        with pytest.raises(ValueError, match=r"input|track"):
            trainer._load_model_checkpoint(altered)
    finally:
        _close_trainer(trainer)


def test_checkpoint_restores_random_streams_and_active_scaler(tmp_path):
    trainer = SpartaTrainer(_config(tmp_path, "state_roundtrip"))
    trainer.scaler = torch.amp.GradScaler("cpu", init_scale=128.0, growth_interval=1)
    try:
        trainer.train_stage(1)
        checkpoint = torch.load(trainer._checkpoint_path(1, "last"), weights_only=False)
        assert checkpoint["scaler_state"]["scale"] > 128.0
        expected = (random.random(), np.random.rand(), torch.rand(5))
        trainer.scaler = torch.amp.GradScaler("cpu", init_scale=1.0)
        trainer._restore_training_state(checkpoint)
        actual = (random.random(), np.random.rand(), torch.rand(5))
        _assert_nested_equal(actual, expected)
        assert trainer.scaler.state_dict() == checkpoint["scaler_state"]
        del checkpoint["rng_state"]
        with pytest.raises(ValueError, match="legacy checkpoint"):
            trainer._restore_training_state(checkpoint)
    finally:
        _close_trainer(trainer)


def _configure_cache_test_blacklist(config, tmp_path):
    blacklist = tmp_path / "blacklist.bed"
    blacklist.write_bytes(b"chr1\t0\t2\n")
    config.values["resources"]["atac_blacklist"] = {"path": str(blacklist)}
    return hashlib.sha256(blacklist.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "attribute,stale_value",
    [
        ("peak_center_method", None),
        ("min_peak_spacing_bp", None),
        ("min_peak_spacing_bp", 256),
        ("atac_blacklist_sha256", None),
        ("atac_blacklist_sha256", "0" * 64),
    ],
)
def test_peak_metadata_follows_dataset_and_checkpoint_inputs(builder, monkeypatch, tmp_path, attribute, stale_value):
    config = _config(tmp_path, "peak_centers")
    blacklist_digest = _configure_cache_test_blacklist(config, tmp_path)
    config.values["outputs"]["dataset_dir"] = str(tmp_path / "new_data")
    config.values["data"]["chromosomes"] = ["chr1"]
    config.values["dataset"].update(max_nodes=3, window_bp=4, n_jobs=1, hic_resolution_bp=16)
    labels = tmp_path / "labels.tsv"
    labels.write_text(
        "ensembl_id\tgene_symbol\tchrom\tstrand\ttss_0based\ttpm\n"
        "ENSG1\tGENE1\tchr1\t+\t4\t1\nENSG2\tGENE2\tchr1\t+\t5\t2\n",
        encoding="utf-8",
    )
    config.values["outputs"]["gene_labels"] = str(labels)
    peaks = tmp_path / "peaks.bed"
    peaks.write_text("chr1\t8\t24\ta\t1\t.\t1\t1\t1\t11\n", encoding="utf-8")
    config.values["resources"]["atac_peaks"]["path"] = str(peaks)
    contacts = sparse.eye(2, format="csr")
    monkeypatch.setattr(builder.ChromosomeDataCache, "_load_hic_matrices", lambda *args: (contacts, contacts))

    summary = builder.build_hdf5_dataset(config)
    assert summary["peak_center_method"] == "narrowpeak_summit"
    assert summary["min_peak_spacing_bp"] == 512
    assert summary["atac_blacklist_sha256"] == blacklist_digest
    path = config.output_path("dataset_dir") / "chr1.h5"
    with h5py.File(path) as handle:
        assert handle.attrs["peak_center_method"] == "narrowpeak_summit"
        assert handle.attrs["min_peak_spacing_bp"] == 512
        assert handle.attrs["atac_blacklist_sha256"] == blacklist_digest
        assert handle["genomic_coordinates"][0, :2].tolist() == [4, 19]
    assert builder.build_hdf5_dataset(config)["chromosomes"][0]["reused"]

    summit = SpartaHDF5Dataset(path)
    assert summit.input_metadata["preprocessing"]["peak_center_method"] == "narrowpeak_summit"
    assert summit.input_metadata["preprocessing"]["min_peak_spacing_bp"] == 512
    assert summit.input_metadata["preprocessing"]["atac_blacklist_sha256"] == blacklist_digest
    with h5py.File(path, "a") as handle:
        if stale_value is None:
            del handle.attrs[attribute]
        else:
            handle.attrs[attribute] = stale_value
    legacy = SpartaHDF5Dataset(path)
    if stale_value is None:
        assert attribute not in legacy.input_metadata["preprocessing"]
    with pytest.raises(ValueError, match=attribute):
        summit.ensure_compatible(legacy, "legacy test split")
    with pytest.raises(ValueError, match=rf"{attribute}.*--force"):
        builder.build_hdf5_dataset(config)


def test_dataset_stage_detects_legacy_completion_marker(tmp_path):
    config = _config(tmp_path, "legacy_peaks")
    marker = config.output_path("dataset_dir") / "_SUCCESS.json"
    marker.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="predates peak-center tracking"):
        pipeline.run_pipeline(config, stage="dataset")


@pytest.mark.parametrize("spacing", [None, 256, 512])
def test_dataset_stage_checks_spacing_before_reusing_completion_marker(tmp_path, spacing):
    config = _config(tmp_path, "peak_spacing")
    blacklist_digest = _configure_cache_test_blacklist(config, tmp_path)
    marker = config.output_path("dataset_dir") / "_SUCCESS.json"
    summary = {
        "configuration_sha256": config.digest(),
        "peak_center_method": "narrowpeak_summit",
        "atac_blacklist_sha256": blacklist_digest,
    }
    if spacing is not None:
        summary["min_peak_spacing_bp"] = spacing
    marker.write_text(json.dumps(summary), encoding="utf-8")
    if spacing == 512:
        pipeline.run_pipeline(config, stage="dataset")
    else:
        with pytest.raises(ValueError, match=r"min_peak_spacing_bp.*--force"):
            pipeline.run_pipeline(config, stage="dataset")


def test_dataset_stage_checks_blacklist_contents_before_reusing_completion_marker(tmp_path):
    config = _config(tmp_path, "blacklist_contents")
    blacklist_digest = _configure_cache_test_blacklist(config, tmp_path)
    marker = config.output_path("dataset_dir") / "_SUCCESS.json"
    summary = {
        "configuration_sha256": config.digest(),
        "peak_center_method": "narrowpeak_summit",
        "min_peak_spacing_bp": 512,
    }
    marker.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match=r"atac_blacklist_sha256.*--force"):
        pipeline.run_pipeline(config, stage="dataset")
    summary["atac_blacklist_sha256"] = blacklist_digest
    marker.write_text(json.dumps(summary), encoding="utf-8")
    pipeline.run_pipeline(config, stage="dataset")
    config.resource_path("atac_blacklist").write_bytes(b"chr1\t0\t3\n")
    with pytest.raises(ValueError, match=r"atac_blacklist_sha256.*--force"):
        pipeline.run_pipeline(config, stage="dataset")

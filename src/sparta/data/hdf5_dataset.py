"""PyTorch dataset for chromosome-partitioned SPARTA HDF5 files."""

from __future__ import annotations

import bisect
import json
import math
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler

PREPROCESSING_ATTRIBUTES = (
    "schema_version",
    "cell_line",
    "genome_assembly",
    "regulatory_window_bp",
    "hic_resolution_bp",
    "hic_normalization",
    "epigenomic_log1p",
    "expression_log2p1",
    "expression_zscore",
    "expression_mean",
    "expression_std",
)
# Older SPARTA files did not record these fields. Missing and present values
# must still be distinguished when comparing files.
OPTIONAL_PREPROCESSING_ATTRIBUTES = (
    "expression_zscore_ddof",
    "exclude_promoter_overlaps",
    "peak_selection",
)


def _decode(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _metadata_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    return value.decode("utf-8") if isinstance(value, bytes) else value


def _check_metadata(reference: dict[str, Any], actual: dict[str, Any], source: str) -> None:
    differences = sorted(key for key in reference.keys() | actual.keys() if reference.get(key) != actual.get(key))
    if differences:
        raise ValueError(f"Incompatible HDF5 preprocessing attributes in {source}: {', '.join(differences)}")


class SpartaHDF5Dataset(Dataset):
    """Read SPARTA samples while lazily caching one HDF5 handle per worker."""

    def __init__(
        self,
        hdf5_paths: list[str | Path] | str | Path,
        *,
        return_hic: bool = True,
        return_metadata: bool = False,
        coordinate_scale: float = 1.0,
        epigenomic_tracks: list[str] | None = None,
    ) -> None:
        if isinstance(hdf5_paths, (str, Path)):
            hdf5_paths = [hdf5_paths]
        self.paths = [Path(path).resolve() for path in hdf5_paths]
        if not self.paths:
            raise ValueError("At least one HDF5 path is required.")
        missing = [str(path) for path in self.paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"HDF5 files not found: {missing[:3]}")

        self.return_hic = return_hic
        self.return_metadata = return_metadata
        self.coordinate_scale = float(coordinate_scale)
        self._handles: dict[int, h5py.File] = {}
        self.file_lengths: list[int] = []

        first_attributes: dict[str, Any] | None = None
        for path in self.paths:
            with h5py.File(path, "r") as handle:
                self.file_lengths.append(int(handle["expression_target"].shape[0]))
                attributes = {
                    "max_nodes": int(handle.attrs["max_nodes"]),
                    "window_bp": int(handle.attrs["window_bp"]),
                    "number_of_epigenomic_tracks": int(handle.attrs["number_of_epigenomic_tracks"]),
                    "epigenomic_track_names": json.loads(handle.attrs["epigenomic_track_names"]),
                }
                missing_metadata = [key for key in PREPROCESSING_ATTRIBUTES if key not in handle.attrs]
                if missing_metadata:
                    raise ValueError(
                        f"Missing HDF5 preprocessing metadata in {path}: {', '.join(missing_metadata)}. "
                        "Rebuild the dataset or supply verified preprocessing metadata."
                    )
                attributes.update({key: _metadata_value(handle.attrs[key]) for key in PREPROCESSING_ATTRIBUTES})
                attributes.update(
                    {key: _metadata_value(handle.attrs.get(key)) for key in OPTIONAL_PREPROCESSING_ATTRIBUTES}
                )
                # Preserve the exact input metadata recorded in checkpoints built from legacy files.
                for key in ("peak_center_method", "min_peak_spacing_bp"):
                    if key in handle.attrs:
                        attributes[key] = _metadata_value(handle.attrs[key])
                for key in ("expression_mean", "expression_std"):
                    if not math.isfinite(float(attributes[key])):
                        raise ValueError(f"Non-finite HDF5 preprocessing attribute {key} in {path}.")
                if float(attributes["expression_std"]) <= 0:
                    raise ValueError(f"expression_std must be positive in {path}.")
                if first_attributes is None:
                    first_attributes = attributes
                else:
                    _check_metadata(first_attributes, attributes, str(path))

        if first_attributes is None:
            raise ValueError("No HDF5 metadata could be read.")
        self.max_nodes = first_attributes["max_nodes"]
        self.preprocessing_metadata = first_attributes
        self.window_bp = first_attributes["window_bp"]
        self.all_epigenomic_tracks = first_attributes["epigenomic_track_names"]
        self.cumulative_lengths = np.cumsum(self.file_lengths).tolist()

        self.selected_epigenomic_tracks = (
            list(epigenomic_tracks) if epigenomic_tracks is not None else list(self.all_epigenomic_tracks)
        )
        missing_tracks = [name for name in self.selected_epigenomic_tracks if name not in self.all_epigenomic_tracks]
        if missing_tracks:
            raise ValueError(
                f"Requested epigenomic tracks are unavailable: {', '.join(missing_tracks)}. "
                f"Available tracks: {', '.join(self.all_epigenomic_tracks)}"
            )
        self.epigenomic_track_indices = [
            self.all_epigenomic_tracks.index(name) for name in self.selected_epigenomic_tracks
        ]

    @property
    def number_of_epigenomic_tracks(self) -> int:
        return len(self.epigenomic_track_indices)

    @property
    def input_metadata(self) -> dict[str, Any]:
        """Describe the ordered model inputs independently of training settings."""
        return {
            "epigenomic_tracks": list(self.selected_epigenomic_tracks),
            "preprocessing": self.preprocessing_metadata,
            "coordinate_scale": self.coordinate_scale,
        }

    def ensure_compatible(self, other: SpartaHDF5Dataset, source: str) -> None:
        """Check scientific preprocessing across training/validation/test splits."""
        _check_metadata(self.preprocessing_metadata, other.preprocessing_metadata, source)
        if self.selected_epigenomic_tracks != other.selected_epigenomic_tracks:
            raise ValueError(f"Incompatible input epigenomic track order in {source}.")

    def __len__(self) -> int:
        return self.cumulative_lengths[-1]

    def _location(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        file_index = bisect.bisect_right(self.cumulative_lengths, index)
        previous_total = self.cumulative_lengths[file_index - 1] if file_index else 0
        return file_index, index - previous_total

    def _handle(self, file_index: int) -> h5py.File:
        handle = self._handles.get(file_index)
        if handle is None:
            handle = h5py.File(self.paths[file_index], "r")
            self._handles[file_index] = handle
        return handle

    def __getitem__(self, index: int) -> dict[str, Any]:
        file_index, local_index = self._location(index)
        handle = self._handle(file_index)
        dna_sequence = torch.from_numpy(handle["dna_sequence"][local_index].astype(np.float32))
        all_signals = handle["epigenomic_signals"][local_index].astype(np.float32)
        epigenomic_signals = torch.from_numpy(all_signals[:, self.epigenomic_track_indices, :])
        absolute_coordinates = torch.from_numpy(handle["genomic_coordinates"][local_index].astype(np.float64))
        promoter_mask = torch.from_numpy(handle["promoter_mask"][local_index].astype(np.bool_))
        node_mask = torch.from_numpy(handle["node_mask"][local_index].astype(np.bool_))
        valid_promoter = promoter_mask & node_mask
        if valid_promoter.sum().item() != 1:
            raise ValueError(f"Sample {index} must contain exactly one valid promoter node.")
        promoter_index = int(valid_promoter.nonzero(as_tuple=False)[0, 0])
        promoter_coordinate = absolute_coordinates[promoter_index]
        genomic_coordinates = (absolute_coordinates - promoter_coordinate).to(torch.float32) * self.coordinate_scale
        genomic_coordinates = genomic_coordinates * node_mask.to(genomic_coordinates.dtype)

        sample: dict[str, Any] = {
            "dna_sequence": dna_sequence,
            "epigenomic_signals": epigenomic_signals,
            "genomic_coordinates": genomic_coordinates,
            "absolute_coordinates": absolute_coordinates,
            "promoter_mask": promoter_mask,
            "node_mask": node_mask,
            "expression_target": torch.tensor(
                handle["expression_target"][local_index],
                dtype=torch.float32,
            ),
        }
        if self.return_hic:
            sample["hic_scale"] = torch.from_numpy(handle["hic_scale"][local_index].astype(np.float32))
            sample["hic_oe"] = torch.from_numpy(handle["hic_oe"][local_index].astype(np.float32))
        if self.return_metadata:
            for key in ("gene_id", "gene_name", "chromosome", "strand"):
                sample[key] = _decode(handle[key][local_index])
        for key, value in sample.items():
            if isinstance(value, torch.Tensor) and value.is_floating_point() and not torch.isfinite(value).all():
                raise ValueError(
                    f"Non-finite {key} in sample {index} of {self.paths[file_index]}. "
                    "Rebuild the HDF5 dataset from finite source values; casting stored inf/NaN cannot recover them."
                )
        return sample

    def close(self) -> None:
        """Close lazily opened HDF5 handles."""

        for handle in getattr(self, "_handles", {}).values():
            handle.close()
        self._handles = {}

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_handles"] = {}
        return state

    def __del__(self) -> None:
        self.close()

    @staticmethod
    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key in batch[0]:
            values = [sample[key] for sample in batch]
            if isinstance(values[0], torch.Tensor):
                result[key] = torch.stack(values)
            else:
                result[key] = values
        return result


def create_dataloader(
    hdf5_paths: list[str | Path] | str | Path,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    return_hic: bool = True,
    return_metadata: bool = False,
    coordinate_scale: float = 1.0,
    epigenomic_tracks: list[str] | None = None,
    generator: torch.Generator | None = None,
) -> DataLoader:
    """Create a DataLoader with worker-safe HDF5 access."""

    dataset = SpartaHDF5Dataset(
        hdf5_paths,
        return_hic=return_hic,
        return_metadata=return_metadata,
        coordinate_scale=coordinate_scale,
        epigenomic_tracks=epigenomic_tracks,
    )
    # Worker initialization must not advance the model/dropout or sampling RNG.
    # A fresh iterator after resume can otherwise reshuffle the next epoch.
    worker_generator = torch.Generator().manual_seed(generator.initial_seed()) if generator is not None else None
    sampler = RandomSampler(dataset, generator=generator) if shuffle else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        generator=worker_generator,
        num_workers=num_workers,
        collate_fn=SpartaHDF5Dataset.collate,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        drop_last=False,
    )

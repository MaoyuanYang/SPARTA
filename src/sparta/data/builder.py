"""Build chromosome-partitioned SPARTA HDF5 datasets."""

from __future__ import annotations

import gc
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import pyBigWig
from joblib import Parallel, delayed
from pyfaidx import Fasta
from scipy import sparse

from sparta import __version__
from sparta.config import SpartaConfig
from sparta.data import MIN_PEAK_SPACING_BP

LOGGER = logging.getLogger(__name__)


def normalize_chromosome(value: str) -> str:
    """Normalize chromosome names to the `chrN` convention."""

    name = str(value).strip()
    if name.lower().startswith("chr"):
        name = name[3:]
    if name.upper() in {"M", "MT"}:
        return "chrM"
    return f"chr{name}"


def _hic_chromosome_name(chromosome: str, available: list[str]) -> str:
    bare = chromosome.removeprefix("chr")
    for candidate in (chromosome, bare, f"chr{bare}"):
        if candidate in available:
            return candidate
    raise ValueError(f"Chromosome '{chromosome}' is absent from the Hi-C file. Available examples: {available[:10]}")


def _compression_arguments(config: SpartaConfig) -> dict[str, Any]:
    compression = config.get("dataset.compression", "gzip")
    if compression in {None, "none", "None"}:
        return {}
    arguments: dict[str, Any] = {"compression": compression}
    if compression == "gzip":
        arguments["compression_opts"] = int(config.get("dataset.compression_level", 4))
        arguments["shuffle"] = True
    return arguments


class ChromosomeDataCache:
    """In-memory sequence, epigenomic, and sparse Hi-C data for one chromosome."""

    def __init__(
        self,
        chromosome: str,
        reference_fasta: Path,
        epigenomic_tracks: dict[str, Path],
        hic_path: Path,
        hic_resolution_bp: int,
        hic_normalization: str,
    ) -> None:
        self.chromosome = chromosome
        self.hic_resolution_bp = hic_resolution_bp
        self.track_names = list(epigenomic_tracks)

        LOGGER.info("Loading reference sequence for %s.", chromosome)
        fasta = Fasta(str(reference_fasta))
        try:
            if chromosome not in fasta:
                raise ValueError(f"Chromosome '{chromosome}' is absent from {reference_fasta}")
            self.sequence = fasta[chromosome][:].seq.upper().encode("ascii")
        finally:
            fasta.close()
        self.chromosome_length = len(self.sequence)

        self.epigenomic_signals: dict[str, np.ndarray] = {}
        for track_name, track_path in epigenomic_tracks.items():
            LOGGER.info("Loading %s BigWig signal for %s.", track_name, chromosome)
            bigwig = pyBigWig.open(str(track_path))
            try:
                bigwig_length = bigwig.chroms().get(chromosome)
                if bigwig_length is None:
                    raise ValueError(
                        f"Chromosome '{chromosome}' is absent from BigWig track '{track_name}': {track_path}"
                    )
                if bigwig_length != self.chromosome_length:
                    raise ValueError(
                        f"Chromosome length mismatch for track '{track_name}' on {chromosome}: "
                        f"BigWig={bigwig_length}, FASTA={self.chromosome_length}"
                    )
                values = bigwig.values(
                    chromosome,
                    0,
                    bigwig_length,
                    numpy=True,
                )
                self.epigenomic_signals[track_name] = np.nan_to_num(
                    values,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ).astype(np.float32)
            finally:
                bigwig.close()

        LOGGER.info("Loading Hi-C matrices for %s.", chromosome)
        self.hic_scale, self.hic_oe = self._load_hic_matrices(
            hic_path,
            chromosome,
            hic_resolution_bp,
            hic_normalization,
        )

    @staticmethod
    def _records_to_sparse(
        records: list[Any],
        resolution_bp: int,
        number_of_bins: int,
    ) -> sparse.csr_matrix:
        rows: list[int] = []
        columns: list[int] = []
        values: list[float] = []
        for record in records:
            row = int(record.binX) // resolution_bp
            column = int(record.binY) // resolution_bp
            value = float(record.counts)
            if row >= number_of_bins or column >= number_of_bins or not np.isfinite(value):
                continue
            rows.append(row)
            columns.append(column)
            values.append(value)
            if row != column:
                rows.append(column)
                columns.append(row)
                values.append(value)
        return sparse.csr_matrix(
            (values, (rows, columns)),
            shape=(number_of_bins, number_of_bins),
            dtype=np.float32,
        )

    @classmethod
    def _load_hic_matrices(
        cls,
        hic_path: Path,
        chromosome: str,
        resolution_bp: int,
        normalization: str,
    ) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
        try:
            import hicstraw
        except ImportError as error:
            raise ImportError(
                "The hic-straw package is required for dataset construction. "
                "Install SPARTA with its declared dependencies."
            ) from error

        if not hic_path.is_file():
            raise FileNotFoundError(f"Hi-C file not found: {hic_path}")
        try:
            hic = hicstraw.HiCFile(str(hic_path))
            chromosomes = hic.getChromosomes()
            chromosome_names = [entry.name for entry in chromosomes]
            hic_chromosome = _hic_chromosome_name(chromosome, chromosome_names)
            chromosome_record = next(entry for entry in chromosomes if entry.name == hic_chromosome)
            number_of_bins = math.ceil(chromosome_record.length / resolution_bp)
            end_coordinate = number_of_bins * resolution_bp

            observed_zoom = hic.getMatrixZoomData(
                hic_chromosome,
                hic_chromosome,
                "observed",
                normalization,
                "BP",
                resolution_bp,
            )
            oe_zoom = hic.getMatrixZoomData(
                hic_chromosome,
                hic_chromosome,
                "oe",
                normalization,
                "BP",
                resolution_bp,
            )
            observed_records = observed_zoom.getRecords(
                0,
                end_coordinate,
                0,
                end_coordinate,
            )
            oe_records = oe_zoom.getRecords(
                0,
                end_coordinate,
                0,
                end_coordinate,
            )
            if not observed_records:
                raise ValueError(f"No observed Hi-C contacts found for {chromosome} at {resolution_bp} bp.")
            if not oe_records:
                raise ValueError(
                    f"No observed-over-expected Hi-C contacts found for {chromosome} at {resolution_bp} bp."
                )
            return (
                cls._records_to_sparse(observed_records, resolution_bp, number_of_bins),
                cls._records_to_sparse(oe_records, resolution_bp, number_of_bins),
            )
        except Exception as error:
            raise RuntimeError(f"Failed to read Hi-C data for {chromosome} from {hic_path}: {error}") from error

    def sequence_one_hot(self, start: int, end: int) -> np.ndarray:
        """Extract a fixed-length one-hot sequence with zero-padding at boundaries."""

        window_bp = end - start
        if window_bp <= 0:
            raise ValueError(f"Invalid sequence interval: [{start}, {end})")
        clipped_start = max(0, start)
        clipped_end = min(self.chromosome_length, end)
        sequence = self.sequence[clipped_start:clipped_end]
        left_padding = max(0, -start)
        right_padding = max(0, end - self.chromosome_length)
        padded = (b"N" * left_padding) + sequence + (b"N" * right_padding)
        if len(padded) != window_bp:
            raise RuntimeError(f"Sequence extraction returned {len(padded)} bp; expected {window_bp} bp.")
        encoded = np.frombuffer(padded, dtype=np.uint8)
        result = np.zeros((4, window_bp), dtype=np.uint8)
        result[0] = encoded == ord("A")
        result[1] = encoded == ord("C")
        result[2] = encoded == ord("G")
        result[3] = encoded == ord("T")
        return result

    def epigenomic_window(self, track_name: str, start: int, end: int) -> np.ndarray:
        """Extract a fixed-length signal window with zero-padding at boundaries."""

        window_bp = end - start
        result = np.zeros(window_bp, dtype=np.float32)
        clipped_start = max(0, start)
        clipped_end = min(self.chromosome_length, end)
        if clipped_start >= clipped_end:
            return result
        target_start = clipped_start - start
        target_end = target_start + (clipped_end - clipped_start)
        result[target_start:target_end] = self.epigenomic_signals[track_name][clipped_start:clipped_end]
        return result

    def hic_submatrices(self, node_centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Extract SCALE and observed-over-expected node contact matrices."""

        bins = node_centers // self.hic_resolution_bp
        matrix_size = self.hic_scale.shape[0]
        valid = (bins >= 0) & (bins < matrix_size)
        clipped = np.clip(bins, 0, matrix_size - 1)

        def extract(matrix: sparse.csr_matrix) -> np.ndarray:
            values = matrix[clipped, :][:, clipped].toarray().astype(np.float32)
            invalid_indices = np.flatnonzero(~valid)
            if invalid_indices.size:
                values[invalid_indices, :] = 0
                values[:, invalid_indices] = 0
            return np.nan_to_num(values).astype(np.float32)

        return extract(self.hic_scale), extract(self.hic_oe)

    def close(self) -> None:
        """Release chromosome-level arrays."""

        self.sequence = b""
        self.epigenomic_signals.clear()
        self.hic_scale = sparse.csr_matrix((0, 0), dtype=np.float32)
        self.hic_oe = sparse.csr_matrix((0, 0), dtype=np.float32)
        gc.collect()


@dataclass(frozen=True)
class SampleArrays:
    """All fixed-size tensors and metadata for one target gene."""

    dna_sequence: np.ndarray
    epigenomic_signals: np.ndarray
    hic_scale: np.ndarray
    hic_oe: np.ndarray
    genomic_coordinates: np.ndarray
    promoter_mask: np.ndarray
    node_mask: np.ndarray
    expression_target: float
    gene_id: str
    gene_name: str
    chromosome: str
    strand: str
    number_of_nodes: int


def _select_nodes(
    promoter: int,
    peak_centers: np.ndarray,
    peak_scores: np.ndarray,
    max_nodes: int,
    regulatory_radius_bp: int,
    window_bp: int,
    peak_selection: str,
    exclude_promoter_overlaps: bool,
) -> tuple[np.ndarray, int]:
    left_index = np.searchsorted(
        peak_centers,
        promoter - regulatory_radius_bp,
        side="left",
    )
    right_index = np.searchsorted(
        peak_centers,
        promoter + regulatory_radius_bp,
        side="right",
    )
    candidate_centers = peak_centers[left_index:right_index]
    candidate_scores = peak_scores[left_index:right_index]

    if exclude_promoter_overlaps:
        keep = np.abs(candidate_centers - promoter) >= window_bp
        candidate_centers = candidate_centers[keep]
        candidate_scores = candidate_scores[keep]

    maximum_peaks = max_nodes - 1
    if len(candidate_centers) > maximum_peaks:
        if peak_selection == "score":
            selected = np.argsort(candidate_scores, kind="stable")[::-1][:maximum_peaks]
        else:
            selected = np.argsort(
                np.abs(candidate_centers - promoter),
                kind="stable",
            )[:maximum_peaks]
        candidate_centers = candidate_centers[selected]

    nodes = np.concatenate([np.asarray([promoter], dtype=np.int64), candidate_centers.astype(np.int64)])
    order = np.argsort(nodes, kind="stable")
    promoter_index = int(np.flatnonzero(order == 0)[0])
    return nodes[order], promoter_index


def _build_sample(
    row: pd.Series,
    chromosome: str,
    peak_centers: np.ndarray,
    peak_scores: np.ndarray,
    cache: ChromosomeDataCache,
    parameters: dict[str, Any],
) -> SampleArrays:
    max_nodes = int(parameters["max_nodes"])
    window_bp = int(parameters["window_bp"])
    promoter = int(row["_tss"])
    nodes, promoter_index = _select_nodes(
        promoter=promoter,
        peak_centers=peak_centers,
        peak_scores=peak_scores,
        max_nodes=max_nodes,
        regulatory_radius_bp=int(parameters["regulatory_window_bp"]) // 2,
        window_bp=window_bp,
        peak_selection=str(parameters["peak_selection"]),
        exclude_promoter_overlaps=bool(parameters.get("exclude_promoter_overlaps", True)),
    )
    number_of_nodes = len(nodes)
    number_of_tracks = len(cache.track_names)
    half_window = window_bp // 2

    dna_sequence = np.zeros((max_nodes, 4, window_bp), dtype=np.uint8)
    epigenomic_signals = np.zeros(
        (max_nodes, number_of_tracks, window_bp),
        dtype=np.float32,
    )
    for node_index, center in enumerate(nodes):
        start = int(center) - half_window
        end = start + window_bp
        dna_sequence[node_index] = cache.sequence_one_hot(start, end)
        for track_index, track_name in enumerate(cache.track_names):
            values = cache.epigenomic_window(track_name, start, end)
            if parameters.get("epigenomic_log1p", True):
                values = np.log1p(np.maximum(values, 0.0))
            epigenomic_signals[node_index, track_index] = values.astype(np.float32)

    hic_scale = np.zeros((max_nodes, max_nodes), dtype=np.float32)
    hic_oe = np.zeros((max_nodes, max_nodes), dtype=np.float32)
    scale_values, oe_values = cache.hic_submatrices(nodes)
    hic_scale[:number_of_nodes, :number_of_nodes] = scale_values
    hic_oe[:number_of_nodes, :number_of_nodes] = oe_values

    genomic_coordinates = np.zeros(max_nodes, dtype=np.int64)
    genomic_coordinates[:number_of_nodes] = nodes
    promoter_mask = np.zeros(max_nodes, dtype=np.bool_)
    promoter_mask[promoter_index] = True
    node_mask = np.arange(max_nodes) < number_of_nodes
    return SampleArrays(
        dna_sequence=dna_sequence,
        epigenomic_signals=epigenomic_signals,
        hic_scale=hic_scale,
        hic_oe=hic_oe,
        genomic_coordinates=genomic_coordinates,
        promoter_mask=promoter_mask,
        node_mask=node_mask,
        expression_target=float(row["_expression"]),
        gene_id=str(row["ensembl_id"]),
        gene_name=str(row.get("gene_symbol", "")),
        chromosome=chromosome,
        strand=str(row["strand"]),
        number_of_nodes=number_of_nodes,
    )


def _create_hdf5_datasets(
    handle: h5py.File,
    number_of_genes: int,
    max_nodes: int,
    number_of_tracks: int,
    window_bp: int,
    compression: dict[str, Any],
) -> dict[str, h5py.Dataset]:
    string_dtype = h5py.string_dtype(encoding="utf-8")
    return {
        "dna_sequence": handle.create_dataset(
            "dna_sequence",
            shape=(number_of_genes, max_nodes, 4, window_bp),
            dtype=np.uint8,
            chunks=(1, max_nodes, 4, window_bp),
            **compression,
        ),
        "epigenomic_signals": handle.create_dataset(
            "epigenomic_signals",
            shape=(number_of_genes, max_nodes, number_of_tracks, window_bp),
            dtype=np.float32,
            chunks=(1, max_nodes, number_of_tracks, window_bp),
            **compression,
        ),
        "hic_scale": handle.create_dataset(
            "hic_scale",
            shape=(number_of_genes, max_nodes, max_nodes),
            dtype=np.float32,
            chunks=(1, max_nodes, max_nodes),
            **compression,
        ),
        "hic_oe": handle.create_dataset(
            "hic_oe",
            shape=(number_of_genes, max_nodes, max_nodes),
            dtype=np.float32,
            chunks=(1, max_nodes, max_nodes),
            **compression,
        ),
        "genomic_coordinates": handle.create_dataset(
            "genomic_coordinates",
            shape=(number_of_genes, max_nodes),
            dtype=np.int64,
            chunks=(1, max_nodes),
            **compression,
        ),
        "expression_target": handle.create_dataset(
            "expression_target",
            shape=(number_of_genes,),
            dtype=np.float32,
        ),
        "number_of_nodes": handle.create_dataset(
            "number_of_nodes",
            shape=(number_of_genes,),
            dtype=np.int16,
        ),
        "promoter_mask": handle.create_dataset(
            "promoter_mask",
            shape=(number_of_genes, max_nodes),
            dtype=np.bool_,
            chunks=(1, max_nodes),
            **compression,
        ),
        "node_mask": handle.create_dataset(
            "node_mask",
            shape=(number_of_genes, max_nodes),
            dtype=np.bool_,
            chunks=(1, max_nodes),
            **compression,
        ),
        "gene_id": handle.create_dataset(
            "gene_id",
            shape=(number_of_genes,),
            dtype=string_dtype,
        ),
        "gene_name": handle.create_dataset(
            "gene_name",
            shape=(number_of_genes,),
            dtype=string_dtype,
        ),
        "chromosome": handle.create_dataset(
            "chromosome",
            shape=(number_of_genes,),
            dtype=string_dtype,
        ),
        "strand": handle.create_dataset(
            "strand",
            shape=(number_of_genes,),
            dtype=string_dtype,
        ),
    }


def _write_batch(
    datasets: dict[str, h5py.Dataset],
    start: int,
    samples: list[SampleArrays],
) -> None:
    stop = start + len(samples)
    batch_slice = slice(start, stop)
    array_fields = (
        "dna_sequence",
        "epigenomic_signals",
        "hic_scale",
        "hic_oe",
        "genomic_coordinates",
        "promoter_mask",
        "node_mask",
    )
    for field in array_fields:
        datasets[field][batch_slice] = np.stack([getattr(sample, field) for sample in samples])
    datasets["expression_target"][batch_slice] = np.asarray(
        [sample.expression_target for sample in samples],
        dtype=np.float32,
    )
    datasets["number_of_nodes"][batch_slice] = np.asarray(
        [sample.number_of_nodes for sample in samples],
        dtype=np.int16,
    )
    for field in ("gene_id", "gene_name", "chromosome", "strand"):
        datasets[field][batch_slice] = [getattr(sample, field) for sample in samples]


def _write_chromosome(
    config: SpartaConfig,
    chromosome: str,
    genes: pd.DataFrame,
    peaks: pd.DataFrame,
    peak_center_method: str,
    expression_mean: float,
    expression_std: float,
    output_path: Path,
) -> dict[str, Any]:
    parameters = config.section("dataset")
    max_nodes = int(parameters["max_nodes"])
    window_bp = int(parameters["window_bp"])
    number_of_tracks = len(config.section("epigenomic_tracks"))
    compression = _compression_arguments(config)
    temporary_path = output_path.with_name(f"{output_path.name}.part")
    if temporary_path.exists():
        temporary_path.unlink()

    cache = ChromosomeDataCache(
        chromosome=chromosome,
        reference_fasta=config.resource_path("reference_fasta"),
        epigenomic_tracks=config.epigenomic_track_paths(),
        hic_path=config.resource_path("hic"),
        hic_resolution_bp=int(parameters["hic_resolution_bp"]),
        hic_normalization=str(parameters.get("hic_normalization", "SCALE")),
    )
    peak_centers = peaks["center"].to_numpy(dtype=np.int64)
    peak_scores = peaks["score"].to_numpy(dtype=np.float64)
    batch_size = int(parameters.get("batch_size", 16))
    number_of_jobs = int(parameters.get("n_jobs", 1))

    try:
        with h5py.File(temporary_path, "w") as handle:
            datasets = _create_hdf5_datasets(
                handle=handle,
                number_of_genes=len(genes),
                max_nodes=max_nodes,
                number_of_tracks=number_of_tracks,
                window_bp=window_bp,
                compression=compression,
            )
            for start in range(0, len(genes), batch_size):
                stop = min(start + batch_size, len(genes))
                rows = [genes.iloc[index] for index in range(start, stop)]
                if number_of_jobs == 1:
                    samples = [
                        _build_sample(
                            row,
                            chromosome,
                            peak_centers,
                            peak_scores,
                            cache,
                            parameters,
                        )
                        for row in rows
                    ]
                else:
                    samples = Parallel(n_jobs=number_of_jobs, prefer="threads")(
                        delayed(_build_sample)(
                            row,
                            chromosome,
                            peak_centers,
                            peak_scores,
                            cache,
                            parameters,
                        )
                        for row in rows
                    )
                _write_batch(datasets, start, samples)
                LOGGER.info(
                    "%s: wrote genes %d-%d of %d.",
                    chromosome,
                    start + 1,
                    stop,
                    len(genes),
                )

            handle.attrs["schema_version"] = "1.1"
            handle.attrs["sparta_version"] = __version__
            handle.attrs["model_class"] = "SpartaModel"
            handle.attrs["cell_line"] = str(config.get("project.cell_line"))
            handle.attrs["genome_assembly"] = str(config.get("project.genome_assembly"))
            handle.attrs["chromosome"] = chromosome
            handle.attrs["max_nodes"] = max_nodes
            handle.attrs["window_bp"] = window_bp
            handle.attrs["regulatory_window_bp"] = int(parameters["regulatory_window_bp"])
            handle.attrs["number_of_epigenomic_tracks"] = number_of_tracks
            handle.attrs["epigenomic_track_names"] = json.dumps(list(config.section("epigenomic_tracks")))
            handle.attrs["hic_resolution_bp"] = int(parameters["hic_resolution_bp"])
            handle.attrs["hic_normalization"] = str(parameters.get("hic_normalization", "SCALE"))
            handle.attrs["epigenomic_log1p"] = bool(parameters.get("epigenomic_log1p", True))
            handle.attrs["expression_log2p1"] = bool(parameters.get("expression_log2p1", True))
            handle.attrs["expression_zscore"] = bool(parameters.get("expression_zscore", True))
            handle.attrs["expression_mean"] = float(expression_mean)
            handle.attrs["expression_std"] = float(expression_std)
            handle.attrs["expression_zscore_ddof"] = int(parameters.get("expression_zscore_ddof", 1))
            handle.attrs["exclude_promoter_overlaps"] = bool(parameters.get("exclude_promoter_overlaps", True))
            handle.attrs["peak_selection"] = str(parameters["peak_selection"])
            handle.attrs["peak_center_method"] = peak_center_method
            handle.attrs["min_peak_spacing_bp"] = MIN_PEAK_SPACING_BP
            handle.attrs["configuration_sha256"] = config.digest()
        temporary_path.replace(output_path)
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise
    finally:
        cache.close()

    return {
        "chromosome": chromosome,
        "genes": len(genes),
        "regulatory_peaks": len(peaks),
        "output": str(output_path),
    }


def _prepare_gene_table(
    config: SpartaConfig,
) -> tuple[pd.DataFrame, float, float]:
    label_path = config.output_path("gene_labels")
    if not label_path.is_file():
        raise FileNotFoundError(f"Gene expression label table not found: {label_path}. Run the labels stage first.")
    genes = pd.read_csv(label_path, sep="\t")
    required_columns = {"ensembl_id", "gene_symbol", "chrom", "strand", "tss_0based", "tpm"}
    missing = sorted(required_columns - set(genes.columns))
    if missing:
        raise ValueError(f"Gene expression label table is missing columns: {', '.join(missing)}")

    genes = genes.dropna(subset=["tss_0based", "tpm"]).copy()
    genes["_chromosome"] = genes["chrom"].map(normalize_chromosome)
    genes["_tss"] = pd.to_numeric(genes["tss_0based"], errors="raise").astype(np.int64)
    expression = pd.to_numeric(genes["tpm"], errors="raise").to_numpy(dtype=np.float64)
    parameters = config.section("dataset")
    if parameters.get("expression_log2p1", True):
        expression = np.log2(expression + 1.0)

    expression_mean = float(expression.mean())
    expression_std = float(expression.std(ddof=int(parameters.get("expression_zscore_ddof", 1))))
    if parameters.get("expression_zscore", True):
        if not np.isfinite(expression_std) or expression_std <= 0:
            raise ValueError("Expression standard deviation must be positive for z-score normalization.")
        expression = (expression - expression_mean) / expression_std
    else:
        expression_mean = 0.0
        expression_std = 1.0
    genes["_expression"] = expression.astype(np.float32)
    return genes, expression_mean, expression_std


def _space_peak_centers(peaks: pd.DataFrame, strength: pd.Series) -> pd.DataFrame:
    """Keep peaks in descending signal order, enforcing spacing within each chromosome."""

    retained = []
    for _, group in peaks.groupby("chrom", sort=True):
        group = group.sort_values("center", kind="stable")
        centers = group["center"].to_numpy()
        # Open bounds retain centers exactly MIN_PEAK_SPACING_BP apart.
        left = centers.searchsorted(centers - MIN_PEAK_SPACING_BP, side="right")
        right = centers.searchsorted(centers + MIN_PEAK_SPACING_BP, side="left")
        suppressed = np.zeros(len(group), dtype=bool)
        keep = np.zeros(len(group), dtype=bool)
        # Stable ranking breaks ties by center, then by input order for identical centers.
        priority = np.argsort(-strength.loc[group.index].to_numpy(dtype=np.float64), kind="stable")
        for index in priority:
            if not suppressed[index]:
                keep[index] = True
                suppressed[left[index] : right[index]] = True
        retained.extend(group.index[keep])
    return peaks.loc[retained].copy()


def _prepare_peak_table(config: SpartaConfig) -> pd.DataFrame:
    raw = pd.read_csv(config.resource_path("atac_peaks"), sep="\t", header=None, comment="#")
    peak_format = str(config.get("resources.atac_peaks.format", "auto")).lower()
    if peak_format == "auto":
        peak_format = "narrowpeak" if raw.shape[1] == 10 else "bed"
    if peak_format not in {"bed", "narrowpeak"}:
        raise ValueError("resources.atac_peaks.format must be auto, bed, or narrowPeak.")
    if peak_format == "narrowpeak" and raw.shape[1] < 10:
        raise ValueError("narrowPeak input must include the summit offset in column 10.")
    if raw.shape[1] < 5:
        raise ValueError("ATAC peaks must include the first five BED columns.")

    peaks = raw.iloc[:, :5].copy()
    peaks.columns = ["chrom", "start", "end", "name", "score"]
    peaks["chrom"] = peaks["chrom"].map(normalize_chromosome)
    peaks["start"] = pd.to_numeric(peaks["start"], errors="coerce")
    peaks["end"] = pd.to_numeric(peaks["end"], errors="coerce")
    peaks["score"] = pd.to_numeric(peaks["score"], errors="coerce").fillna(0.0)
    peaks = peaks.dropna(subset=["start", "end"]).copy()
    centers = (peaks["start"] + peaks["end"]) // 2
    strength = peaks["score"]
    if peak_format == "narrowpeak":
        offsets = pd.to_numeric(raw.loc[peaks.index, 9], errors="coerce")
        valid = offsets.eq(-1) | (offsets.ge(0) & offsets.lt(peaks["end"] - peaks["start"]) & offsets.mod(1).eq(0))
        if not valid.all():
            index = valid.index[~valid][0]
            raise ValueError(
                f"Invalid narrowPeak summit offset {raw.at[index, 9]!r} at "
                f"{peaks.at[index, 'chrom']}:{peaks.at[index, 'start']}-{peaks.at[index, 'end']}. "
                "Expected -1 or an integer offset inside the peak interval."
            )
        centers = centers.where(offsets.eq(-1), peaks["start"] + offsets)
        strength = pd.to_numeric(raw.loc[peaks.index, 6], errors="coerce")
    peaks["center"] = centers.astype(np.int64)
    peaks = _space_peak_centers(peaks, strength)
    peaks.attrs["peak_center_method"] = "narrowpeak_summit" if peak_format == "narrowpeak" else "interval_midpoint"
    return peaks


def build_hdf5_dataset(config: SpartaConfig, force: bool = False) -> dict[str, Any]:
    """Build bounded-memory, chromosome-partitioned training datasets."""

    genes, expression_mean, expression_std = _prepare_gene_table(config)
    peaks = _prepare_peak_table(config)
    peak_center_method = peaks.attrs["peak_center_method"]
    peak_metadata = {"peak_center_method": peak_center_method, "min_peak_spacing_bp": MIN_PEAK_SPACING_BP}
    output_dir = config.output_path("dataset_dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    chromosomes = list(config.get("data.chromosomes"))
    chromosome_summaries: list[dict[str, Any]] = []

    # Check the whole partial build before adding any new chromosomes to it.
    if not force:
        for chromosome in chromosomes:
            existing = output_dir / f"{chromosome}.h5"
            if existing.is_file():
                with h5py.File(existing, "r") as handle:
                    for key, expected in peak_metadata.items():
                        stored = handle.attrs.get(key)
                        if stored != expected:
                            raise ValueError(
                                f"Cannot reuse {existing}: {key}={stored!r}, expected {expected!r}. "
                                "Use a new dataset directory or rebuild with --force."
                            )

    for chromosome in chromosomes:
        chromosome_genes = genes.loc[genes["_chromosome"] == chromosome].reset_index(drop=True)
        if chromosome_genes.empty:
            LOGGER.warning("No labeled genes found on %s; skipping.", chromosome)
            continue
        chromosome_peaks = peaks.loc[peaks["chrom"] == chromosome].reset_index(drop=True)
        output_path = output_dir / f"{chromosome}.h5"
        if output_path.exists() and not force:
            LOGGER.info("Using existing chromosome dataset: %s", output_path)
            chromosome_summaries.append(
                {
                    "chromosome": chromosome,
                    "genes": len(chromosome_genes),
                    "regulatory_peaks": len(chromosome_peaks),
                    "output": str(output_path),
                    "reused": True,
                }
            )
            continue
        LOGGER.info(
            "Building %s with %d genes and %d ATAC peaks.",
            chromosome,
            len(chromosome_genes),
            len(chromosome_peaks),
        )
        chromosome_summaries.append(
            _write_chromosome(
                config=config,
                chromosome=chromosome,
                genes=chromosome_genes,
                peaks=chromosome_peaks,
                peak_center_method=peak_center_method,
                expression_mean=expression_mean,
                expression_std=expression_std,
                output_path=output_path,
            )
        )

    if not chromosome_summaries:
        raise ValueError("No chromosome datasets were generated.")
    return {
        "schema_version": "1.1",
        "configuration_sha256": config.digest(),
        **peak_metadata,
        "expression_mean": expression_mean,
        "expression_std": expression_std,
        "total_genes": int(sum(item["genes"] for item in chromosome_summaries)),
        "chromosomes": chromosome_summaries,
        "output_dir": str(output_dir),
    }

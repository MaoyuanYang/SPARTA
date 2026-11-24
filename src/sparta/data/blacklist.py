"""Exclude ATAC peak centers in reference blacklist BED intervals."""

from __future__ import annotations

import gzip
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from sparta.config import SpartaConfig
from sparta.data import normalize_chromosome


def configured_blacklist_path(config: SpartaConfig) -> Path:
    if not config.get("resources.atac_blacklist.path", None):
        raise ValueError("Set resources.atac_blacklist.path to the blacklist BED for this genome assembly.")
    path = config.resource_path("atac_blacklist")
    if not path.is_file():
        raise FileNotFoundError(f"ATAC blacklist BED not found: {path}")
    return path


def _blacklist_bytes(path: Path) -> bytes:
    if path.suffix.lower() == ".gz":
        with gzip.open(path, "rb") as handle:
            return handle.read()
    return path.read_bytes()


def blacklist_sha256(path: Path) -> str:
    return hashlib.sha256(_blacklist_bytes(path)).hexdigest()


def exclude_blacklisted_peaks(peaks: pd.DataFrame, path: Path) -> tuple[pd.DataFrame, str]:
    """Filter center coordinates against half-open intervals and fingerprint the BED used."""

    contents = _blacklist_bytes(path)
    intervals: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for line_number, line in enumerate(contents.decode("utf-8-sig").splitlines(), start=1):
        fields = line.split()
        if not fields or fields[0].startswith("#") or fields[0] in {"track", "browser"}:
            continue
        try:
            chrom, start_text, end_text = fields[:3]
            start, end = int(start_text), int(end_text)
            if not 0 <= start < end:
                raise ValueError
        except ValueError as error:
            raise ValueError(f"Invalid blacklist BED interval in {path} at line {line_number}: {line}") from error
        intervals[normalize_chromosome(chrom)].append((start, end))

    keep = pd.Series(True, index=peaks.index)
    for chrom, group in peaks.groupby("chrom", sort=False):
        if chrom in intervals:
            starts, ends = np.sort(np.asarray(intervals[chrom], dtype=np.int64), axis=0).T
            centers = group["center"].to_numpy()
            # At a center outside the union, every interval that started has also ended.
            started = starts.searchsorted(centers, side="right")
            ended = ends.searchsorted(centers, side="right")
            keep.loc[group.index] = started == ended
    return peaks.loc[keep].copy(), hashlib.sha256(contents).hexdigest()

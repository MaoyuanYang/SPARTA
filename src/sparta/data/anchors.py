"""Curate MANE Select transcription start sites with FANTOM5 CAGE peaks."""

from __future__ import annotations

import gzip
import logging
import re
from collections.abc import Iterator
from pathlib import Path
from typing import TextIO

import numpy as np
import pandas as pd

from sparta.config import SpartaConfig

LOGGER = logging.getLogger(__name__)
GTF_ATTRIBUTE_PATTERN = re.compile(r'(\S+)\s+"([^"]+)"')


def _open_text(path: Path) -> TextIO:
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _gtf_records(path: Path) -> Iterator[tuple[list[str], dict[str, str]]]:
    with _open_text(path) as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9:
                continue
            attributes: dict[str, str] = {}
            tags: list[str] = []
            for key, value in GTF_ATTRIBUTE_PATTERN.findall(fields[8]):
                if key == "tag":
                    tags.append(value)
                else:
                    attributes[key] = value
            if tags:
                attributes["tag"] = ",".join(tags)
            yield fields, attributes


def _stable_identifier(value: str | None) -> str | None:
    return value.split(".", maxsplit=1)[0] if value else None


def _unique_mapping(frame: pd.DataFrame, key: str, value: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=[key, value])
    result = frame.dropna(subset=[key, value]).copy()
    result = result.groupby(key, group_keys=False).filter(lambda group: group[value].nunique() == 1)
    return result[[key, value]].drop_duplicates()


def _load_histone_identifiers(path: Path) -> tuple[set[str], set[str]]:
    table = pd.read_csv(path, sep="\t", dtype=str)
    hgnc_ids = set(table.get("HGNC ID", pd.Series(dtype=str)).dropna().astype(str))
    symbol_columns = ("Approved symbol", "Gene symbol", "Symbol", "gene_symbol", "Gene")
    symbol_column = next((name for name in symbol_columns if name in table.columns), None)
    symbols = set(table[symbol_column].dropna().astype(str)) if symbol_column else set()
    if not hgnc_ids and not symbols:
        raise ValueError(
            f"Histone-gene table must contain 'HGNC ID' or a recognized gene-symbol column: {path}"
        )
    return hgnc_ids, symbols


def _extract_mane_and_mappings(
    gtf_path: Path,
    histone_hgnc_ids: set[str],
    histone_symbols: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    mane_records: list[dict] = []
    gene_mapping_records: list[dict] = []
    transcript_mapping_records: list[dict] = []
    protein_coding_genes: set[str] = set()
    excluded_histone_genes: set[str] = set()
    mane_transcript_count = 0

    for fields, attributes in _gtf_records(gtf_path):
        feature_type = fields[2]
        ensembl_id = _stable_identifier(attributes.get("gene_id"))
        transcript_id = _stable_identifier(attributes.get("transcript_id"))
        gene_symbol = attributes.get("gene_name")
        hgnc_id = attributes.get("hgnc_id")

        if feature_type in {"gene", "transcript"} and ensembl_id:
            gene_mapping_records.append(
                {
                    "ensembl_id": ensembl_id,
                    "gene_symbol": gene_symbol,
                    "hgnc_id": hgnc_id,
                }
            )
        if feature_type == "transcript" and transcript_id and ensembl_id:
            transcript_mapping_records.append(
                {
                    "transcript_id": transcript_id,
                    "ensembl_id": ensembl_id,
                }
            )
        if feature_type != "transcript":
            continue

        if attributes.get("gene_type") == "protein_coding" and ensembl_id:
            protein_coding_genes.add(ensembl_id)
        tags = set(attributes.get("tag", "").split(","))
        if attributes.get("gene_type") != "protein_coding" or "MANE_Select" not in tags:
            continue

        mane_transcript_count += 1
        if not ensembl_id or not gene_symbol:
            continue
        if (hgnc_id and hgnc_id in histone_hgnc_ids) or gene_symbol in histone_symbols:
            excluded_histone_genes.add(ensembl_id)
            continue

        start_1based = int(fields[3])
        end_1based = int(fields[4])
        strand = fields[6]
        tss_0based = start_1based - 1 if strand == "+" else end_1based - 1
        mane_records.append(
            {
                "ensembl_id": ensembl_id,
                "gene_symbol": gene_symbol,
                "hgnc_id": hgnc_id,
                "chrom": fields[0],
                "strand": strand,
                "mane_tss_0based": tss_0based,
            }
        )

    mane = pd.DataFrame(mane_records)
    if mane.empty:
        raise ValueError(f"No protein-coding MANE Select transcripts were found in {gtf_path}")
    duplicate_gene_count = int(mane.loc[mane.duplicated("ensembl_id", keep=False), "ensembl_id"].nunique())
    if duplicate_gene_count:
        LOGGER.warning(
            "%d genes have multiple MANE Select records; retaining the first record.",
            duplicate_gene_count,
        )
    mane = mane.drop_duplicates("ensembl_id", keep="first").copy()

    gene_mappings = pd.DataFrame(gene_mapping_records).drop_duplicates()
    transcript_mappings = pd.DataFrame(transcript_mapping_records).drop_duplicates()
    hgnc_mapping = _unique_mapping(gene_mappings, "hgnc_id", "ensembl_id")
    transcript_mapping = _unique_mapping(transcript_mappings, "transcript_id", "ensembl_id")
    statistics = {
        "protein_coding_genes": len(protein_coding_genes),
        "mane_transcripts": mane_transcript_count,
        "excluded_histone_genes": len(excluded_histone_genes),
        "retained_mane_genes": len(mane),
        "unique_hgnc_mappings": len(hgnc_mapping),
        "unique_transcript_mappings": len(transcript_mapping),
    }
    return mane, hgnc_mapping, transcript_mapping, statistics


def _load_cage_candidates(config: SpartaConfig) -> pd.DataFrame:
    bed_columns = (
        "chrom",
        "chrom_start",
        "chrom_end",
        "peak_id",
        "score",
        "strand",
        "thick_start",
        "thick_end",
        "item_rgb",
    )
    bed = pd.read_csv(
        config.resource_path("cage_peaks_bed"),
        sep="\t",
        header=None,
        names=bed_columns,
        usecols=range(len(bed_columns)),
        dtype=str,
        low_memory=False,
    )
    bed = bed[["peak_id", "chrom", "strand", "thick_start"]].rename(
        columns={"chrom": "cage_chrom", "strand": "cage_strand", "thick_start": "cage_tss_0based"}
    )
    bed["cage_tss_0based"] = pd.to_numeric(bed["cage_tss_0based"], errors="coerce")
    bed = bed.dropna(subset=["peak_id", "cage_chrom", "cage_strand", "cage_tss_0based"]).copy()
    bed["cage_tss_0based"] = bed["cage_tss_0based"].astype(np.int64)

    names = pd.read_csv(
        config.resource_path("cage_peak_names"),
        sep="\t",
        header=None,
        names=("peak_id", "new_name", "old_name"),
        dtype=str,
        low_memory=False,
    )
    primary_peaks = names.loc[
        names["new_name"].astype(str).str.contains("p1@", regex=False, na=False),
        ["peak_id"],
    ].drop_duplicates()

    annotation_columns = (
        "peak_id",
        "transcript_id",
        "distance",
        "entrez_id",
        "hgnc_id",
        "uniprot_id",
        "gene_name",
        "gene_symbol",
        "synonym",
        "source",
    )
    annotations = pd.read_csv(
        config.resource_path("cage_annotations"),
        sep="\t",
        header=None,
        names=annotation_columns,
        usecols=range(len(annotation_columns)),
        dtype=str,
        low_memory=False,
    )
    annotations = annotations[["peak_id", "transcript_id", "hgnc_id", "gene_symbol"]].copy()
    annotations = annotations.replace({"nan": np.nan, "None": np.nan, "": np.nan})
    annotations["transcript_id"] = annotations["transcript_id"].str.split(".").str[0]

    candidates = primary_peaks.merge(annotations, on="peak_id", how="inner")
    candidates = candidates.merge(bed, on="peak_id", how="inner")
    return candidates.drop_duplicates()


def _map_cage_to_ensembl(
    candidates: pd.DataFrame,
    hgnc_mapping: pd.DataFrame,
    transcript_mapping: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    mapped = candidates.merge(
        hgnc_mapping.rename(columns={"ensembl_id": "ensembl_id_by_hgnc"}),
        on="hgnc_id",
        how="left",
    )
    mapped = mapped.merge(
        transcript_mapping.rename(columns={"ensembl_id": "ensembl_id_by_transcript"}),
        on="transcript_id",
        how="left",
    )
    mapped["ensembl_id"] = mapped["ensembl_id_by_hgnc"].combine_first(
        mapped["ensembl_id_by_transcript"]
    )
    mapped["mapping_source"] = pd.Series(index=mapped.index, dtype="object")
    mapped.loc[mapped["ensembl_id_by_hgnc"].notna(), "mapping_source"] = "hgnc_id"
    transcript_only = mapped["ensembl_id_by_hgnc"].isna() & mapped["ensembl_id_by_transcript"].notna()
    mapped.loc[transcript_only, "mapping_source"] = "transcript_id"

    statistics = {
        "raw_primary_cage_records": len(candidates),
        "mapped_by_hgnc": int((mapped["mapping_source"] == "hgnc_id").sum()),
        "mapped_by_transcript": int((mapped["mapping_source"] == "transcript_id").sum()),
        "unmapped_cage_records": int(mapped["ensembl_id"].isna().sum()),
    }
    mapped = mapped.dropna(subset=["ensembl_id"]).drop_duplicates(
        ["peak_id", "ensembl_id", "cage_tss_0based", "cage_chrom", "cage_strand"]
    )
    statistics["genes_with_multiple_cage_peaks"] = int(
        (mapped.groupby("ensembl_id").size() > 1).sum()
    )
    return mapped, statistics


def curate_tss_anchors(config: SpartaConfig) -> dict[str, int | float | str]:
    """Create a high-confidence MANE Select and CAGE-curated TSS table."""

    histone_hgnc_ids, histone_symbols = _load_histone_identifiers(
        config.resource_path("histone_genes")
    )
    mane, hgnc_mapping, transcript_mapping, gtf_statistics = _extract_mane_and_mappings(
        config.resource_path("gencode_gtf"),
        histone_hgnc_ids,
        histone_symbols,
    )
    cage_candidates = _load_cage_candidates(config)
    cage_mapped, cage_statistics = _map_cage_to_ensembl(
        cage_candidates,
        hgnc_mapping,
        transcript_mapping,
    )

    merged = mane.merge(
        cage_mapped[
            [
                "peak_id",
                "ensembl_id",
                "cage_tss_0based",
                "cage_chrom",
                "cage_strand",
                "mapping_source",
            ]
        ],
        on="ensembl_id",
        how="left",
    )
    has_peak = merged["peak_id"].notna()
    consistent = (merged["chrom"] == merged["cage_chrom"]) & (
        merged["strand"] == merged["cage_strand"]
    )
    inconsistent_cage_records = int((has_peak & ~consistent).sum())
    valid = merged.loc[has_peak & consistent].copy()
    valid["tss_distance_bp"] = (
        valid["mane_tss_0based"] - valid["cage_tss_0based"]
    ).abs()
    best = (
        valid.sort_values(["ensembl_id", "tss_distance_bp", "peak_id"])
        .drop_duplicates("ensembl_id", keep="first")
        [
            [
                "ensembl_id",
                "peak_id",
                "cage_tss_0based",
                "tss_distance_bp",
                "mapping_source",
            ]
        ]
        .rename(columns={"peak_id": "cage_peak_id", "mapping_source": "cage_mapping_source"})
    )

    curated = mane.merge(best, on="ensembl_id", how="left")
    max_distance = int(config.get("data.max_cage_distance_bp"))
    use_cage = curated["cage_tss_0based"].notna() & (
        curated["tss_distance_bp"] <= max_distance
    )
    curated["tss_0based"] = curated["mane_tss_0based"]
    curated.loc[use_cage, "tss_0based"] = curated.loc[use_cage, "cage_tss_0based"]
    curated["tss_0based"] = curated["tss_0based"].astype(np.int64)
    curated["tss_source"] = np.where(use_cage, "FANTOM5_CAGE_P1", "MANE_Select")

    chromosomes = list(config.get("data.chromosomes"))
    curated = curated.loc[curated["chrom"].isin(chromosomes)].copy()
    curated["chrom"] = pd.Categorical(curated["chrom"], categories=chromosomes, ordered=True)
    curated = curated.sort_values(["chrom", "tss_0based", "ensembl_id"]).reset_index(drop=True)
    curated["chrom"] = curated["chrom"].astype(str)
    output_columns = [
        "ensembl_id",
        "gene_symbol",
        "hgnc_id",
        "chrom",
        "strand",
        "mane_tss_0based",
        "cage_tss_0based",
        "cage_peak_id",
        "cage_mapping_source",
        "tss_distance_bp",
        "tss_0based",
        "tss_source",
    ]
    output = config.output_path("tss_anchors")
    output.parent.mkdir(parents=True, exist_ok=True)
    curated[output_columns].to_csv(output, sep="\t", index=False)

    corrected_genes = int((curated["tss_source"] == "FANTOM5_CAGE_P1").sum())
    summary: dict[str, int | float | str] = {
        **gtf_statistics,
        **cage_statistics,
        "inconsistent_cage_records": inconsistent_cage_records,
        "cage_corrected_genes": corrected_genes,
        "mane_retained_genes": len(curated) - corrected_genes,
        "final_genes": len(curated),
        "max_cage_distance_bp": max_distance,
        "output": str(output),
    }
    LOGGER.info(
        "Curated %d TSS anchors (%d CAGE-corrected, %d MANE Select).",
        len(curated),
        summary["cage_corrected_genes"],
        summary["mane_retained_genes"],
    )
    return summary

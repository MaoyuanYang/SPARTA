"""Synthetic integration tests for TSS and expression-label curation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sparta.config import SpartaConfig
from sparta.data.anchors import curate_tss_anchors
from sparta.data.labels import build_expression_labels


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_tss_and_expression_curation(tmp_path: Path) -> None:
    histone = _write(
        tmp_path / "histone.tsv",
        "HGNC ID\tApproved symbol\nHGNC:2\tHIST1\n",
    )
    gtf = _write(
        tmp_path / "annotation.gtf",
        "\n".join(
            [
                (
                    'chr1\ttest\ttranscript\t101\t200\t.\t+\t.\tgene_id "ENSG1.1"; '
                    'transcript_id "ENST1.1"; gene_name "GENE1"; hgnc_id "HGNC:1"; '
                    'gene_type "protein_coding"; tag "MANE_Select"; tag "basic";'
                ),
                (
                    'chr1\ttest\ttranscript\t301\t400\t.\t+\t.\tgene_id "ENSG2.1"; '
                    'transcript_id "ENST2.1"; gene_name "HIST1"; hgnc_id "HGNC:2"; '
                    'gene_type "protein_coding"; tag "MANE_Select";'
                ),
            ]
        )
        + "\n",
    )
    cage_bed = _write(
        tmp_path / "cage.bed",
        "chr1\t95\t106\tpeak1\t0\t+\t100\t101\t0\n",
    )
    cage_names = _write(
        tmp_path / "cage_names.tsv",
        "peak1\tp1@GENE1\told\n",
    )
    cage_annotations = _write(
        tmp_path / "cage_annotations.tsv",
        "peak1\tENST1.1\t0\t1\tHGNC:1\tP1\tGENE1\tGENE1\t-\ttest\n",
    )
    rna_seq = _write(
        tmp_path / "rna.tsv",
        "gene_id\tTPM\nENSG1.1\t2.0\nENSG1.2\t3.0\n",
    )
    anchors = tmp_path / "anchors.tsv"
    labels = tmp_path / "labels.tsv"
    values = {
        "resources": {
            "histone_genes": {"path": str(histone)},
            "gencode_gtf": {"path": str(gtf)},
            "cage_peaks_bed": {"path": str(cage_bed)},
            "cage_peak_names": {"path": str(cage_names)},
            "cage_annotations": {"path": str(cage_annotations)},
            "rna_seq": {"path": str(rna_seq)},
        },
        "outputs": {
            "tss_anchors": str(anchors),
            "gene_labels": str(labels),
        },
        "data": {
            "chromosomes": ["chr1"],
            "max_cage_distance_bp": 256,
            "expression_columns": {"gene_id": "gene_id", "tpm": "TPM"},
        },
    }
    config = SpartaConfig(source_path=tmp_path / "config.yaml", values=values)

    anchor_summary = curate_tss_anchors(config)
    label_summary = build_expression_labels(config)
    curated_anchors = pd.read_csv(anchors, sep="\t")
    curated_labels = pd.read_csv(labels, sep="\t")

    assert anchor_summary["final_genes"] == 1
    assert curated_anchors.loc[0, "tss_source"] == "FANTOM5_CAGE_P1"
    assert curated_anchors.loc[0, "tss_0based"] == 100
    assert label_summary["duplicate_rna_seq_rows_collapsed"] == 1
    assert curated_labels.loc[0, "tpm"] == 5.0

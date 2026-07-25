"""Integrate curated TSS anchors with gene-level RNA-seq TPM labels."""

from __future__ import annotations

import logging

import pandas as pd

from sparta.config import SpartaConfig

LOGGER = logging.getLogger(__name__)


def build_expression_labels(config: SpartaConfig) -> dict[str, int | float | str]:
    """Join curated anchors to RNA-seq labels using stable Ensembl identifiers."""

    anchor_path = config.output_path("tss_anchors")
    if not anchor_path.is_file():
        raise FileNotFoundError(
            f"Curated TSS anchors not found: {anchor_path}. Run the anchors stage first."
        )
    anchors = pd.read_csv(anchor_path, sep="\t", dtype={"ensembl_id": str})
    required_anchor_columns = {"ensembl_id", "chrom", "strand", "tss_0based"}
    missing_anchor_columns = sorted(required_anchor_columns - set(anchors.columns))
    if missing_anchor_columns:
        raise ValueError(
            f"TSS anchor table is missing columns: {', '.join(missing_anchor_columns)}"
        )

    source_columns = config.get("data.expression_columns")
    source_gene_id = source_columns["gene_id"]
    source_tpm = source_columns["tpm"]
    expression = pd.read_csv(config.resource_path("rna_seq"), sep="\t", dtype=str)
    missing_expression_columns = [
        name for name in (source_gene_id, source_tpm) if name not in expression.columns
    ]
    if missing_expression_columns:
        raise ValueError(
            f"RNA-seq table is missing columns: {', '.join(missing_expression_columns)}"
        )

    expression = expression[[source_gene_id, source_tpm]].rename(
        columns={source_gene_id: "ensembl_id", source_tpm: "tpm"}
    )
    expression["ensembl_id"] = expression["ensembl_id"].str.split(".").str[0]
    expression["tpm"] = pd.to_numeric(expression["tpm"], errors="coerce")
    invalid_tpm_rows = int(expression["tpm"].isna().sum())
    expression = expression.dropna(subset=["ensembl_id", "tpm"]).copy()
    duplicate_rows = int(expression["ensembl_id"].duplicated().sum())
    if duplicate_rows:
        expression = expression.groupby("ensembl_id", as_index=False)["tpm"].sum()

    merged = anchors.merge(expression, on="ensembl_id", how="left", validate="one_to_one")
    missing_labels = int(merged["tpm"].isna().sum())
    merged = merged.dropna(subset=["tpm"]).copy()
    chromosomes = list(config.get("data.chromosomes"))
    merged = merged.loc[merged["chrom"].isin(chromosomes)].copy()

    output = config.output_path("gene_labels")
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output, sep="\t", index=False)
    LOGGER.info("Created expression labels for %d genes.", len(merged))
    return {
        "anchor_genes": len(anchors),
        "invalid_tpm_rows": invalid_tpm_rows,
        "duplicate_rna_seq_rows_collapsed": duplicate_rows,
        "anchors_without_expression": missing_labels,
        "labeled_genes": len(merged),
        "mean_tpm": float(merged["tpm"].mean()) if len(merged) else 0.0,
        "median_tpm": float(merged["tpm"].median()) if len(merged) else 0.0,
        "output": str(output),
    }

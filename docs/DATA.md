# Data preparation

SPARTA combines reference annotation, CAGE, RNA-seq, ATAC-seq, ChIP-seq,
reference sequence, and Hi-C data. Large biological data files are intentionally
excluded from Git.

## Required inputs

| Configuration key | Expected format | Purpose |
|---|---|---|
| `resources.histone_genes` | TSV | HGNC histone-gene exclusion list |
| `resources.gencode_gtf` | GTF or GTF.GZ | GENCODE MANE Select transcripts |
| `resources.cage_peaks_bed` | BED | FANTOM5 CAGE peak coordinates |
| `resources.cage_peak_names` | TSV | FANTOM5 peak names used to select P1@ peaks |
| `resources.cage_annotations` | TSV | CAGE-to-gene annotations |
| `resources.rna_seq` | TSV | Gene-level RNA-seq TPM values |
| `resources.atac_peaks` | narrowPeak or BED | Candidate regulatory-node peaks |
| `resources.reference_fasta` | FASTA | Reference genome sequence |
| `resources.hic` | `.hic` | Chromatin contact data |
| `epigenomic_tracks.*` | BigWig | Node-level epigenomic signals |

The manuscript uses the hg38 assembly throughout. All coordinate-bearing inputs
must use the same assembly and chromosome naming convention.

## Source portals

The example configuration records source pages for:

- GENCODE gene annotation
- FANTOM5 CAGE
- HGNC gene groups
- ENCODE RNA-seq, ATAC-seq, ChIP-seq, and Hi-C data

These pages are provenance metadata. Direct downloads are performed only when a
resource has an explicit `url` value. This avoids silently choosing a different
experiment or file revision.

## Expected tabular schemas

### Histone genes

The table must include `HGNC ID` and preferably one of:

- `Approved symbol`
- `Gene symbol`
- `Symbol`
- `gene_symbol`
- `Gene`

### RNA-seq

Column names are configurable under `data.expression_columns`. The defaults are:

```text
gene_id    TPM
```

Ensembl version suffixes are removed. Duplicate stable Ensembl identifiers are
collapsed by summing TPM values.

### ATAC peaks

Both formats use the first five BED columns (`chrom`, `start`, `end`, `name`,
`score`). For narrowPeak, SPARTA centers each node on the zero-based summit
coordinate `start + peak`, where `peak` is the offset in column 10. An offset
of `0` selects the first base. The narrowPeak sentinel `-1` selects the interval
midpoint, rounded down. For BED input, SPARTA uses the interval midpoint. This
also recovers the summit from BED windows already centered on it.

Set `resources.atac_peaks.format` to `narrowPeak` for raw ENCODE narrowPeak
files, including downloads named `.bed.gz`. Set it to `bed` for other ten-column
BED tables. If omitted or set to `auto`, SPARTA interprets exactly ten columns
as narrowPeak and other supported BED tables as intervals. Nodes with the same
chromosome and center are deduplicated. Distinct summits in a shared interval
remain separate candidates.

When more than 399 peaks occur in a regulatory neighborhood, nodes are selected
by genomic proximity or score according to `dataset.peak_selection`.

Earlier versions used interval midpoints for raw narrowPeak files. To use the
corrected coordinates, build the dataset in a new `outputs.dataset_dir`. The
dataset stage normally reuses existing outputs. Use `--force` only if you
intend to replace them. Results and weights from earlier runs correspond to
the input coordinates used for those runs. New HDF5 files and the dataset
summary record the applied rule as `peak_center_method` (`narrowpeak_summit`
or `interval_midpoint`).

## Generated files

The pipeline writes:

1. A MANE Select and CAGE-curated TSS table.
2. A gene table with RNA-seq TPM labels.
3. One HDF5 dataset per chromosome.
4. Dataset and curation summary JSON files.

Each chromosome HDF5 file contains:

| Dataset | Shape |
|---|---|
| `dna_sequence` | genes x nodes x 4 x window |
| `epigenomic_signals` | genes x nodes x tracks x window |
| `hic_scale` | genes x nodes x nodes |
| `hic_oe` | genes x nodes x nodes |
| `genomic_coordinates` | genes x nodes |
| `expression_target` | genes |
| `promoter_mask` | genes x nodes |
| `node_mask` | genes x nodes |

Metadata attributes record preprocessing transforms, track order, dimensions,
Hi-C resolution, and label normalization statistics.

Schema 1.1 stores epigenomic signals and both Hi-C tensors as float32, including
the in-memory caches before log transforms. It also records
`expression_zscore_ddof`, `exclude_promoter_overlaps`, and `peak_selection`.
The loader compares preprocessing metadata across every chromosome and across
the training, validation, and test splits. Missing required metadata or mismatched
normalization, assembly, cell line, transforms, dimensions, or track order is an
error. Full configuration checksums are not used for this comparison because
training-only settings do not change preprocessing.

Consistent schema 1.0 files with the required metadata remain readable. Optional
attributes absent from all legacy files are treated as unknown; they must not
be mixed with files recording different or known values. Do not infer missing
metadata without verifying how a file was built. Files already containing
`inf` or `NaN` must be rebuilt from the source data; changing their dtype cannot
recover lost values. The loader reports the file, sample, and affected tensor.

## Storage and memory

The full dataset is large. HDF5 tensors are compressed and written in bounded
batches, but chromosome-level BigWig and sparse Hi-C caches can still require
substantial memory. Reduce `dataset.n_jobs` and `dataset.batch_size` if memory is
limited. Do not commit generated HDF5 files, checkpoints, reference genomes, or
raw tracks.

Float32 doubles the uncompressed signal and contact storage compared with the
old float16 representation; actual compressed file sizes depend on the data.

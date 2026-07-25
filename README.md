<p align="center">
  <img src="assets/sparta-logo.png" alt="SPARTA logo" width="900">
</p>

# SPARTA

SPARTA (Spatial and Physics-Aware Regulatory Transformer Architecture) predicts
gene expression from DNA sequence, epigenomic profiles, chromatin accessibility,
and chromatin conformation. The implementation follows the terminology and
two-stage optimization strategy described in the accompanying manuscript.

## Highlights

- Configuration-driven data curation, dataset construction, training, and test
  evaluation.
- A single command runs the complete workflow.
- Paper-aligned model outputs: regulatory potential, biochemical affinity,
  basal expression, interaction energy, interaction probability,
  distance-decay force, and path-insulation force.
- Chromosome-held-out evaluation with chromosome 16 for validation and
  chromosomes 8 and 9 for testing.
- Reproducible checkpoints, run metadata, metrics, and test predictions.
- A standard `src/` layout with tests, continuous integration, citation
  metadata, and an open-source license.

## Repository layout

```text
.
|-- configs/                 Reproducible experiment configurations
|-- docs/                    Data and configuration documentation
|   `-- Supplementary_Table_S1.xlsx   Accessions and download URLs for all cell lines
|-- scripts/                 Direct command-line entry point
|-- src/sparta/
|   |-- data/                TSS curation, expression labels, and HDF5 datasets
|   |-- models/              Biochemical and physics-based model components
|   |-- training/            Losses and two-stage training
|   |-- cli.py               Installed `sparta` command
|   |-- config.py            Configuration loading and validation
|   `-- pipeline.py          End-to-end workflow orchestration
|-- tests/                   Lightweight unit and model smoke tests
|-- CITATION.cff
|-- LICENSE
`-- pyproject.toml
```

## Installation

SPARTA is intended for Linux with Python 3.10 or newer. A CUDA-capable GPU is
strongly recommended for model training.

```bash
conda env create -f environment.yml
conda activate sparta
```

Alternatively:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

The `.hic` reader and BigWig dependencies may require platform-specific system
libraries. See [Data preparation](docs/DATA.md) for expected inputs.

## End-to-end workflow

The full workflow has six steps: configuration, data download, TSS/label
curation, dataset construction, two-stage training, and held-out evaluation.
Every step is driven by one YAML file, and every command below is also
available without installing the package by replacing `sparta` with
`python scripts/run_pipeline.py`.

### 1. Prepare the configuration

Copy the example configuration and replace every example path with the
corresponding local file:

```bash
cp configs/k562.example.yaml configs/k562.yaml
```

Only the K562 example configuration is shipped. To run SPARTA on the other
cell lines analyzed in the manuscript (GM12878, HCT116, HepG2, MCF-7, and
A549), set `project.cell_line` accordingly and replace the K562 file paths and
URLs. The ENCODE/4DN accessions and direct download URLs for every required
file are listed in
[docs/Supplementary_Table_S1.xlsx](docs/Supplementary_Table_S1.xlsx).

### 2. Download input data

The example configuration already contains direct `url` values for all
required K562 resources (reference annotation, FANTOM5 CAGE, HGNC histone
genes, hg38 FASTA, RNA-seq, ATAC-seq, ChIP-seq, and Hi-C). Download them with:

```bash
sparta run --config configs/k562.yaml --stage download
```

Files are written to the `path` of each resource and are not re-downloaded
unless `--force` is given. Alternatively, download the files manually from the
portals listed in the supplementary table. `source_page` entries are
provenance metadata and are never scraped automatically.

Then verify the configuration and that all inputs exist:

```bash
sparta validate --config configs/k562.yaml --check-files
```

### 3. Curate TSS anchors and expression labels

```bash
sparta run --config configs/k562.yaml --stage anchors
sparta run --config configs/k562.yaml --stage labels
```

- `anchors` builds the MANE Select plus FANTOM5 CAGE-curated TSS table
  (`outputs.tss_anchors`).
- `labels` integrates RNA-seq TPM values, applying `log2(TPM + 1)` and global
  z-score normalization (`outputs.gene_labels`).

Each stage writes a `*.summary.json` next to its output and is skipped on
re-runs unless `--force` is passed.

### 4. Build the HDF5 dataset

```bash
sparta run --config configs/k562.yaml --stage dataset
```

This materializes one compressed HDF5 file per chromosome under
`outputs.dataset_dir`, containing DNA one-hot sequences, eight epigenomic
tracks in 1,024-bp node windows, 5-kb SCALE-normalized and
observed-over-expected Hi-C matrices, node masks, and expression targets. A
`_SUCCESS.json` marker records the construction summary. This is the most
memory-intensive step; reduce `dataset.n_jobs` and `dataset.batch_size` if
memory is limited.

### 5. Two-stage training

```bash
sparta run --config configs/k562.yaml --stage train
```

Training follows the manuscript's two-stage strategy:

- **Stage 1 (physics pre-training):** the model is optimized on the
  stratified and global Hi-C correlation losses with the three biochemical
  readout heads frozen.
- **Stage 2 (joint fine-tuning):** the full model is unfrozen and trained with
  the expression loss plus down-weighted Hi-C losses, using a lower learning
  rate for the pre-trained backbone than for the readout heads.

Per-stage `last` and `best` checkpoints, a JSONL metric history
(`metrics.jsonl`), and resolved run metadata are written to
`outputs.training_dir`. Interrupted runs resume automatically when
`training.resume: true`.

### 6. Held-out evaluation

Evaluation runs automatically at the end of the training stage. The best
available checkpoints are loaded and the held-out test chromosomes (chr8 and
chr9; chr16 was used for validation/model selection) are scored. Outputs in
`outputs.training_dir`:

- `test_predictions.tsv` — per-gene observed and predicted expression.
- `run_complete.json` — test metrics (including Pearson correlation) and run
  summary.

### Run everything with one command

```bash
sparta run --config configs/k562.yaml --stage all
```

`--stage all` executes curation, dataset construction, training, and test
evaluation in order, reusing existing stage outputs. Set
`pipeline.download_missing: true` to prepend the download stage, and add
`--force` to rebuild completed data stages.

## Method-to-code terminology

| Manuscript term             | Code name                     |
| --------------------------- | ----------------------------- |
| Regulatory node embedding   | `regulatory_node_embedding` |
| Regulatory potential, phi   | `regulatory_potential`      |
| Biochemical affinity, alpha | `biochemical_affinity`      |
| Basal expression, beta      | `basal_expression`          |
| Interaction energy          | `interaction_energy`        |
| Interaction probability     | `interaction_probability`   |
| Distance-decay force        | `distance_decay_force`      |
| Path-insulation force       | `path_insulation_force`     |
| Predicted expression        | `predicted_expression`      |

For a target gene, the model implements the additive readout

```text
predicted_expression =
    basal_expression
    + sum(regulatory_potential
          * biochemical_affinity
          * promoter_interaction_probability)
```

with the promoter self-contribution excluded.

## Reproducibility

The example configuration records the manuscript defaults:

- 4-Mb regulatory neighborhood, extending 2 Mb on each side of the TSS.
- Up to 399 ATAC-seq regulatory nodes plus the TSS node.
- 1,024-bp windows at every node.
- Eight epigenomic tracks: ATAC, CTCF, RAD21, H3K27ac, H3K4me3,
  H3K4me1, H3K27me3, and H3K9me3.
- 5-kb SCALE-normalized and observed-over-expected Hi-C matrices.
- `log2(TPM + 1)` followed by global z-score normalization.
- Physics pre-training followed by joint fine-tuning with differential learning
  rates.

Each training directory contains resolved configuration metadata, JSONL metric
history, resumable last checkpoints, best checkpoints, a completion marker, and
held-out test predictions.

## Development

```bash
python -m pip install -e ".[dev]"
ruff check .
ruff format --check .
pytest
```

## Citation

Please cite the associated SPARTA manuscript and this software repository. A
machine-readable citation is provided in [CITATION.cff](CITATION.cff).

## License

This project is released under the [MIT License](LICENSE).

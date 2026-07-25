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

## Quick start

Copy the example configuration and replace every example path with the
corresponding local file. Optional download URLs can also be provided for
resources that permit direct download.

```bash
cp configs/k562.example.yaml configs/k562.yaml
sparta validate --config configs/k562.yaml --check-files
sparta run --config configs/k562.yaml --stage all
```

The same workflow can be started without installing the package:

```bash
python scripts/run_pipeline.py run --config configs/k562.yaml --stage all
```

`--stage all` runs TSS anchor curation, RNA-seq label integration, HDF5 dataset
construction, two-stage training, and held-out test evaluation. Existing stage
outputs are reused. Add `--force` to rebuild completed data stages.

Individual stages are also available:

```bash
sparta run --config configs/k562.yaml --stage anchors
sparta run --config configs/k562.yaml --stage labels
sparta run --config configs/k562.yaml --stage dataset
sparta run --config configs/k562.yaml --stage train
```

Use `--stage download` only after adding direct `url` values to the
configuration. Source portal addresses are included as `source_page` metadata
and are never scraped automatically.

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

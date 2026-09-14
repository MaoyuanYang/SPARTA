# Configuration reference

All paths and experiment parameters are supplied through one YAML file. Relative
paths are resolved from the directory containing that file. Environment
variables and `~` are expanded.

## Resource entries

Each input resource uses the following structure:

```yaml
resources:
  example:
    path: ../data/raw/example.dat
    url: null
    sha256: null
    source_page: https://example.org/
```

- `path` is required.
- `url` is optional and is used only by the `download` stage.
- `sha256` is optional but recommended for direct downloads.
- `source_page` records provenance and is not downloaded.

Epigenomic tracks use the same structure under `epigenomic_tracks`.
Their insertion order defines dataset channel order and is included in the
configuration checksum. Checkpoints also record the actual selected HDF5 track
order and preprocessing metadata and reject incompatible inputs when loaded.

## Data curation

- `data.chromosomes`: chromosomes retained in the curated dataset.
- `data.max_cage_distance_bp`: maximum MANE-to-CAGE distance accepted for TSS
  correction.
- `data.expression_columns`: source column names for gene identifiers and TPM.

## Dataset construction

- `dataset.max_nodes`: total padded nodes, including the TSS.
- `dataset.window_bp`: sequence and epigenomic window length.
- `dataset.regulatory_window_bp`: total TSS-centered regulatory neighborhood.
- `dataset.peak_selection`: `nearest` or `score`.
- `dataset.hic_resolution_bp`: Hi-C resolution.
- `dataset.hic_normalization`: normalization passed to hic-straw.
- `dataset.exclude_promoter_overlaps`: removes non-promoter windows that overlap
  the promoter window.
- `dataset.epigenomic_log1p`: applies `log1p(max(signal, 0))`.
- `dataset.expression_log2p1`: applies `log2(TPM + 1)`.
- `dataset.expression_zscore`: applies global z-score normalization.
- `dataset.expression_zscore_ddof`: standard-deviation degrees of freedom; `1`
  preserves the original training scripts.
- `dataset.n_jobs`: shared-memory worker threads.
- `dataset.batch_size`: number of genes materialized before an HDF5 write.

## Model

- `model.hidden_dim`: regulatory-node embedding dimension.
- `model.physics_layers`: number of physics-based transformer layers.
- `model.physics_heads`: attention heads per layer.
- `model.dropout`: dropout probability.
- `model.insulation_projection_dim`: path-insulation projection dimension.
- `model.initial_distance_reference_bp`: initial learnable distance reference.
- `model.initial_distance_decay`: initial learnable distance-decay exponent.
- `model.initial_temperature`: initial interaction-probability temperature.

`hidden_dim` must be divisible by `physics_heads`.

## Training

- `training.seed`: global random seed.
- `training.device`: `auto`, `cuda`, `cpu`, or another valid PyTorch device.
- `training.train_chromosomes`: training split.
- `training.validation_chromosomes`: validation split.
- `training.test_chromosomes`: final held-out test split.
- `training.batch_size`, `num_workers`, `use_amp`, `gradient_clip_norm`:
  runtime settings.
- `training.validation_interval`: validation frequency in epochs.
- `training.resume`: resume from per-stage last checkpoints.

Resume restores the optimizer, scheduler, AMP gradient scaler, Python/NumPy/
PyTorch random states, and separate data-sampling and worker-seeding generators.
Checkpoints are replaced atomically after each completed epoch. Reproducible
continuation requires the same configuration, data, device type, and runtime;
it is an epoch-boundary guarantee for the current deterministic HDF5 reader,
not mid-batch recovery or restoration of arbitrary worker-side random transforms.
Legacy checkpoints lacking input metadata or complete training state cannot
be resumed safely; start a new run in a separate output directory.
See [training reliability](TRAINING_RELIABILITY.md) for validation and migration details.

The `stage1` and `stage2` sections define epochs, learning rates, and loss
weights. Stage 1 freezes the three biochemical readout heads. Stage 2 unfreezes
the complete model and uses separate rates for the pre-trained backbone and
readout heads.

`training.hic_loss` defines the distance-bin boundaries, minimum pairs per
stratified bin, and minimum pairs for the global Hi-C Pearson loss.

## Outputs

The `outputs` section controls all generated paths:

```yaml
outputs:
  tss_anchors: ../data/processed/tss_anchors.tsv
  gene_labels: ../data/processed/gene_labels.tsv
  dataset_dir: ../data/processed/hdf5
  training_dir: ../results/k562
```

No output path is embedded in Python source.

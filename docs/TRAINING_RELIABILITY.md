# Training reliability fixes

Scope: the five reproduced findings reviewed against commit
`878043222cb84b74b10b2ea4f0bde11bc0b4b35d`. The user requested these fixes on
2026-09-14. Existing project conventions in CONTRIBUTING.md govern this work.
There is no UI change. The chromosome split (including chr22 and chrX), physical
parameter initialization, and global expression standardization remain unchanged.

## Acceptance and verification

| ID | Expected behavior | Regression scenario |
|---|---|---|
| AC-1 | Finite signal and contact values such as 70000 survive preprocessing and HDF5 storage without becoming infinity. Invalid stored data fail clearly. | Cache, window, contact, HDF5 round-trip, and non-finite sample tests. |
| AC-2 | Masked nodes do not enter CNN BatchNorm statistics or receive input gradients. | Compare valid outputs, gradients, and running statistics with extra/mutated padding, with dropout disabled. |
| AC-3 | Channel order is part of configuration/checkpoint identity. | Reordered tracks change the digest; checkpoint loading rejects different selected channel order. |
| AC-4 | On the same deterministic runtime, epoch-boundary resume continues model, optimizer, scheduler, scaler, sampling, and RNG state. | Compare uninterrupted and interrupted runs, including the stage transition; cover worker-backed deterministic HDF5 loading. |
| AC-5 | Files with missing or incompatible scientific preprocessing metadata cannot be combined across files or train/validation/test splits. | Alter normalization, assembly, transform flags, or expression statistics; require descriptive errors. |

## Verification results

Local validation on 2026-09-14: **33 passed**, including 26 regression cases and
the seven existing tests. Before the production fixes, the focused precision,
padding, digest, and metadata tests reported 20 failures and one pass.
After the fixes, interrupted and uninterrupted two-stage runs produced exactly
equal model, optimizer, and scheduler states and identical prediction files.
This covered interruption in either stage with no workers and interruption in
stage 1 with one persistent worker. An active CPU GradScaler state and Python,
NumPy, and PyTorch random streams also survived a checkpoint round-trip.

`ruff check src scripts tests`, `ruff format --check src scripts tests`, and
`python -m compileall -q src scripts tests` passed. CI now installs the CPU
training dependencies and runs the complete test suite.

The local runtime was Windows, Python 3.12.3, PyTorch 2.5.0+cu124 running on CPU,
NumPy 1.26.4, h5py 3.11.0, pytest 7.4.4, and Ruff 0.16.7. Existing local
dependencies were reused; pytest is older than the declared development range.
CI uses the declared pytest range, but the updated workflow has not run remotely.
FASTA/BigWig readers were replaced by controlled test fixtures; no full biological
dataset or default-size training run was executed. The observed warnings came
from the existing scheduler API and the installed dateutil dependency.

## Compatibility and operational notes

The fixes change padding-related training statistics. Existing model parameter
names and shapes are preserved, but retraining is appropriate for comparisons
using the corrected behavior. Old checkpoints without the new input and resume
metadata cannot certify a safe continuation and must not be silently resumed.

Regenerate datasets that already contain overflowed values. Converting an
existing infinity to float32 does not recover its original value. New datasets
store epigenomic and Hi-C arrays as float32, increasing their uncompressed size.

The loader requires the scientific metadata written by the original SPARTA
builder. Legacy/custom files missing those fields must be rebuilt or supplied
with verified provenance. Training hyperparameters are not part of the shared
preprocessing contract.

Exact continuation is scoped to epoch boundaries, the same runtime/device
configuration, and the current deterministic HDF5 dataset. Arbitrary stochastic
worker transforms and mid-batch recovery are not supported. GPU execution must
be verified on a CUDA host; CPU tests cannot establish GPU determinism.

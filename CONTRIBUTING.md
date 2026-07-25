# Contributing

Thank you for helping improve SPARTA.

## Development setup

```bash
python -m pip install -e ".[dev]"
pre-commit install
```

Before opening a pull request, run:

```bash
ruff check .
ruff format --check .
pytest
```

## Pull requests

- Keep changes focused and describe their scientific or engineering rationale.
- Add tests for behavior changes and bug fixes.
- Use English for code, comments, logs, documentation, and configuration keys.
- Do not commit controlled-access data, large biological data files,
  checkpoints, credentials, or machine-specific configuration files.
- Document changes that affect HDF5 schemas, model outputs, or default
  preprocessing parameters.

## Reporting issues

Include the SPARTA version, operating system, Python and PyTorch versions,
configuration excerpt, complete error message, and the smallest reproducible
example that can be shared.

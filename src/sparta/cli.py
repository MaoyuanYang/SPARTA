"""Command-line interface for SPARTA."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence

from sparta.config import load_config, validate_config
from sparta.logging_utils import configure_logging
from sparta.pipeline import ALL_STAGES, run_pipeline

LOGGER = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sparta",
        description="Configuration-driven SPARTA data construction and training.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="Validate a SPARTA YAML configuration.")
    validate_parser.add_argument("--config", required=True, help="Path to the YAML configuration.")
    validate_parser.add_argument("--check-files", action="store_true", help="Require local input files.")

    run_parser = subparsers.add_parser("run", help="Run one pipeline stage or the full workflow.")
    run_parser.add_argument("--config", required=True, help="Path to the YAML configuration.")
    run_parser.add_argument(
        "--stage",
        choices=(*ALL_STAGES, "all"),
        default="all",
        help="Pipeline stage to run.",
    )
    run_parser.add_argument("--force", action="store_true", help="Rebuild existing stage outputs.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)

    try:
        config = load_config(args.config)
        if args.command == "validate":
            report = validate_config(config, check_files=args.check_files)
            for warning in report.warnings:
                LOGGER.warning(warning)
            report.raise_for_errors()
            LOGGER.info("Configuration is valid: %s", config.source_path)
            return 0

        run_pipeline(config, stage=args.stage, force=args.force)
        return 0
    except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
        LOGGER.error("%s", error)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

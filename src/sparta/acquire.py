"""Optional acquisition of explicitly configured public resources."""

from __future__ import annotations

import hashlib
import logging
import shutil
import urllib.request
from pathlib import Path
from typing import Any

from sparta.config import SpartaConfig

LOGGER = logging.getLogger(__name__)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(entry_name: str, entry: dict[str, Any], target: Path, force: bool) -> None:
    if target.exists() and not force:
        LOGGER.info("Using existing resource %s: %s", entry_name, target)
        return
    url = entry.get("url")
    if not url:
        raise FileNotFoundError(
            f"Resource '{entry_name}' is missing and has no direct download URL: {target}. "
            "Download it from the configured source_page or provide a url."
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.part")
    LOGGER.info("Downloading %s from %s", entry_name, url)
    request = urllib.request.Request(url, headers={"User-Agent": "SPARTA/1.0"})
    try:
        with urllib.request.urlopen(request) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        expected = entry.get("sha256")
        if expected:
            observed = _sha256(temporary)
            if observed.lower() != str(expected).lower():
                raise ValueError(f"SHA-256 mismatch for '{entry_name}': expected {expected}, observed {observed}")
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()


def acquire_resources(config: SpartaConfig, force: bool = False) -> None:
    """Download missing resources only from explicit URLs."""

    for name, entry in config.section("resources").items():
        _download(name, entry, config.resolve_path(entry["path"]), force)
    for name, entry in config.section("epigenomic_tracks").items():
        _download(f"epigenomic_tracks.{name}", entry, config.resolve_path(entry["path"]), force)

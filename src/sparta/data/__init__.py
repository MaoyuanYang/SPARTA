"""Data curation and dataset utilities for SPARTA."""

MIN_PEAK_SPACING_BP = 512


def normalize_chromosome(value: str) -> str:
    """Normalize chromosome names to the `chrN` convention."""

    name = str(value).strip()
    if name.lower().startswith("chr"):
        name = name[3:]
    if name.upper() in {"M", "MT"}:
        return "chrM"
    return f"chr{name}"

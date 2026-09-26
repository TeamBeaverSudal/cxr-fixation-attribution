"""Minimal adapters for the credentialed REFLACX Phase-3 file layout."""

from __future__ import annotations

from pathlib import Path

import numpy as np


RECORD_FILES = (
    "fixations.csv",
    "anomaly_location_ellipses.csv",
    "timestamps_transcription.csv",
)


def _col(dataframe, *candidates):
    """Resolve one schema-compatible column name."""
    for candidate in candidates:
        if candidate in dataframe.columns:
            return candidate
    raise KeyError(
        f"none of {candidates} found in columns {list(dataframe.columns)}"
    )


def _seconds(values):
    """Return timestamp values in seconds, accepting millisecond exports."""
    values = np.asarray(values, dtype=float)
    if len(values) > 1 and values[-1] - values[0] > 3600:
        return values / 1000.0
    return values


def find_records(root):
    """Index required per-reading CSV files by their REFLACX record directory.

    REFLACX places the three study inputs in one directory per reading. The
    recursive index tolerates the versioned parent directories used by the
    PhysioNet release while rejecting ambiguous duplicate record/file pairs.
    """
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"REFLACX root is not a directory: {root}")

    records = {}
    for filename in RECORD_FILES:
        for path in sorted(root.rglob(filename)):
            record_id = path.parent.name
            files = records.setdefault(record_id, {})
            existing = files.get(filename)
            if existing is not None and existing != path:
                raise RuntimeError(
                    f"ambiguous {filename} for record {record_id}: "
                    f"{existing} and {path}"
                )
            files[filename] = path
    return records

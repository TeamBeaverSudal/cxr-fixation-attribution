#!/usr/bin/env python3
"""Run one analysis module while preserving its native command-line options."""

from __future__ import annotations

import argparse
import runpy
import sys


ANALYSES = {
    "structured-comparison": "finding_level_gaze_targets.experiments.structured_comparison",
    "primary": "finding_level_gaze_targets.experiments.strongest_window_inference",
    "feature-controls": "finding_level_gaze_targets.experiments.refined_within_record_controls",
    "record-substitution": "finding_level_gaze_targets.experiments.matched_other_patient_scanpath",
    "training-fraction": "finding_level_gaze_targets.experiments.annotation_fraction_sensitivity",
    "architecture-selection": "finding_level_gaze_targets.experiments.architecture_selection",
    "linker-audit": "finding_level_gaze_targets.experiments.linker_audit",
    "aggregate": "finding_level_gaze_targets.experiments.summarize_consistent_runs",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", choices=sorted(ANALYSES))
    if len(sys.argv) == 1 or sys.argv[1] in {"-h", "--help"}:
        parser.print_help()
        return
    if sys.argv[1] not in ANALYSES:
        parser.error(
            f"unknown analysis {sys.argv[1]!r}; choose from "
            + ", ".join(sorted(ANALYSES))
        )
    analysis, remainder = sys.argv[1], sys.argv[2:]
    sys.argv = [ANALYSES[analysis], *remainder]
    runpy.run_module(ANALYSES[analysis], run_name="__main__")


if __name__ == "__main__":
    main()

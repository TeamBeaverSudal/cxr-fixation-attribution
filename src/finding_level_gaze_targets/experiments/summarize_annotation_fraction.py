#!/usr/bin/env python3
"""Strictly aggregate the crossed training-fraction experiment.

The study design crosses five nested patient-subsample chains with five
optimizer seeds. Optimizer seeds are repeated computations, not independent
observations: learned metrics are averaged within each chain before the five
chain means are summarized. The full-data condition has one patient set and is
summarized across its five optimizer seeds.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


RUN_NAME = "training-fraction/subset-{subset}/seed-{model}"
RESULT = re.compile(
    r"FRACTION_RESULT method=(?P<method>STRUCTURED|LEARNED) "
    r"fraction=(?P<fraction>[0-9.]+) subset_seed=(?P<subset>\d+) "
    r"n=\d+ pg=(?P<pg>[0-9.]+) iou=(?P<iou>[0-9.]+)"
)
LOOKBACK = re.compile(
    r"FRACTION_RESULT method=STRUCTURED fraction=(?P<fraction>[0-9.]+) "
    r"subset_seed=(?P<subset>\d+).* lookback=(?P<lookback>[0-9.]+)"
)


def _summary(values: list[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=float)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "sample_sd": float(array.std(ddof=1)) if array.size > 1 else None,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _run_dir(root: Path, subset: int, model: int) -> Path:
    return root / RUN_NAME.format(subset=subset, model=model)


def aggregate(root: Path) -> dict:
    values = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    lookbacks = defaultdict(lambda: defaultdict(list))

    for subset in range(5):
        for model in range(5):
            run_dir = _run_dir(root, subset, model)
            if not (run_dir / "COMPLETE").is_file():
                raise RuntimeError(f"run is not complete: {run_dir.name}")
            text = (run_dir / "run.log").read_text(errors="replace")
            for match in RESULT.finditer(text):
                fraction = f"{float(match['fraction']):.2f}"
                method = match["method"].lower()
                reported_subset = int(match["subset"])
                if reported_subset != subset:
                    raise RuntimeError(
                        f"subset mismatch in {run_dir.name}: {reported_subset}"
                    )
                for metric in ("pg", "iou"):
                    values[(method, fraction)][subset][metric].append(
                        (model, float(match[metric]))
                    )
            for match in LOOKBACK.finditer(text):
                fraction = f"{float(match['fraction']):.2f}"
                lookbacks[fraction][subset].append(
                    (model, float(match["lookback"]))
                )

    output: dict[str, dict] = {}
    for fraction in ("0.10", "0.25", "0.50", "1.00"):
        expected_subsets = {0} if fraction == "1.00" else set(range(5))
        output[fraction] = {}
        for method in ("structured", "learned"):
            subsets = values.get((method, fraction), {})
            if set(subsets) != expected_subsets:
                raise RuntimeError(
                    f"incomplete patient chains: method={method} fraction={fraction}"
                )
            output[fraction][method] = {}
            for metric in ("pg", "iou"):
                chain_means: list[float] = []
                optimizer_sds: list[float] = []
                for subset in sorted(subsets):
                    ordered = sorted(subsets[subset][metric])
                    seeds = [seed for seed, _ in ordered]
                    metric_values = [value for _, value in ordered]
                    if method == "structured":
                        if seeds != [0]:
                            raise RuntimeError(
                                "structured result must occur once at model seed 0: "
                                f"fraction={fraction} subset={subset}"
                            )
                    elif seeds != [0, 1, 2, 3, 4]:
                        raise RuntimeError(
                            "learned result must contain optimizer seeds 0--4: "
                            f"fraction={fraction} subset={subset}"
                        )
                    chain_means.append(float(np.mean(metric_values)))
                    if len(metric_values) > 1:
                        optimizer_sds.append(float(np.std(metric_values, ddof=1)))
                summary = _summary(chain_means)
                if optimizer_sds:
                    summary["mean_within_chain_optimizer_sd"] = float(
                        np.mean(optimizer_sds)
                    )
                    summary["within_chain_optimizer_sd_range"] = [
                        float(np.min(optimizer_sds)),
                        float(np.max(optimizer_sds)),
                    ]
                output[fraction][method][metric] = summary

        selected = lookbacks.get(fraction, {})
        if set(selected) != expected_subsets:
            raise RuntimeError(f"structured lookbacks are incomplete: {fraction}")
        by_subset: list[float] = []
        for subset in sorted(selected):
            ordered = sorted(selected[subset])
            if len(ordered) != 1 or ordered[0][0] != 0:
                raise RuntimeError(
                    "structured lookback must occur once at model seed 0: "
                    f"fraction={fraction} subset={subset}"
                )
            by_subset.append(ordered[0][1])
        output[fraction]["structured"]["selected_lookback_seconds"] = {
            "by_patient_subset": by_subset,
            **_summary(by_subset),
        }

    return {
        "status": "complete",
        "design": {
            "patient_subsample_chains": 5,
            "optimizer_seeds_per_chain": 5,
            "aggregation_order": "optimizer_seeds_within_chain_then_patient_chains",
        },
        "fractions": output,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="server runs directory")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rendered = json.dumps(aggregate(args.root), indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()

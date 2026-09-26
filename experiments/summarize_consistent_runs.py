#!/usr/bin/env python3
"""Aggregate identifier-free outputs from the consistent-seed reanalysis.

The script never treats optimizer seeds as independent test observations.
Partial training fractions are first averaged over optimizer seeds within each
patient-subsample chain and are then summarized across the five chains.
The full-data row is summarized across optimizer seeds because its training
patients do not change.  Patient partitions remain separate sensitivity runs.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


RESULT = re.compile(
    r"FRACTION_RESULT method=(?P<method>\w+) fraction=(?P<fraction>[0-9.]+) "
    r"subset_seed=(?P<subset>\d+) n=\d+ pg=(?P<pg>[0-9.]+) "
    r"iou=(?P<iou>[0-9.]+)"
)
FRACTION_LOOKBACK = re.compile(
    r"FRACTION_RESULT method=STRUCTURED fraction=(?P<fraction>[0-9.]+) "
    r"subset_seed=(?P<subset>\d+).* lookback=(?P<lookback>[0-9.]+)"
)
SELECTOR = re.compile(
    r"SELECTOR_MEAN method=(?P<method>\w+) seeds=(?P<seeds>\d+) n=(?P<n>\d+) "
    r"pg=(?P<pg>[0-9.]+) pg_sd=(?P<pg_sd>[0-9.]+).*"
    r"iou=(?P<iou>[0-9.]+) iou_sd=(?P<iou_sd>[0-9.]+)"
)
STRUCTURED = re.compile(
    r"SELECTOR method=combined_structured_validation_selected n=(?P<n>\d+) "
    r"lookback=(?P<lookback>[0-9.]+) pg=(?P<pg>[0-9.]+) "
    r"iou=(?P<iou>[0-9.]+)"
)
DONOR = re.compile(
    r"DONORTEST estimator=per_instance_seed_mean seeds=(?P<seeds>\d+) "
    r"metric=(?P<metric>\w+) n_inst=(?P<n>\d+) n_patients=(?P<patients>\d+) "
    r"learned=(?P<learned>[0-9.]+).*donor=(?P<other>[0-9.]+).*"
    r"mean_delta=(?P<delta>[+-][0-9.]+) ci=\[(?P<lo>[+-][0-9.]+),"
    r"(?P<hi>[+-][0-9.]+)\]"
)
DONOR_SELECT = re.compile(
    r"DONORSELECT validation_primary=pointing seeds=(?P<seeds>\d+) "
    r"maximum_by_seed=(?P<maximums>[0-9,]+) "
    r"coverage=(?P<coverage>[0-9.]+) "
    r"eligible=(?P<eligible>\d+)/(?P<total>\d+)"
)
INFERENCE = re.compile(
    r"INFERENCE estimator=per_instance_seed_mean seeds=(?P<seeds>\d+) "
    r"comparison=(?P<comparison>\w+) metric=(?P<metric>\w+) n=(?P<n>\d+) "
    r"patients=(?P<patients>\d+) delta=(?P<delta>[+-][0-9.]+) "
    r"ci=\[(?P<lo>[+-][0-9.]+),(?P<hi>[+-][0-9.]+)\] "
    r"patient_wilcoxon_p=(?P<p>[0-9.eE+-]+)"
)


def _mean_sd(values: list[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=float)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "sd": float(array.std(ddof=1)) if array.size > 1 else None,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _require_complete(run_dir: Path) -> None:
    if not (run_dir / "COMPLETE").is_file():
        raise RuntimeError(f"run is not complete: {run_dir.name}")


def fraction_summary(root: Path) -> dict:
    for subset_seed in range(5):
        for model_seed in range(5):
            _require_complete(
                root
                / (
                    "reviewer-consistent-fraction-"
                    f"subset-{subset_seed}-model-{model_seed}-20260818"
                )
            )
    values = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    lookbacks = defaultdict(lambda: defaultdict(list))
    for log in sorted(root.glob("reviewer-consistent-fraction-*/run.log")):
        if "superseded" in log.parent.name:
            continue
        model_match = re.search(r"-model-(\d+)-", log.parent.name)
        if not model_match:
            continue
        model_seed = int(model_match.group(1))
        text = log.read_text(errors="replace")
        for match in RESULT.finditer(text):
            key = (match["method"].lower(), match["fraction"])
            subset = int(match["subset"])
            for metric in ("pg", "iou"):
                values[key][subset][metric].append(
                    (model_seed, float(match[metric]))
                )
        for match in FRACTION_LOOKBACK.finditer(text):
            lookbacks[match["fraction"]][int(match["subset"])].append(
                (model_seed, float(match["lookback"]))
            )

    if set(fraction for _method, fraction in values) != {"0.10", "0.25", "0.50", "1.00"}:
        raise RuntimeError("training-fraction outputs are incomplete")

    output = {}
    for (method, fraction), subsets in sorted(values.items()):
        fraction_key = f"{float(fraction):.2f}"
        expected_subsets = {0} if fraction_key == "1.00" else set(range(5))
        if set(subsets) != expected_subsets:
            raise RuntimeError(
                f"incomplete patient-subset chains: {method} fraction={fraction_key}"
            )
        output.setdefault(fraction_key, {})[method] = {}
        for metric in ("pg", "iou"):
            chain_means = []
            within_seed_sd = []
            for subset in sorted(subsets):
                ordered = sorted(subsets[subset][metric])
                metric_values = [value for _, value in ordered]
                if method == "structured":
                    if len(metric_values) != 1 or ordered[0][0] != 0:
                        raise RuntimeError(
                            "structured result must be emitted once by model seed 0"
                        )
                    chain_means.append(metric_values[0])
                else:
                    if len(metric_values) != 5:
                        raise RuntimeError(
                            f"incomplete optimizer seeds: {fraction_key} subset={subset}"
                        )
                    chain_means.append(float(np.mean(metric_values)))
                    within_seed_sd.append(float(np.std(metric_values, ddof=1)))
            summary = _mean_sd(chain_means)
            if within_seed_sd:
                summary["mean_within_chain_optimizer_sd"] = float(
                    np.mean(within_seed_sd)
                )
                summary["within_chain_optimizer_sd_range"] = [
                    float(np.min(within_seed_sd)),
                    float(np.max(within_seed_sd)),
                ]
            output[fraction_key][method][metric] = summary

    for fraction, subsets in lookbacks.items():
        fraction_key = f"{float(fraction):.2f}"
        expected_subsets = {0} if fraction_key == "1.00" else set(range(5))
        if set(subsets) != expected_subsets:
            raise RuntimeError(
                f"incomplete structured lookbacks: fraction={fraction_key}"
            )
        selected = []
        for subset in sorted(subsets):
            ordered = sorted(subsets[subset])
            if len(ordered) != 1 or ordered[0][0] != 0:
                raise RuntimeError(
                    f"structured lookback must be emitted once: "
                    f"fraction={fraction_key} subset={subset}"
                )
            selected.append(ordered[0][1])
        output[fraction_key]["structured"]["selected_lookback_seconds"] = {
            "by_patient_subset": selected,
            **_mean_sd(selected),
        }
    return output


def partition_summary(root: Path) -> dict:
    for split_seed in range(5):
        _require_complete(
            root / f"reviewer-consistent-partition-{split_seed}-20260818"
        )
    partitions = {}
    for log in sorted(root.glob("reviewer-consistent-partition-*/run.log")):
        split_match = re.search(r"partition-(\d+)-", log.parent.name)
        if not split_match:
            continue
        split = split_match.group(1)
        text = log.read_text(errors="replace")
        learned = next(
            (match for match in SELECTOR.finditer(text) if match["method"] == "full"),
            None,
        )
        structured = next(STRUCTURED.finditer(text), None)
        if learned is None or structured is None:
            continue
        partitions[split] = {
            "n": int(learned["n"]),
            "seeds": int(learned["seeds"]),
            "selected_lookback": float(structured["lookback"]),
            "learned_pg": float(learned["pg"]),
            "learned_pg_optimizer_sd": float(learned["pg_sd"]),
            "learned_iou": float(learned["iou"]),
            "learned_iou_optimizer_sd": float(learned["iou_sd"]),
            "structured_pg": float(structured["pg"]),
            "structured_iou": float(structured["iou"]),
            "delta_pg": float(learned["pg"]) - float(structured["pg"]),
            "delta_iou": float(learned["iou"]) - float(structured["iou"]),
        }
    if set(partitions) != {"0", "1", "2", "3", "4"}:
        raise RuntimeError("patient-partition outputs are incomplete")
    if any(value["seeds"] != 5 for value in partitions.values()):
        raise RuntimeError("a patient partition does not contain five optimizer seeds")
    return partitions


def donor_summary(root: Path) -> dict:
    run_dir = root / "reviewer-consistent-other-patient-20260818"
    _require_complete(run_dir)
    log = run_dir / "run.log"
    text = log.read_text(errors="replace")
    selection_match = DONOR_SELECT.search(text)
    if selection_match is None:
        raise RuntimeError("other-patient selection output is missing")
    maxima = [int(value) for value in selection_match["maximums"].split(",")]
    if (
        int(selection_match["seeds"]) != 5
        or len(maxima) != 5
        or int(selection_match["eligible"]) != 948
        or int(selection_match["total"]) != 987
    ):
        raise RuntimeError("other-patient selection contract changed")
    output = {}
    for match in DONOR.finditer(text):
        output[match["metric"]] = {
            "seeds": int(match["seeds"]),
            "n": int(match["n"]),
            "patients": int(match["patients"]),
            "target_record": float(match["learned"]),
            "other_patient_record": float(match["other"]),
            "difference": float(match["delta"]),
            "ci": [float(match["lo"]), float(match["hi"])],
        }
    if set(output) != {"pg", "iou"}:
        raise RuntimeError("other-patient control output is incomplete")
    if any(value["seeds"] != 5 for value in output.values()):
        raise RuntimeError("other-patient control does not contain five seeds")
    output["selection"] = {
        "seeds": int(selection_match["seeds"]),
        "maximum_records_by_seed": maxima,
        "coverage": float(selection_match["coverage"]),
        "eligible": int(selection_match["eligible"]),
        "total": int(selection_match["total"]),
    }
    return output


def primary_summary(root: Path) -> dict:
    run_dir = root / "reviewer-consistent-primary-20260818"
    _require_complete(run_dir)
    log = run_dir / "run.log"
    text = log.read_text(errors="replace")
    output = {"models": {}, "inference": {}}
    for match in SELECTOR.finditer(text):
        output["models"][match["method"]] = {
            "seeds": int(match["seeds"]),
            "n": int(match["n"]),
            "pointing": float(match["pg"]),
            "pointing_optimizer_sd": float(match["pg_sd"]),
            "iou": float(match["iou"]),
            "iou_optimizer_sd": float(match["iou_sd"]),
        }
    structured = next(STRUCTURED.finditer(text), None)
    if structured is not None:
        output["structured"] = {
            "n": int(structured["n"]),
            "selected_lookback": float(structured["lookback"]),
            "pointing": float(structured["pg"]),
            "iou": float(structured["iou"]),
        }
    for match in INFERENCE.finditer(text):
        output["inference"].setdefault(match["comparison"], {})[
            match["metric"]
        ] = {
            "seeds": int(match["seeds"]),
            "n": int(match["n"]),
            "patients": int(match["patients"]),
            "difference": float(match["delta"]),
            "ci": [float(match["lo"]), float(match["hi"])],
            "patient_wilcoxon_p": float(match["p"]),
        }
    if set(output["models"]) != {"full", "four_indicator"}:
        raise RuntimeError("primary learned-model outputs are incomplete")
    if any(
        value["seeds"] != 5 or value["n"] != 987
        for value in output["models"].values()
    ):
        raise RuntimeError("primary learned-model contract changed")
    if "structured" not in output:
        raise RuntimeError("primary structured output is missing")
    if (
        output["structured"]["n"] != 987
        or output["structured"]["selected_lookback"] != 3.0
    ):
        raise RuntimeError("primary structured-reference contract changed")
    expected = {
        "full_minus_selected_structured",
        "four_indicator_minus_selected_structured",
        "four_indicator_minus_full",
    }
    if set(output["inference"]) != expected:
        raise RuntimeError("primary inference outputs are incomplete")
    for comparison in output["inference"].values():
        if set(comparison) != {"pg", "iou"}:
            raise RuntimeError("primary inference metric is missing")
        if any(
            value["seeds"] != 5
            or value["n"] != 987
            or value["patients"] != 398
            for value in comparison.values()
        ):
            raise RuntimeError("primary inference contract changed")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = {
        "estimand": (
            "per-instance mean across five optimizer seeds; patients are the "
            "resampling unit for paired inference"
        ),
        "primary": primary_summary(arguments.root),
        "training_fraction": fraction_summary(arguments.root),
        "patient_partitions": partition_summary(arguments.root),
        "other_patient_control": donor_summary(arguments.root),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    print(rendered)
    if arguments.output is not None:
        arguments.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()

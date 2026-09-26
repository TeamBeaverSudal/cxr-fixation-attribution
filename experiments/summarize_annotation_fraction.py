#!/usr/bin/env python3
"""Summarize identifier-free annotation-fraction logs across patient subsets."""

from __future__ import annotations

import argparse
import re
import statistics
from pathlib import Path


SUBSET = re.compile(
    r"SUBSET fraction=(?P<fraction>[0-9.]+) subset_seed=(?P<seed>\d+) "
    r"patients=(?P<patients>\d+)/(?P<all_patients>\d+) "
    r"instances=(?P<instances>\d+)/(?P<all_instances>\d+) "
    r"ellipses=(?P<ellipses>\d+) labels=(?P<labels>\d+)/(?P<all_labels>\d+)"
)
RESULT = re.compile(
    r"FRACTION_RESULT method=(?P<method>STRUCTURED|LEARNED) "
    r"fraction=(?P<fraction>[0-9.]+) subset_seed=(?P<seed>\d+) "
    r"n=(?P<n>\d+) pg=(?P<pg>[0-9.]+) iou=(?P<iou>[0-9.]+)"
)


def _summary(values):
    values = [float(value) for value in values]
    if len(values) == 1:
        return f"{values[0]:.4f}"
    return (
        f"{statistics.fmean(values):.4f} "
        f"(SD {statistics.stdev(values):.4f}; "
        f"range [{min(values):.4f},{max(values):.4f}])"
    )


def main(paths, temporal_pg, temporal_iou):
    rows = {}
    for path in paths:
        text = Path(path).read_text()
        for match in SUBSET.finditer(text):
            key = (int(match["seed"]), float(match["fraction"]))
            rows.setdefault(key, {}).update(
                {name: int(match[name]) for name in (
                    "patients", "all_patients", "instances", "all_instances",
                    "ellipses", "labels", "all_labels"
                )}
            )
        for match in RESULT.finditer(text):
            key = (int(match["seed"]), float(match["fraction"]))
            rows.setdefault(key, {})[match["method"].lower()] = {
                "pg": float(match["pg"]), "iou": float(match["iou"])
            }

    complete = {
        key: value for key, value in rows.items()
        if "learned" in value and "structured" in value
    }
    fractions = sorted({fraction for _, fraction in complete})
    full_rows = [value for (seed, fraction), value in complete.items() if fraction == 1.0]
    if len(full_rows) != 1:
        raise SystemExit(f"expected one complete full-data row, found {len(full_rows)}")
    full = full_rows[0]["learned"]

    print(
        "fraction repeats train_instances train_ellipses structured_pg learned_pg "
        "delta_pg structured_iou learned_iou delta_iou retained_gain_pg retained_gain_iou"
    )
    for fraction in fractions:
        values = [value for (seed, frac), value in sorted(complete.items()) if frac == fraction]
        structured_pg = [value["structured"]["pg"] for value in values]
        learned_pg = [value["learned"]["pg"] for value in values]
        structured_iou = [value["structured"]["iou"] for value in values]
        learned_iou = [value["learned"]["iou"] for value in values]
        mean_learned_pg = statistics.fmean(learned_pg)
        mean_learned_iou = statistics.fmean(learned_iou)
        gain_pg = (mean_learned_pg - temporal_pg) / (full["pg"] - temporal_pg)
        gain_iou = (mean_learned_iou - temporal_iou) / (full["iou"] - temporal_iou)
        print(
            f"{fraction:.2f} {len(values)} "
            f"{_summary([value['instances'] for value in values])} "
            f"{_summary([value['ellipses'] for value in values])} "
            f"{_summary(structured_pg)} {_summary(learned_pg)} "
            f"{statistics.fmean(a-b for a, b in zip(learned_pg, structured_pg)):+.4f} "
            f"{_summary(structured_iou)} {_summary(learned_iou)} "
            f"{statistics.fmean(a-b for a, b in zip(learned_iou, structured_iou)):+.4f} "
            f"{gain_pg:.3f} {gain_iou:.3f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+")
    parser.add_argument("--temporal-pg", type=float, default=0.6211)
    parser.add_argument("--temporal-iou", type=float, default=0.2773)
    args = parser.parse_args()
    main(args.logs, args.temporal_pg, args.temporal_iou)

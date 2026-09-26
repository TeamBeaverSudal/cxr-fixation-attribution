"""Load, validate, and summarize the camera-ready aggregate result registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_registry() -> Path:
    return repository_root() / "results" / "medai2026-camera-ready.json"


def load_registry(path: Path | None = None) -> dict:
    return json.loads((path or default_registry()).read_text(encoding="utf-8"))


def validate_registry(data: dict) -> list[str]:
    errors: list[str] = []
    cohort = data.get("cohort", {})
    linked = cohort.get("mention_linked_instances", {})
    if linked != {"train": 1895, "validation": 547, "test": 987}:
        errors.append("mention-linked train/validation/test cohorts changed")
    if cohort.get("test_patients") != 398:
        errors.append("the primary test cohort must contain 398 patients")

    estimator = data.get("estimator", {})
    if estimator.get("optimizer_seeds") != [0, 1, 2, 3, 4]:
        errors.append("optimizer seeds must be 0--4")
    if estimator.get("bootstrap", {}).get("unit") != "patient":
        errors.append("the bootstrap unit must be the patient")

    table_1 = {row.get("method"): row for row in data.get("table_1", [])}
    structured = table_1.get("combined_structured_3_0s", {})
    learned = table_1.get("ten_indicator_learned_five_seed_mean", {})
    if (structured.get("pointing"), structured.get("iou")) != (0.7893, 0.3549):
        errors.append("the validation-selected structured result changed")
    if (learned.get("pointing"), learned.get("iou")) != (0.8245, 0.3584):
        errors.append("the five-seed learned result changed")

    partitions = data.get("patient_partitions", [])
    if [row.get("partition") for row in partitions] != [0, 1, 2, 3, 4]:
        errors.append("all five patient partitions are required")
    if any(row.get("optimizer_seeds") != 5 for row in partitions):
        errors.append("every patient partition must contain five optimizer seeds")

    table_4 = data.get("table_4", {})
    if table_4.get("completed_tasks") != 25:
        errors.append("all 25 crossed training-fraction tasks are required")
    if [row.get("fraction") for row in table_4.get("rows", [])] != [0.1, 0.25, 0.5, 1.0]:
        errors.append("training fractions must be 10%, 25%, 50%, and 100%")
    return errors


def summary(data: dict) -> str:
    table_1 = {row["method"]: row for row in data["table_1"]}
    structured = table_1["combined_structured_3_0s"]
    learned = table_1["ten_indicator_learned_five_seed_mean"]
    inference = data["primary_inference"]["learned_minus_structured_3_0s"]
    return (
        "MedAI 2026 finding-level gaze targets (987 instances; 398 patients)\n"
        f"  structured 3.0 s: PG {structured['pointing']:.4f}; IoU {structured['iou']:.4f}\n"
        f"  learned five-seed: PG {learned['pointing']:.4f}; IoU {learned['iou']:.4f}\n"
        f"  learned - structured: PG {inference['pointing']['difference']:+.4f}; "
        f"IoU {inference['iou']['difference']:+.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    data = load_registry(args.registry)
    errors = validate_registry(data)
    if errors:
        raise SystemExit("\n".join(errors))
    print(summary(data) if args.summary else "camera-ready result registry: valid")


if __name__ == "__main__":
    main()

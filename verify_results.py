#!/usr/bin/env python3
"""Validate the frozen reference-study results and publication linkage.

This offline check compares the public aggregate with independently encoded
expectations and verifies the study's cohort, estimators, controls, and
sensitivity analyses. Supplying ``--pdf`` also verifies the exact associated
publication PDF hash and key printed tokens.

Model reruns from credentialed data are documented in docs/REPRODUCIBILITY.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS = ROOT / "results" / "study-results.json"
ASSOCIATED_PDF_SHA256 = "05ddf457d5b0ce73e120083f6ee5c5cf502320e2a43410bd3b65636070865327"
ASSOCIATED_TEX_SHA256 = "bddc681ace1b69795f074d2e98a767625a02ec809f0bd1818a1d21ab6746767b"
ASSOCIATED_BIBLIOGRAPHY_SHA256 = "2c829a163c352b3c259523424414a63b5d6f49c87929dae26cb47b3b2c280b79"
PUBLICATION_TITLE = (
    "From Complete Scanpaths to Finding-Level Gaze Targets in Chest Radiography: "
    "Structured Cues and Learned Reweighting"
)


# These values are intentionally independent of the JSON registry. A changed
# result artifact must therefore update an explicit study contract, not merely
# rewrite the file that the checker reads.
EXPECTED = {
    "associated_publication.status": "accepted_camera_ready_candidate",
    "associated_publication.title": PUBLICATION_TITLE,
    "associated_publication.pdf_sha256": ASSOCIATED_PDF_SHA256,
    "associated_publication.tex_sha256": ASSOCIATED_TEX_SHA256,
    "associated_publication.bibliography_sha256": ASSOCIATED_BIBLIOGRAPHY_SHA256,
    "associated_publication.pages": 7,
    "estimator.optimizer_seeds": [0, 1, 2, 3, 4],
    "estimator.primary_estimand": "instance_weighted_mean_difference",
    "estimator.bootstrap.unit": "patient",
    "estimator.bootstrap.resamples": 10000,
    "cohort.mention_linked_instances.train": 1895,
    "cohort.mention_linked_instances.validation": 547,
    "cohort.mention_linked_instances.test": 987,
    "cohort.annotated_test_instances_before_linking": 1093,
    "cohort.test_patients": 398,
    "cohort.eligible_training_patients": 735,
    "cohort.training_instances_without_in_region_fixation": 4,
    "cohort.test_instances_with_in_region_fixation": 982,
    "record_substitution.eligible_instances": 948,
    "record_substitution.eligible_patients": 389,
    "record_substitution.matched_records_selected_per_seed": [8, 8, 8, 8, 8],
    "primary_inference.learned_minus_structured_3_0s.pointing.difference": 0.0353,
    "primary_inference.learned_minus_structured_3_0s.pointing.ci95": [0.0067, 0.0634],
    "primary_inference.learned_minus_structured_3_0s.pointing.patient_signed_rank_p": 0.0766815,
    "primary_inference.learned_minus_structured_3_0s.iou.difference": 0.0035,
    "primary_inference.learned_minus_structured_3_0s.iou.ci95": [-0.0034, 0.0102],
    "primary_inference.learned_minus_structured_3_0s.iou.patient_signed_rank_p": 0.841748,
    "primary_inference.four_indicator_minus_ten_indicator.pointing.difference": 0.0128,
    "primary_inference.four_indicator_minus_ten_indicator.pointing.ci95": [0.0006, 0.025],
    "primary_inference.four_indicator_minus_ten_indicator.iou.difference": -0.001,
    "primary_inference.four_indicator_minus_ten_indicator.iou.ci95": [-0.0042, 0.0023],
    "record_substitution.target_record.pointing": 0.8281,
    "record_substitution.target_record.iou": 0.3605,
    "record_substitution.matched_other_patient_records.pointing": 0.5464,
    "record_substitution.matched_other_patient_records.iou": 0.2836,
    "record_substitution.target_minus_substitution.pointing.difference": 0.2816,
    "record_substitution.target_minus_substitution.pointing.ci95": [0.2473, 0.3159],
    "record_substitution.target_minus_substitution.iou.difference": 0.0769,
    "record_substitution.target_minus_substitution.iou.ci95": [0.0653, 0.0885],
    "table_4.completed_tasks": 25,
    "table_4.patient_subsample_chains": 5,
    "table_4.optimizer_seeds_per_chain": 5,
    "execution_provenance.deployment_commit": "45ada8abc9b7e3ec98b37b4dcb58f0521c67cef8",
    "execution_provenance.deployment_commit_scope": "private_execution_repository",
    "execution_provenance.strict_aggregate_sha256": (
        "e8e54ebc4ad4d83cef88bfd18c487b94267c9b78c85d6982262bf6f48758a71a"
    ),
}

EXPECTED_TABLE_1 = [
    ("complete_scanpath_density", 0.4063, 0.2008),
    ("anatomical_prior", 0.5035, 0.2736),
    ("temporal_1_5s", 0.6211, 0.2773),
    ("prior_x_scanpath", 0.6717, 0.3056),
    ("prior_x_scanpath_directional", 0.7183, 0.3408),
    ("prior_x_scanpath_temporal", 0.7528, 0.3201),
    ("combined_structured_1_5s", 0.7923, 0.3439),
    ("combined_structured_3_0s", 0.7893, 0.3549),
    ("ten_indicator_learned_five_seed_mean", 0.8245, 0.3584),
]

EXPECTED_TABLE_2 = [
    (0.5, 0.3348, 0.7325, 0.3192),
    (1.0, 0.3528, 0.7761, 0.3345),
    (1.5, 0.3620, 0.7923, 0.3439),
    (2.0, 0.3695, 0.7984, 0.3492),
    (3.0, 0.3733, 0.7893, 0.3549),
]

EXPECTED_TABLE_3 = [
    ("ten_indicator_selector", 0.8245, 0.3584),
    ("four_indicator_selector", 0.8373, 0.3574),
    ("finding_temporal_kinematic_only", 0.7295, 0.3135),
    ("positional_features_permuted", 0.5534, 0.2306),
    ("temporal_kinematic_features_permuted", 0.6833, 0.2976),
    ("spatial_indicators_masked", 0.7495, 0.3068),
]

EXPECTED_TABLE_4 = [
    (0.10, 0.767740, 0.778120, 0.319076, 0.345680),
    (0.25, 0.802604, 0.790080, 0.335380, 0.351800),
    (0.50, 0.819048, 0.789480, 0.341512, 0.353480),
    (1.00, 0.824500, 0.789300, 0.358340, 0.354900),
]

EXPECTED_PARTITIONS = [
    (0, 987, 0.0352, 0.0035),
    (1, 1009, 0.0269, 0.0122),
    (2, 971, 0.0356, 0.0040),
    (3, 992, 0.0308, 0.0047),
    (4, 998, 0.0167, 0.0002),
]


def at(data, dotted):
    value = data
    for part in dotted.split("."):
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def require(condition, message, failures):
    if condition:
        return 1
    failures.append(message)
    return 0


def validate_registry(data):
    failures = []
    passed = 0
    for path, expected in EXPECTED.items():
        try:
            observed = at(data, path)
        except (KeyError, IndexError, TypeError) as exc:
            failures.append(f"missing {path}: {exc}")
            continue
        passed += require(
            observed == expected,
            f"{path}: {observed!r} != {expected!r}",
            failures,
        )

    got = [(r["method"], r["pointing"], r["iou"]) for r in data.get("table_1", [])]
    passed += require(got == EXPECTED_TABLE_1, "structured/learned comparison changed", failures)

    got = [
        (r["lookback_seconds"], r["validation_iou"], r["test_pointing"], r["test_iou"])
        for r in data.get("table_2", [])
    ]
    passed += require(
        got == EXPECTED_TABLE_2,
        "Table II does not match the completed lookback sweep",
        failures,
    )
    if got:
        selected = max(got, key=lambda row: row[1])[0]
        passed += require(
            selected == 3.0,
            f"validation IoU selects {selected}, not 3.0 s",
            failures,
        )

    got = [(r["condition"], r["pointing"], r["iou"]) for r in data.get("table_3", [])]
    passed += require(
        got == EXPECTED_TABLE_3,
        "Table III does not match the five-seed feature controls",
        failures,
    )

    got = [
        (
            r["fraction"],
            r["learned_pointing"],
            r["structured_pointing"],
            r["learned_iou"],
            r["structured_iou"],
        )
        for r in data.get("table_4", {}).get("rows", [])
    ]
    passed += require(
        got == EXPECTED_TABLE_4,
        "Table IV does not match the strict training-fraction aggregate",
        failures,
    )
    for row in data.get("table_4", {}).get("rows", []):
        passed += require(
            set(row["selected_lookbacks"]) == {3},
            f"fraction {row['fraction']} did not select only 3.0 s",
            failures,
        )

    got = [
        (r["partition"], r["test_instances"], r["delta_pointing"], r["delta_iou"])
        for r in data.get("patient_partitions", [])
    ]
    passed += require(
        got == EXPECTED_PARTITIONS,
        "patient partitions do not match the five-seed aggregate",
        failures,
    )
    for row in data.get("patient_partitions", []):
        passed += require(
            row.get("optimizer_seeds") == 5,
            f"partition {row.get('partition')} is not a five-seed run",
            failures,
        )

    hashes = data.get("execution_provenance", {}).get("executed_source_sha256", {})
    passed += require(
        len(hashes) == 4 and all(len(v) == 64 for v in hashes.values()),
        "execution source hashes are incomplete",
        failures,
    )
    return passed, failures


def validate_pdf(pdf_path):
    failures = []
    passed = 0
    digest = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    passed += require(
        digest == ASSOCIATED_PDF_SHA256,
        f"PDF SHA-256 {digest} != associated publication hash",
        failures,
    )

    pdftotext = shutil.which("pdftotext")
    if pdftotext is None:
        print("SKIP: pdftotext is unavailable; exact PDF hash still passed", file=sys.stderr)
        return passed, failures
    with tempfile.TemporaryDirectory(prefix="medai-pdf-") as tmp:
        output = Path(tmp) / "paper.txt"
        proc = subprocess.run(
            [pdftotext, str(pdf_path), str(output)],
            capture_output=True,
            text=True,
        )
        if proc.returncode:
            failures.append(f"pdftotext failed: {proc.stderr.strip()}")
            return passed, failures
        text = " ".join(output.read_text(errors="replace").split())

    tokens = [
        "987 REFLACX test instances",
        "spanning 398 patients",
        "0.035",
        "0.007– 0.063" if "0.007– 0.063" in text else "0.007–0.063",
        "0.077",
        "0.842",
        "948 of 987 test instances from 389 patients",
        "0.017–0.036",
    ]
    for token in tokens:
        passed += require(token in text, f"PDF is missing expected token: {token}", failures)
    return passed, failures


def validate_file_hash(path, expected, label):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    failures = (
        []
        if digest == expected
        else [f"{label} SHA-256 {digest} != associated publication hash"]
    )
    return int(not failures), failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--pdf", type=Path)
    parser.add_argument("--tex", type=Path)
    parser.add_argument("--bibliography", type=Path)
    parser.add_argument(
        "--all",
        action="store_true",
        help="compatibility alias; the complete contract is always checked",
    )
    args = parser.parse_args()

    data = json.loads(args.results.read_text())
    passed, failures = validate_registry(data)
    if args.pdf:
        pdf_passed, pdf_failures = validate_pdf(args.pdf)
        passed += pdf_passed
        failures.extend(pdf_failures)
    if args.tex:
        file_passed, file_failures = validate_file_hash(
            args.tex, ASSOCIATED_TEX_SHA256, "TeX source"
        )
        passed += file_passed
        failures.extend(file_failures)
    if args.bibliography:
        file_passed, file_failures = validate_file_hash(
            args.bibliography, ASSOCIATED_BIBLIOGRAPHY_SHA256, "bibliography"
        )
        passed += file_passed
        failures.extend(file_failures)

    if failures:
        print(f"FAIL: {len(failures)} assertion(s) failed after {passed} passes", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print(f"PASS: {passed} study-result contract assertions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

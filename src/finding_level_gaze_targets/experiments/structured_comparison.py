#!/usr/bin/env python3
"""Reproduce deterministic Table I rows and the complete lookback sweep.

Every construction uses the mention-resolved patient split, training-only
anatomical quantities, validation-selected calibration, and the common map
renderer. Raw transcripts are required because lookbacks beyond the cached
1.5-second window must recover the preceding-sentence clipping boundary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import zoom

from finding_level_gaze_targets.baselines.anatomy import (
    label_prior,
    prior_heat,
    split,
)
from finding_level_gaze_targets.baselines.structured import (
    apply_directional_terms,
    combined_structured_heat,
    complete_scanpath_heat,
    directional_statistics,
    scanpath_support_heat,
)
from finding_level_gaze_targets.data import reflacx
from finding_level_gaze_targets.linking.rules import (
    apply_lookback,
    assert_cached_lookback,
    reconstruct_base_mentions,
)
from finding_level_gaze_targets.maps.core import (
    EVAL_RES,
    TUNE_SIGMAS,
    iou,
    pointing,
    raster,
    temporal_window_heat,
    tune_thresholds,
    word_feat,
)


LOOKBACKS = (0.5, 1.0, 1.5, 2.0, 3.0)
THRESHOLDS = tune_thresholds()
DIRECTIONAL_INDICES = np.asarray([0, 1, 3, 4], dtype=int)
DEFAULT_REGISTRY = Path(__file__).resolve().parents[3] / "results/study-results.json"


def _iou_pre(upscaled, target, threshold):
    prediction = upscaled >= threshold
    intersection = (prediction & target).sum()
    union = prediction.sum() + target.sum() - intersection
    return intersection / union if union > 0 else np.nan


def _tune(map_functions, targets, sigmas=TUNE_SIGMAS):
    best = (sigmas[0], THRESHOLDS[0], -1.0)
    for sigma in sigmas:
        upscaled = []
        for make_map, target in zip(map_functions, targets):
            heatmap = make_map(sigma)
            if heatmap.shape[0] != EVAL_RES:
                heatmap = zoom(heatmap, EVAL_RES / heatmap.shape[0], order=0)
            upscaled.append((heatmap, target))
        for threshold in THRESHOLDS:
            value = np.nanmean(
                [_iou_pre(heatmap, target, threshold) for heatmap, target in upscaled]
            )
            if value > best[2]:
                best = (sigma, threshold, float(value))
    return best


def _score(map_functions, targets, sigma, threshold):
    heatmaps = [make_map(sigma) for make_map in map_functions]
    return {
        "pointing": float(
            np.nanmean(
                [pointing(heatmap, target) for heatmap, target in zip(heatmaps, targets)]
            )
        ),
        "iou": float(
            np.nanmean(
                [iou(heatmap, target, threshold) for heatmap, target in zip(heatmaps, targets)]
            )
        ),
    }


def _assert_registry_match(measured_table_1, measured_table_2, registry_path):
    registry = json.loads(Path(registry_path).read_text(encoding="utf-8"))
    expected_table_1 = {
        row["method"]: row for row in registry["table_1"] if "five_seed" not in row["method"]
    }
    expected_table_2 = {
        float(row["lookback_seconds"]): row for row in registry["table_2"]
    }

    if set(measured_table_1) != set(expected_table_1):
        raise RuntimeError("Table I method set does not match the result registry")
    for method, measured in measured_table_1.items():
        expected = expected_table_1[method]
        for metric in ("pointing", "iou"):
            if abs(measured[metric] - expected[metric]) > 5e-5:
                raise RuntimeError(
                    f"Table I mismatch for {method}/{metric}: "
                    f"measured={measured[metric]:.6f}, expected={expected[metric]:.6f}"
                )

    if set(measured_table_2) != set(expected_table_2):
        raise RuntimeError("Table II lookback set does not match the result registry")
    for lookback, measured in measured_table_2.items():
        expected = expected_table_2[lookback]
        pairs = (
            ("validation_iou", "validation_iou"),
            ("pointing", "test_pointing"),
            ("iou", "test_iou"),
        )
        for measured_key, expected_key in pairs:
            if abs(measured[measured_key] - expected[expected_key]) > 5e-5:
                raise RuntimeError(
                    f"Table II mismatch for {lookback:.1f}s/{measured_key}: "
                    f"measured={measured[measured_key]:.6f}, "
                    f"expected={expected[expected_key]:.6f}"
                )


def run(cache, raw_root, registry_path):
    import torch

    records, labels = torch.load(cache, weights_only=False)
    label_index = {label: index for index, label in enumerate(labels)}
    partition = split(records, 0)
    raw_records = reflacx.find_records(raw_root)

    def instances(name, reconstruct):
        output = []
        for record in records:
            if partition(record) != name:
                continue
            transcript = None
            if reconstruct:
                transcript = raw_records.get(record["rid"], {}).get(
                    "timestamps_transcription.csv"
                )
                if transcript is None:
                    raise RuntimeError(f"missing raw transcript for split={name}")
            for finding in record["labels"]:
                if not finding["mentions"]:
                    continue
                base_mentions = (
                    reconstruct_base_mentions(transcript, finding["label"])
                    if reconstruct
                    else None
                )
                if reconstruct:
                    assert_cached_lookback(finding["mentions"], base_mentions)
                output.append(
                    (
                        record["fix"],
                        finding["mentions"],
                        finding["ellipses"],
                        label_index[finding["label"]],
                        word_feat(finding.get("mtext", [])),
                        finding["label"],
                        record["subject"],
                        base_mentions,
                    )
                )
        return output

    train = instances("train", False)
    validation = instances("val", True)
    test = instances("test", True)
    print(
        f"COHORT train={len(train)} val={len(validation)} test={len(test)}",
        flush=True,
    )
    validation_targets = [raster(item[2]) for item in validation]
    test_targets = [raster(item[2]) for item in test]

    priors = label_prior((item[2], item[5]) for item in train)
    directions = directional_statistics(train)
    modulation_cache = {}

    def masks(items, directional):
        output = []
        for item in items:
            anatomical_map = priors.get(item[5])
            if anatomical_map is not None and directional:
                key = (item[5], tuple(item[4][DIRECTIONAL_INDICES]))
                anatomical_map = modulation_cache.setdefault(
                    key,
                    apply_directional_terms(anatomical_map, item[4], directions),
                )
            output.append(anatomical_map)
        return output

    validation_plain = masks(validation, False)
    test_plain = masks(test, False)
    validation_directional = masks(validation, True)
    test_directional = masks(test, True)

    def complete_maps(items, _masks):
        return [
            lambda sigma, item=item: complete_scanpath_heat(item[0], sigma)
            for item in items
        ]

    def prior_maps(items, item_masks):
        return [
            lambda sigma, mask=mask: prior_heat(mask, sigma) for mask in item_masks
        ]

    def temporal_maps(items, _masks):
        return [
            lambda sigma, item=item: temporal_window_heat(
                item[0], item[1], sigma
            )
            for item in items
        ]

    def scanpath_maps(items, item_masks):
        return [
            lambda sigma, item=item, mask=mask: scanpath_support_heat(
                item[0], mask, sigma
            )
            for item, mask in zip(items, item_masks)
        ]

    def combined_maps(items, item_masks, lookback):
        return [
            lambda sigma, item=item, mask=mask, lookback=lookback: combined_structured_heat(
                item[0],
                (
                    item[1]
                    if lookback == 1.5
                    else apply_lookback(item[7], lookback)
                ),
                mask,
                sigma,
            )
            for item, mask in zip(items, item_masks)
        ]

    variants = (
        ("complete_scanpath_density", complete_maps, validation_plain, test_plain, TUNE_SIGMAS),
        (
            "anatomical_prior",
            prior_maps,
            validation_plain,
            test_plain,
            (0.0, *TUNE_SIGMAS),
        ),
        ("temporal_1_5s", temporal_maps, validation_plain, test_plain, TUNE_SIGMAS),
        ("prior_x_scanpath", scanpath_maps, validation_plain, test_plain, TUNE_SIGMAS),
        (
            "prior_x_scanpath_directional",
            scanpath_maps,
            validation_directional,
            test_directional,
            TUNE_SIGMAS,
        ),
        (
            "prior_x_scanpath_temporal",
            lambda items, item_masks: combined_maps(items, item_masks, 1.5),
            validation_plain,
            test_plain,
            TUNE_SIGMAS,
        ),
        (
            "combined_structured_1_5s",
            lambda items, item_masks: combined_maps(items, item_masks, 1.5),
            validation_directional,
            test_directional,
            TUNE_SIGMAS,
        ),
        (
            "combined_structured_3_0s",
            lambda items, item_masks: combined_maps(items, item_masks, 3.0),
            validation_directional,
            test_directional,
            TUNE_SIGMAS,
        ),
    )

    table_1 = {}
    for method, factory, validation_masks, test_masks, sigmas in variants:
        validation_maps = factory(validation, validation_masks)
        test_maps = factory(test, test_masks)
        sigma, threshold, validation_iou = _tune(
            validation_maps, validation_targets, sigmas
        )
        scores = _score(test_maps, test_targets, sigma, threshold)
        table_1[method] = scores
        print(
            f"TABLE1 method={method} n={len(test)} "
            f"pg={scores['pointing']:.4f} iou={scores['iou']:.4f} "
            f"sigma={sigma} threshold={threshold:.4f} "
            f"val_iou={validation_iou:.4f}",
            flush=True,
        )

    table_2 = {}
    for lookback in LOOKBACKS:
        validation_maps = combined_maps(
            validation, validation_directional, lookback
        )
        test_maps = combined_maps(test, test_directional, lookback)
        sigma, threshold, validation_iou = _tune(
            validation_maps, validation_targets
        )
        scores = _score(test_maps, test_targets, sigma, threshold)
        table_2[lookback] = {**scores, "validation_iou": validation_iou}
        print(
            f"TABLE2 lookback={lookback:.1f} n={len(test)} "
            f"val_iou={validation_iou:.4f} pg={scores['pointing']:.4f} "
            f"iou={scores['iou']:.4f} sigma={sigma} "
            f"threshold={threshold:.4f}",
            flush=True,
        )

    _assert_registry_match(table_1, table_2, registry_path)
    print("REPRODUCTION_CHECK table_1=pass table_2=pass", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default="align.pt")
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    arguments = parser.parse_args()
    run(arguments.cache, arguments.raw_root, arguments.registry)

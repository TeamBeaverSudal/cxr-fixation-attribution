#!/usr/bin/env python3
"""Inference against the validation-selected extended-window structured reference.

This reruns the prespecified optimizer seeds for the full ten-indicator
selector and the four-direction query control, reconstructs the combined
structured baseline's validation-selected gate from the raw transcripts, and
reports patient-cluster intervals after averaging seed-specific outcomes per
instance. Optimizer seeds are repeated computations, not independent
observations.

Only identifier-free aggregates are printed.
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy.ndimage import zoom
from scipy.stats import wilcoxon

from finding_level_gaze_targets.baselines.anatomy import label_prior, split
from finding_level_gaze_targets.baselines.structured import (
    apply_directional_terms,
    combined_structured_heat,
    directional_statistics,
)
from finding_level_gaze_targets.data import reflacx as gr
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
    tune_thresholds,
    word_feat,
)
from finding_level_gaze_targets.models.selector import blur_norm, predict_raw, train_model
from finding_level_gaze_targets.settings import FUSION, POSITION_ENCODING

POS_MODE = POSITION_ENCODING


THRESHOLDS = tune_thresholds()
DIRECTIONAL_INDICES = np.asarray([0, 1, 3, 4], dtype=int)
LOOKBACKS = (0.5, 1.0, 1.5, 2.0, 3.0)


def _directional_query(word_features):
    output = np.zeros_like(word_features)
    output[DIRECTIONAL_INDICES] = word_features[DIRECTIONAL_INDICES]
    return output


def _iou_pre(upscaled, target, threshold):
    prediction = upscaled >= threshold
    intersection = (prediction & target).sum()
    union = prediction.sum() + target.sum() - intersection
    return intersection / union if union > 0 else np.nan


def _tune(raw_maps, targets, already_smoothed=False):
    best = (TUNE_SIGMAS[0], THRESHOLDS[0], -1.0)
    for sigma in TUNE_SIGMAS:
        upscaled = []
        for raw, target in zip(raw_maps, targets):
            heat = raw(sigma) if callable(raw) else (
                raw if already_smoothed else blur_norm(raw, sigma)
            )
            upscaled.append((zoom(heat, EVAL_RES / heat.shape[0], order=0), target))
        for threshold in THRESHOLDS:
            value = np.nanmean(
                [_iou_pre(heat, target, threshold) for heat, target in upscaled]
            )
            if value > best[2]:
                best = (sigma, threshold, float(value))
    return best


def _score(raw_maps, targets, sigma, threshold, already_smoothed=False):
    heats = []
    for raw in raw_maps:
        heat = raw(sigma) if callable(raw) else (
            raw if already_smoothed else blur_norm(raw, sigma)
        )
        heats.append(heat)
    return {
        "pg": np.asarray(
            [pointing(heat, target) for heat, target in zip(heats, targets)], float
        ),
        "iou": np.asarray(
            [iou(heat, target, threshold) for heat, target in zip(heats, targets)], float
        ),
    }


def _patient_cluster(diff, patients, nboot, seed):
    diff = np.asarray(diff, dtype=float)
    patients = np.asarray(patients, dtype=object)
    valid = np.isfinite(diff)
    diff, patients = diff[valid], patients[valid]
    order = list(dict.fromkeys(patients.tolist()))
    mapping = {patient: index for index, patient in enumerate(order)}
    group = np.asarray([mapping[patient] for patient in patients], dtype=int)
    sums = np.bincount(group, weights=diff, minlength=len(order))
    counts = np.bincount(group, minlength=len(order)).astype(float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(order), size=(nboot, len(order)))
    boot = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    patient_means = sums / counts
    p_value = float(wilcoxon(patient_means).pvalue) if np.any(patient_means) else 1.0
    return float(np.mean(diff)), float(lo), float(hi), p_value, len(order)


def run(cache, raw_root, epochs, seeds, split_seed, model_names, nboot, boot_seed):
    import torch

    records, labels = torch.load(cache, weights_only=False)
    label_index = {label: index for index, label in enumerate(labels)}
    partition = split(records, split_seed)
    raw_records = gr.find_records(raw_root)

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
                    raise RuntimeError("missing raw transcript")
            for item in record["labels"]:
                if not item["mentions"]:
                    continue
                base = (
                    reconstruct_base_mentions(transcript, item["label"])
                    if reconstruct
                    else None
                )
                if reconstruct:
                    assert_cached_lookback(item["mentions"], base)
                output.append(
                    (
                        record["fix"],
                        item["mentions"],
                        item["ellipses"],
                        label_index[item["label"]],
                        word_feat(item.get("mtext", [])),
                        item["label"],
                        record["subject"],
                        base,
                    )
                )
        return output

    train = instances("train", False)
    validation = instances("val", True)
    test = instances("test", True)
    print(
        f"COHORT split={split_seed} seeds={','.join(map(str, seeds))} train={len(train)} "
        f"val={len(validation)} test={len(test)}",
        flush=True,
    )
    validation_targets = [raster(item[2]) for item in validation]
    test_targets = [raster(item[2]) for item in test]

    model_specs = {
        "full": lambda item: item[4],
        "four_indicator": lambda item: _directional_query(item[4]),
    }
    unknown = set(model_names) - set(model_specs)
    if unknown:
        raise ValueError(f"unknown model names: {sorted(unknown)}")
    seed_models = {name: {"pg": [], "iou": []} for name in model_names}
    for seed in seeds:
        for name in model_names:
            transform = model_specs[name]
            training = [(*item[:4], transform(item)) for item in train]
            network = train_model(
                training,
                labels,
                use_position=True,
                epochs=epochs,
                use_text=True,
                fusion=FUSION,
                pos_mode=POS_MODE,
                seed=seed,
            )

            def raw(item, network=network, transform=transform):
                return predict_raw(
                    network,
                    item[0],
                    item[1],
                    item[3],
                    use_position=True,
                    use_text=True,
                    wf=transform(item),
                    pos_mode=POS_MODE,
                )

            validation_maps = [raw(item) for item in validation]
            test_maps = [raw(item) for item in test]
            sigma, threshold, validation_iou = _tune(
                validation_maps, validation_targets
            )
            scores = _score(test_maps, test_targets, sigma, threshold)
            for metric in ("pg", "iou"):
                seed_models[name][metric].append(scores[metric])
            print(
                f"SELECTOR method={name} seed={seed} n={len(test)} "
                f"pg={np.nanmean(scores['pg']):.4f} "
                f"iou={np.nanmean(scores['iou']):.4f} sigma={sigma} "
                f"threshold={threshold:.4f} val_iou={validation_iou:.4f}",
                flush=True,
            )

    models = {}
    for name in seed_models:
        models[name] = {
            metric: np.nanmean(np.stack(seed_models[name][metric]), axis=0)
            for metric in ("pg", "iou")
        }
        pg_means = np.asarray(
            [np.nanmean(values) for values in seed_models[name]["pg"]], dtype=float
        )
        iou_means = np.asarray(
            [np.nanmean(values) for values in seed_models[name]["iou"]], dtype=float
        )
        print(
            f"SELECTOR_MEAN method={name} seeds={len(seeds)} n={len(test)} "
            f"pg={np.mean(pg_means):.4f} pg_sd={np.std(pg_means, ddof=1):.4f} "
            f"pg_range=[{np.min(pg_means):.4f},{np.max(pg_means):.4f}] "
            f"iou={np.mean(iou_means):.4f} iou_sd={np.std(iou_means, ddof=1):.4f} "
            f"iou_range=[{np.min(iou_means):.4f},{np.max(iou_means):.4f}]",
            flush=True,
        )

    prior = label_prior((item[2], item[5]) for item in train)
    directions = directional_statistics(train)
    modulated = {}

    def masks(items):
        output = []
        for item in items:
            mask = prior.get(item[5])
            if mask is not None:
                key = (item[5], tuple(item[4][DIRECTIONAL_INDICES]))
                mask = modulated.setdefault(
                    key, apply_directional_terms(mask, item[4], directions)
                )
            output.append(mask)
        return output

    validation_masks = masks(validation)
    test_masks = masks(test)
    structured_candidates = []
    for lookback in LOOKBACKS:
        candidate_maps = [
            (
                lambda sigma, item=item, mask=mask, lookback=lookback: combined_structured_heat(
                    item[0], apply_lookback(item[7], lookback), mask, sigma
                )
            )
            for item, mask in zip(validation, validation_masks)
        ]
        candidate_sigma, candidate_threshold, candidate_iou = _tune(
            candidate_maps, validation_targets, already_smoothed=True
        )
        structured_candidates.append(
            (candidate_iou, -lookback, lookback, candidate_sigma, candidate_threshold)
        )
        print(
            f"WINDOW_VALIDATION split={split_seed} lookback={lookback:.1f} "
            f"sigma={candidate_sigma} threshold={candidate_threshold:.4f} "
            f"iou={candidate_iou:.4f}",
            flush=True,
        )
    validation_iou, _, selected_lookback, sigma, threshold = max(
        structured_candidates
    )
    test_maps = [
        (
            lambda sigma, item=item, mask=mask: combined_structured_heat(
                item[0], apply_lookback(item[7], selected_lookback), mask, sigma
            )
        )
        for item, mask in zip(test, test_masks)
    ]
    structured = _score(
        test_maps, test_targets, sigma, threshold, already_smoothed=True
    )
    print(
        f"SELECTOR method=combined_structured_validation_selected n={len(test)} "
        f"lookback={selected_lookback:.1f} "
        f"pg={np.nanmean(structured['pg']):.4f} "
        f"iou={np.nanmean(structured['iou']):.4f} sigma={sigma} "
        f"threshold={threshold:.4f} val_iou={validation_iou:.4f}",
        flush=True,
    )

    if split_seed == 0:
        if not (
            selected_lookback == 3.0
            and abs(np.nanmean(structured["pg"]) - 0.7893) <= 5e-5
            and abs(np.nanmean(structured["iou"]) - 0.3549) <= 5e-5
        ):
            raise RuntimeError("primary structured-reference reproduction failed")
        print("REPRODUCTION_CHECK deterministic_structured=status_pass", flush=True)

    patients = [item[6] for item in test]
    comparisons = {}
    if "full" in models:
        comparisons["full_minus_selected_structured"] = models["full"]
    if "four_indicator" in models:
        comparisons["four_indicator_minus_selected_structured"] = models[
            "four_indicator"
        ]
    if {"full", "four_indicator"}.issubset(models):
        comparisons["four_indicator_minus_full"] = {
            metric: models["four_indicator"][metric] - models["full"][metric]
            for metric in ("pg", "iou")
        }
    for comparison, left in comparisons.items():
        for metric_index, metric in enumerate(("pg", "iou")):
            if comparison == "four_indicator_minus_full":
                difference = left[metric]
            else:
                difference = left[metric] - structured[metric]
            observed, lo, hi, p_value, patient_count = _patient_cluster(
                difference, patients, nboot, boot_seed + 10 * metric_index
                + len(comparison)
            )
            print(
                f"INFERENCE estimator=per_instance_seed_mean seeds={len(seeds)} "
                f"comparison={comparison} metric={metric} n={len(test)} "
                f"patients={patient_count} delta={observed:+.4f} "
                f"ci=[{lo:+.4f},{hi:+.4f}] "
                f"patient_wilcoxon_p={p_value:.6g}",
                flush=True,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="align.pt")
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--split-seed", type=int, default=0, dest="split_seed")
    parser.add_argument("--models", default="full,four_indicator")
    parser.add_argument("--boot", type=int, default=10000)
    parser.add_argument("--boot-seed", type=int, default=20260815, dest="boot_seed")
    arguments = parser.parse_args()
    run(
        arguments.cache,
        arguments.raw_root,
        arguments.epochs,
        [int(value) for value in arguments.seeds.split(",")],
        arguments.split_seed,
        [value for value in arguments.models.split(",") if value],
        arguments.boot,
        arguments.boot_seed,
    )

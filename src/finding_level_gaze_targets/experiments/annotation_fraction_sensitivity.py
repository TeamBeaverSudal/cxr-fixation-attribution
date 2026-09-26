#!/usr/bin/env python3
"""Sensitivity to the fraction of ellipse-annotated training patients.

This experiment isolates training-supervision availability. For each sampling
seed, it constructs nested 10%, 25%, and 50% subsets of the mention-resolved
training patients; sampling seed 0 also evaluates the 100% set. The learned
selector and the training-derived combined structured baseline are rebuilt from
the same retained patients. The complete mention-resolved validation and test
cohorts remain fixed so that changing calibration or evaluation composition
does not masquerade as a training-fraction effect.

Only identifier-free aggregates are printed. This is an annotation-fraction
sensitivity analysis, not a measurement of annotation time or a comparison
with an image-only localization model.
"""

from __future__ import annotations

import argparse
import math

import numpy as np
from scipy.ndimage import zoom

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
LOOKBACKS = (0.5, 1.0, 1.5, 2.0, 3.0)


def _iou_pre(upscaled: np.ndarray, target: np.ndarray, threshold: float) -> float:
    prediction = upscaled >= threshold
    intersection = (prediction & target).sum()
    union = prediction.sum() + target.sum() - intersection
    return intersection / union if union > 0 else np.nan


def _tune(raw_maps, targets, sigmas=TUNE_SIGMAS, already_smoothed=False):
    best = (sigmas[0], THRESHOLDS[0], -1.0)
    for sigma in sigmas:
        upscaled = []
        for raw, target in zip(raw_maps, targets):
            heat = raw(sigma) if callable(raw) else (
                raw if already_smoothed else blur_norm(raw, sigma)
            )
            upscaled.append((zoom(heat, EVAL_RES / heat.shape[0], order=0), target))
        for threshold in THRESHOLDS:
            score = np.nanmean(
                [_iou_pre(heat, target, threshold) for heat, target in upscaled]
            )
            if score > best[2]:
                best = (sigma, threshold, float(score))
    return best


def _score(raw_maps, targets, sigma, threshold, already_smoothed=False):
    heats = []
    for raw in raw_maps:
        heat = raw(sigma) if callable(raw) else (
            raw if already_smoothed else blur_norm(raw, sigma)
        )
        heats.append(heat)
    return {
        "iou": np.asarray(
            [iou(heat, target, threshold) for heat, target in zip(heats, targets)],
            dtype=float,
        ),
        "pg": np.asarray(
            [pointing(heat, target) for heat, target in zip(heats, targets)],
            dtype=float,
        ),
    }


def _patient_cluster(diff, patients, nboot, seed):
    diff = np.asarray(diff, dtype=float)
    patients = np.asarray(patients, dtype=object)
    valid = np.isfinite(diff)
    diff, patients = diff[valid], patients[valid]
    patient_order = list(dict.fromkeys(patients.tolist()))
    patient_index = {patient: index for index, patient in enumerate(patient_order)}
    group = np.asarray([patient_index[patient] for patient in patients], dtype=int)
    sums = np.bincount(group, weights=diff, minlength=len(patient_order))
    counts = np.bincount(group, minlength=len(patient_order)).astype(float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(patient_order), size=(nboot, len(patient_order)))
    boot = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(np.mean(diff)), float(lo), float(hi), len(patient_order)


def _nested_patient_subset(train, fraction, subset_seed):
    patients = np.asarray(sorted({item[6] for item in train}), dtype=object)
    rng = np.random.default_rng(20260815 + subset_seed)
    order = rng.permutation(len(patients))
    keep_n = len(patients) if fraction >= 1 else max(1, math.ceil(fraction * len(patients)))
    selected = set(patients[order[:keep_n]].tolist())
    return [item for item in train if item[6] in selected], keep_n, len(patients)


def _combined_structured(training, validation, test, validation_targets, test_targets):
    prior = label_prior((item[2], item[5]) for item in training)
    directions = directional_statistics(training)
    modulated = {}

    def masks(items):
        output = []
        for item in items:
            mask = prior.get(item[5])
            if mask is not None:
                key = (item[5], tuple(item[4][[0, 1, 3, 4]]))
                mask = modulated.setdefault(
                    key, apply_directional_terms(mask, item[4], directions)
                )
            output.append(mask)
        return output

    validation_masks = masks(validation)
    test_masks = masks(test)
    candidates = []
    for lookback in LOOKBACKS:
        validation_maps = [
            (
                lambda sigma, item=item, mask=mask, lookback=lookback:
                combined_structured_heat(
                    item[0], apply_lookback(item[7], lookback), mask, sigma
                )
            )
            for item, mask in zip(validation, validation_masks)
        ]
        sigma, threshold, validation_iou = _tune(
            validation_maps,
            validation_targets,
            sigmas=list(TUNE_SIGMAS),
            already_smoothed=True,
        )
        candidates.append((validation_iou, -lookback, lookback, sigma, threshold))
    validation_iou, _, lookback, sigma, threshold = max(candidates)
    test_maps = [
        (
            lambda sigma, item=item, mask=mask:
            combined_structured_heat(
                item[0], apply_lookback(item[7], lookback), mask, sigma
            )
        )
        for item, mask in zip(test, test_masks)
    ]
    scores = _score(
        test_maps,
        test_targets,
        sigma,
        threshold,
        already_smoothed=True,
    )
    return scores, lookback, sigma, threshold, validation_iou, len(prior)


def _learned(training, validation, test, labels, epochs, model_seed,
             validation_targets, test_targets):
    model_items = [item[:5] for item in training]
    network = train_model(
        model_items,
        labels,
        use_position=True,
        epochs=epochs,
        use_text=True,
        fusion=FUSION,
        pos_mode=POS_MODE,
        seed=model_seed,
    )

    def raw(item):
        return predict_raw(
            network,
            item[0],
            item[1],
            item[3],
            use_position=True,
            use_text=True,
            wf=item[4],
            pos_mode=POS_MODE,
        )

    validation_maps = [raw(item) for item in validation]
    test_maps = [raw(item) for item in test]
    sigma, threshold, validation_iou = _tune(validation_maps, validation_targets)
    scores = _score(test_maps, test_targets, sigma, threshold)
    return scores, sigma, threshold, validation_iou


def run(cache, raw_root, epochs, subset_seed, model_seed, fractions, nboot,
        skip_structured):
    import torch

    records, labels = torch.load(cache, weights_only=False)
    label_index = {label: index for index, label in enumerate(labels)}
    partition = split(records, 0)
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
    validation_targets = [raster(item[2]) for item in validation]
    test_targets = [raster(item[2]) for item in test]
    test_patients = [item[6] for item in test]
    print(
        f"COHORT train={len(train)} val={len(validation)} test={len(test)} "
        f"train_patients={len(set(item[6] for item in train))} "
        f"subset_seed={subset_seed} model_seed={model_seed}",
        flush=True,
    )

    for fraction in fractions:
        selected, kept_patients, total_patients = _nested_patient_subset(
            train, fraction, subset_seed
        )
        present_labels = {item[5] for item in selected}
        ellipse_count = sum(len(item[2]) for item in selected)
        print(
            f"SUBSET fraction={fraction:.2f} subset_seed={subset_seed} "
            f"patients={kept_patients}/{total_patients} instances={len(selected)}/{len(train)} "
            f"ellipses={ellipse_count} labels={len(present_labels)}/{len(labels)} "
            f"missing_labels={len(set(labels) - present_labels)}",
            flush=True,
        )

        structured = None
        if not skip_structured:
            (
                structured,
                lookback,
                s_sigma,
                s_threshold,
                s_val_iou,
                prior_labels,
            ) = _combined_structured(
                selected, validation, test, validation_targets, test_targets
            )
            print(
                f"FRACTION_RESULT method=STRUCTURED fraction={fraction:.2f} "
                f"subset_seed={subset_seed} n={len(test)} pg={np.nanmean(structured['pg']):.4f} "
                f"iou={np.nanmean(structured['iou']):.4f} lookback={lookback:.1f} sigma={s_sigma} "
                f"threshold={s_threshold:.4f} val_iou={s_val_iou:.4f} "
                f"prior_labels={prior_labels}",
                flush=True,
            )

        learned, l_sigma, l_threshold, l_val_iou = _learned(
            selected,
            validation,
            test,
            labels,
            epochs,
            model_seed,
            validation_targets,
            test_targets,
        )
        print(
            f"FRACTION_RESULT method=LEARNED fraction={fraction:.2f} "
            f"subset_seed={subset_seed} n={len(test)} pg={np.nanmean(learned['pg']):.4f} "
            f"iou={np.nanmean(learned['iou']):.4f} sigma={l_sigma} "
            f"threshold={l_threshold:.4f} val_iou={l_val_iou:.4f}",
            flush=True,
        )
        if structured is not None:
            for metric_index, metric in enumerate(("pg", "iou")):
                delta, lo, hi, n_patients = _patient_cluster(
                    learned[metric] - structured[metric],
                    test_patients,
                    nboot,
                    20260815 + subset_seed * 100 + int(round(fraction * 100)) + metric_index,
                )
                print(
                    f"FRACTION_DELTA fraction={fraction:.2f} subset_seed={subset_seed} "
                    f"metric={metric} n={len(test)} patients={n_patients} "
                    f"learned_minus_structured={delta:+.4f} ci=[{lo:+.4f},{hi:+.4f}]",
                    flush=True,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="align.pt")
    parser.add_argument("--raw-root", required=True, dest="raw_root")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--subset-seed", type=int, required=True, dest="subset_seed")
    parser.add_argument("--model-seed", type=int, default=0, dest="model_seed")
    parser.add_argument("--fractions", default="0.10,0.25,0.50")
    parser.add_argument("--boot", type=int, default=5000)
    parser.add_argument("--skip-structured", action="store_true")
    args = parser.parse_args()
    fractions = [float(value) for value in args.fractions.split(",")]
    run(
        args.cache,
        args.raw_root,
        args.epochs,
        args.subset_seed,
        args.model_seed,
        fractions,
        args.boot,
        args.skip_structured,
    )

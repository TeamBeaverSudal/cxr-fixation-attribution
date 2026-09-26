#!/usr/bin/env python3
"""Resolved-cohort controls that perturb scoring features but retain output coordinates.

Position-feature permutation shuffles only the Fourier coordinate features seen
by the scorer. Timing-feature permutation shuffles only the five temporal and
kinematic feature rows. Both render the resulting weights at the original
fixation coordinates, avoiding the output-location change in the legacy joint
position permutation. Only identifier-free aggregates are printed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import zoom

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import EVAL_RES, TUNE_SIGMAS, align_feats, iou, pointing, raster, tune_thresholds, word_feat
from prior_and_swap import split
from selector import _raw_grid, blur_norm, pos_feat, predict_raw, train_model
from masking_control import mask_spatial
from evaluate import FUSION, POS_MODE


def _iou_upscaled(heat, target, threshold):
    prediction = heat >= threshold
    intersection = (prediction & target).sum()
    union = prediction.sum() + target.sum() - intersection
    return intersection / union if union > 0 else np.nan


def _tune(raw_maps, targets):
    best = (TUNE_SIGMAS[0], tune_thresholds()[0], -1.0)
    for sigma in TUNE_SIGMAS:
        heatmaps = [blur_norm(raw, sigma) for raw in raw_maps]
        upscaled = [zoom(heat, EVAL_RES / heat.shape[0], order=0) for heat in heatmaps]
        for threshold in tune_thresholds():
            value = np.nanmean(
                [_iou_upscaled(heat, target, threshold) for heat, target in zip(upscaled, targets)]
            )
            if value > best[2]:
                best = (sigma, threshold, float(value))
    return best


def _score(raw_maps, targets, sigma, threshold):
    heatmaps = [blur_norm(raw, sigma) for raw in raw_maps]
    return (
        float(np.nanmean([pointing(heat, target) for heat, target in zip(heatmaps, targets)])),
        float(np.nanmean([iou(heat, target, threshold) for heat, target in zip(heatmaps, targets)])),
    )


def run(cache, epochs, seed):
    import torch

    records, labels = torch.load(cache, weights_only=False)
    label_index = {label: index for index, label in enumerate(labels)}
    partition = split(records, 0)

    def instances(name):
        return [
            (
                record["fix"], item["mentions"], item["ellipses"],
                label_index[item["label"]], word_feat(item.get("mtext", [])),
            )
            for record in records
            if partition(record) == name
            for item in record["labels"]
            if item["mentions"]
        ]

    train, validation, test = (instances(name) for name in ("train", "val", "test"))
    targets_validation = [raster(item[2]) for item in validation]
    targets_test = [raster(item[2]) for item in test]
    print(f"COHORT seed={seed} train={len(train)} val={len(validation)} test={len(test)}", flush=True)

    network = train_model(
        train, labels, use_position=True, epochs=epochs, use_text=True,
        fusion=FUSION, pos_mode=POS_MODE, seed=seed,
    )

    def ordinary(item, word=None):
        return predict_raw(
            network, item[0], item[1], item[3], use_position=True,
            use_text=True, wf=item[4] if word is None else word, pos_mode=POS_MODE,
        )

    validation_raw = [ordinary(item) for item in validation]
    test_raw = [ordinary(item) for item in test]
    sigma, threshold, validation_iou = _tune(validation_raw, targets_validation)

    rng = np.random.default_rng(20260815 + seed)
    position_raw, timing_raw = [], []
    for item in test:
        fix, mentions, _, label, word = item
        permutation = rng.permutation(len(fix))
        temporal = torch.from_numpy(align_feats(fix, mentions))
        position = pos_feat(fix[:, :2], POS_MODE)
        with torch.no_grad():
            position_weights = network.attn(
                temporal, label, position[permutation], word
            ).numpy()
            timing_weights = network.attn(
                temporal[permutation], label, position, word
            ).numpy()
        position_raw.append(_raw_grid(fix[:, 0], fix[:, 1], position_weights))
        timing_raw.append(_raw_grid(fix[:, 0], fix[:, 1], timing_weights))

    masked_raw = [ordinary(item, mask_spatial(item[4])) for item in test]
    for name, raw_maps in (
        ("learned", test_raw),
        ("position_feature_permutation", position_raw),
        ("timing_feature_permutation", timing_raw),
        ("spatial_indicators_masked", masked_raw),
    ):
        pg, overlap = _score(raw_maps, targets_test, sigma, threshold)
        print(
            f"CONTROL seed={seed} condition={name} n={len(test)} "
            f"pg={pg:.4f} iou={overlap:.4f}",
            flush=True,
        )
    print(
        f"CALIBRATION seed={seed} sigma={sigma} threshold={threshold:.4f} "
        f"validation_iou={validation_iou:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="align.pt")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()
    run(arguments.cache, arguments.epochs, arguments.seed)

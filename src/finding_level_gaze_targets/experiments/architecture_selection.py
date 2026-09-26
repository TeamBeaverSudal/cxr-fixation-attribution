"""Select selector fusion and position encoding using validation IoU only.

The complete two-by-two design compares concatenation with cross-attention and
raw coordinates with Fourier features. Every configuration uses the same
mention-resolved training and validation cohorts and optimizer seed. The test
partition is never evaluated by this selection routine.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
from scipy.ndimage import zoom

from finding_level_gaze_targets.baselines.anatomy import split
from finding_level_gaze_targets.maps.core import (
    EVAL_RES,
    TUNE_SIGMAS,
    raster,
    tune_thresholds,
    word_feat,
)
from finding_level_gaze_targets.models.selector import (
    blur_norm,
    predict_raw,
    train_model,
)
from finding_level_gaze_targets.settings import FUSION, POSITION_ENCODING


VARIANTS = {
    "concat+fourier": {"fusion": "concat", "pos_mode": "fourier"},
    "concat+raw": {"fusion": "concat", "pos_mode": "raw"},
    "crossattn+fourier": {"fusion": "crossattn", "pos_mode": "fourier"},
    "crossattn+raw": {"fusion": "crossattn", "pos_mode": "raw"},
}


def _iou_upscaled(heatmap, target, threshold):
    prediction = heatmap >= threshold
    intersection = (prediction & target).sum()
    union = prediction.sum() + target.sum() - intersection
    return intersection / union if union > 0 else np.nan


def run(cache, epochs, seed=0, split_seed=0, check_frozen=True):
    import torch

    records, labels = torch.load(cache, weights_only=False)
    label_index = {label: index for index, label in enumerate(labels)}
    partition = split(records, split_seed)

    def instances(name):
        return [
            (
                record["fix"],
                finding["mentions"],
                finding["ellipses"],
                label_index[finding["label"]],
                word_feat(finding.get("mtext", [])),
            )
            for record in records
            if partition(record) == name
            for finding in record["labels"]
            if finding["mentions"]
        ]

    training = instances("train")
    validation = instances("val")
    print(
        f"COHORT split={split_seed} seed={seed} train={len(training)} "
        f"val={len(validation)} labels={len(labels)}",
        flush=True,
    )

    thresholds = tune_thresholds()

    def tune(cached):
        best = (TUNE_SIGMAS[0], thresholds[0], -1.0)
        for sigma in TUNE_SIGMAS:
            heatmaps = [
                (
                    zoom(
                        blur_norm(raw, sigma),
                        EVAL_RES / raw.shape[0],
                        order=0,
                    ),
                    target,
                )
                for raw, target in cached
            ]
            for threshold in thresholds:
                score = float(
                    np.nanmean(
                        [
                            _iou_upscaled(heatmap, target, threshold)
                            for heatmap, target in heatmaps
                        ]
                    )
                )
                if score > best[2]:
                    best = (sigma, threshold, score)
        return best

    results = {}
    for name, configuration in VARIANTS.items():
        print(f"TRAIN variant={name} seed={seed}", flush=True)
        network = train_model(
            training,
            labels,
            use_position=True,
            epochs=epochs,
            use_text=True,
            seed=seed,
            **configuration,
        )
        cached = [
            (
                predict_raw(
                    network,
                    item[0],
                    item[1],
                    item[3],
                    use_position=True,
                    use_text=True,
                    wf=item[4],
                    pos_mode=configuration["pos_mode"],
                ),
                raster(item[2]),
            )
            for item in validation
        ]
        sigma, threshold, validation_iou = tune(cached)
        results[name] = validation_iou
        print(
            f"ARCHITECTURE variant={name} val_iou={validation_iou:.4f} "
            f"sigma={sigma} threshold={threshold:.4f}",
            flush=True,
        )

    selected = max(results, key=results.get)
    frozen = f"{FUSION}+{POSITION_ENCODING}"
    print(
        f"ARCHITECTURE_SELECTION selected={selected} frozen={frozen} "
        f"matches={selected == frozen}",
        flush=True,
    )
    if check_frozen and selected != frozen:
        raise RuntimeError(
            f"validation selected {selected}, but settings.py freezes {frozen}"
        )


def self_check():
    """Confirm that all four architecture variants train and validate."""
    import torch

    rng = np.random.default_rng(0)
    records = []
    for index in range(12):
        fixation_count = 12
        center_time = np.sort(rng.uniform(0, 10, fixation_count))
        duration = np.full(fixation_count, 0.2)
        velocity = np.zeros(fixation_count)
        lesion_x, lesion_y = rng.uniform(0.3, 0.7, 2)
        x_coordinate = rng.uniform(0, 1, fixation_count)
        y_coordinate = rng.uniform(0, 1, fixation_count)
        near = rng.random(fixation_count) < 0.4
        x_coordinate[near] = lesion_x + rng.normal(0, 0.02, near.sum())
        y_coordinate[near] = lesion_y + rng.normal(0, 0.02, near.sum())
        fixations = np.stack(
            [x_coordinate, y_coordinate, center_time, duration, velocity], axis=1
        ).astype(np.float32)
        records.append(
            {
                "rid": f"record-{index}",
                "subject": f"subject-{index}",
                "fix": fixations,
                "labels": [
                    {
                        "label": "finding",
                        "ellipses": [
                            (
                                lesion_x - 0.05,
                                lesion_y - 0.05,
                                lesion_x + 0.05,
                                lesion_y + 0.05,
                            )
                        ],
                        "mentions": [(1.0, 2.0, 3.0)],
                        "mtext": ["left lung nodule"],
                    }
                ],
            }
        )

    with tempfile.TemporaryDirectory(prefix="gaze-target-architecture-") as tmp:
        cache = Path(tmp) / "synthetic.pt"
        torch.save((records, ["finding"]), cache)
        run(cache, epochs=1, check_frozen=False)
    print("architecture self-check: pass")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default="align.pt")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--self-check", action="store_true")
    arguments = parser.parse_args()
    if arguments.self_check:
        self_check()
    else:
        run(
            arguments.cache,
            arguments.epochs,
            arguments.seed,
            arguments.split_seed,
        )

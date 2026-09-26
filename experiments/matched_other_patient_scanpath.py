#!/usr/bin/env python3
"""Exact-match other-patient scanpath control for offline attribution.

For each mention-resolved validation or test target, donors are drawn only from
training patients with the same finding and seven-bit spatial-text signature.
The target finding and text query score each donor's completed fixation record,
with temporal features expressed relative to that donor's resolved mention;
the resulting raw maps are averaged. Donor count is selected on validation
pointing, with validation IoU and smaller count as deterministic tie-breakers.
Only identifier-free aggregates are printed.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.ndimage import zoom
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import EVAL_RES, TUNE_SIGMAS, iou, pointing, raster, tune_thresholds, word_feat
from prior_and_swap import split
from selector import blur_norm, predict_raw, train_model
from evaluate import FUSION, POS_MODE


THRESHOLDS = tune_thresholds()
SPATIAL_DIM = 7
DONOR_COUNTS = (1, 2, 4, 8)


def _rank(ranking_seed: int, target_key: str, donor_key: str) -> bytes:
    """Return a stable donor order independent of optimizer initialization."""
    return hashlib.sha256(
        f"{ranking_seed}:{target_key}:{donor_key}".encode()
    ).digest()


def _iou_pre(up: np.ndarray, gt: np.ndarray, threshold: float) -> float:
    pred = up >= threshold
    intersection = (pred & gt).sum()
    union = pred.sum() + gt.sum() - intersection
    return intersection / union if union > 0 else np.nan


def _search(raw_maps, references):
    best = (TUNE_SIGMAS[0], THRESHOLDS[0], -1.0)
    for sigma in TUNE_SIGMAS:
        upscaled = [
            zoom(blur_norm(raw, sigma), EVAL_RES / raw.shape[0], order=0)
            for raw in raw_maps
        ]
        for threshold in THRESHOLDS:
            score = np.nanmean(
                [_iou_pre(heat, target, threshold) for heat, target in zip(upscaled, references)]
            )
            if score > best[2]:
                best = (sigma, threshold, float(score))
    return best


def _score(raw_maps, references, sigma, threshold):
    heatmaps = [blur_norm(raw, sigma) for raw in raw_maps]
    return {
        "iou": np.array(
            [iou(heat, target, threshold) for heat, target in zip(heatmaps, references)]
        ),
        "pg": np.array([pointing(heat, target) for heat, target in zip(heatmaps, references)]),
    }


def _cluster(diff, subjects, nboot, seed):
    ok = np.isfinite(diff)
    diff = np.asarray(diff)[ok]
    subjects = np.asarray(subjects, dtype=object)[ok]
    mapping = {value: index for index, value in enumerate(dict.fromkeys(subjects.tolist()))}
    group = np.array([mapping[value] for value in subjects], dtype=int)
    n_clusters = len(mapping)
    sums = np.bincount(group, weights=diff, minlength=n_clusters)
    counts = np.bincount(group, minlength=n_clusters).astype(float)
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, n_clusters, size=(nboot, n_clusters))
    bootstrap = sums[draw].sum(axis=1) / counts[draw].sum(axis=1)
    lo, hi = np.percentile(bootstrap, [2.5, 97.5])
    patient_means = sums / counts
    p_value = float(wilcoxon(patient_means).pvalue) if np.any(patient_means != 0) else 1.0
    return float(diff.mean()), float(lo), float(hi), p_value, n_clusters


def run(cache, epochs, seeds, split_seed, donor_ranking_seed, nboot, boot_seed):
    import torch

    records, labels = torch.load(cache, weights_only=False)
    label_index = {label: index for index, label in enumerate(labels)}
    partition = split(records, split_seed)

    def instances(name):
        output = []
        for record in records:
            if partition(record) != name:
                continue
            for item in record["labels"]:
                features = word_feat(item.get("mtext", []))
                output.append(
                    (
                        record["fix"],
                        item["mentions"],
                        item["ellipses"],
                        label_index[item["label"]],
                        features,
                        item["label"],
                        record["subject"],
                        f"{record['rid']}::{item['label']}",
                    )
                )
        return output

    train, validation_all, test_all = (instances(name) for name in ("train", "val", "test"))
    validation = [item for item in validation_all if item[1]]
    test = [item for item in test_all if item[1]]
    donors = [item for item in train if item[1]]

    def match_key(item):
        return item[5], tuple(int(value) for value in item[4][:SPATIAL_DIM])

    donor_groups = defaultdict(list)
    for donor in donors:
        donor_groups[match_key(donor)].append(donor)

    validation_references = [raster(item[2]) for item in validation]
    test_references = [raster(item[2]) for item in test]
    print(
        f"split={split_seed} seeds={','.join(map(str, seeds))} "
        f"donor_ranking_seed={donor_ranking_seed} "
        f"train_donors={len(donors)} validation_targets={len(validation)} "
        f"test_targets={len(test)}",
        flush=True,
    )

    per_seed_learned = defaultdict(list)
    per_seed_donor = defaultdict(list)
    common_indices = None
    selected_counts = []
    for seed in seeds:
        print(f"TRAIN seed={seed}", flush=True)
        network = train_model(
            donors,
            labels,
            use_position=True,
            epochs=epochs,
            use_text=True,
            fusion=FUSION,
            pos_mode=POS_MODE,
            seed=seed,
        )

        def target_raw(items):
            return [
                predict_raw(
                    network,
                    item[0],
                    item[1],
                    item[3],
                    use_position=True,
                    use_text=True,
                    wf=item[4],
                    pos_mode=POS_MODE,
                )
                for item in items
            ]

        validation_target_raw = target_raw(validation)
        test_target_raw = target_raw(test)
        learned_operating = _search(validation_target_raw, validation_references)

        def donor_maps(items, maximum):
            eligible_indices = []
            maps = []
            counts = []
            for target_index, target in enumerate(items):
                candidates = [
                    donor
                    for donor in donor_groups.get(match_key(target), [])
                    if donor[6] != target[6]
                ]
                candidates.sort(
                    key=lambda donor: _rank(
                        donor_ranking_seed, target[7], donor[7]
                    )
                )
                selected = candidates[:maximum]
                if not selected:
                    continue
                raw = [
                    predict_raw(
                        network,
                        donor[0],
                        donor[1],
                        target[3],
                        use_position=True,
                        use_text=True,
                        wf=target[4],
                        pos_mode=POS_MODE,
                    )
                    for donor in selected
                ]
                eligible_indices.append(target_index)
                maps.append(np.mean(np.stack(raw), axis=0).astype(np.float32))
                counts.append(len(selected))
            return (
                np.array(eligible_indices, dtype=int),
                maps,
                np.array(counts, dtype=int),
            )

        candidates = []
        for maximum in DONOR_COUNTS:
            indices, maps, counts = donor_maps(validation, maximum)
            references = [validation_references[index] for index in indices]
            operating = _search(maps, references)
            scores = _score(maps, references, operating[0], operating[1])
            candidates.append(
                (
                    float(np.nanmean(scores["pg"])),
                    float(np.nanmean(scores["iou"])),
                    -maximum,
                    maximum,
                    operating,
                )
            )

        _, _, _, selected_maximum, donor_operating = max(candidates)
        indices, maps, counts = donor_maps(test, selected_maximum)
        if common_indices is None:
            common_indices = indices
        elif not np.array_equal(common_indices, indices):
            raise RuntimeError("eligible target cohort changed across seeds")
        references = [test_references[index] for index in indices]
        donor_scores = _score(maps, references, donor_operating[0], donor_operating[1])
        learned_scores = _score(
            [test_target_raw[index] for index in indices],
            references,
            learned_operating[0],
            learned_operating[1],
        )
        selected_counts.append(selected_maximum)
        for metric in ("pg", "iou"):
            per_seed_learned[metric].append(learned_scores[metric])
            per_seed_donor[metric].append(donor_scores[metric])
        print(
            f"SEED_RESULT seed={seed} maximum={selected_maximum} "
            f"eligible={len(indices)}/{len(test)} mean_donors={np.mean(counts):.3f} "
            f"learned_pg={np.nanmean(learned_scores['pg']):.4f} "
            f"donor_pg={np.nanmean(donor_scores['pg']):.4f} "
            f"learned_iou={np.nanmean(learned_scores['iou']):.4f} "
            f"donor_iou={np.nanmean(donor_scores['iou']):.4f}",
            flush=True,
        )

    assert common_indices is not None
    subjects = np.array([test[index][6] for index in common_indices], dtype=object)
    print(
        f"DONORSELECT validation_primary=pointing seeds={len(seeds)} "
        f"maximum_by_seed={','.join(map(str, selected_counts))} "
        f"coverage={len(common_indices) / len(test):.4f} "
        f"eligible={len(common_indices)}/{len(test)}",
        flush=True,
    )
    for metric_index, metric in enumerate(("pg", "iou")):
        learned_seed_means = np.asarray(
            [np.nanmean(values) for values in per_seed_learned[metric]], dtype=float
        )
        donor_seed_means = np.asarray(
            [np.nanmean(values) for values in per_seed_donor[metric]], dtype=float
        )
        learned_scores = np.nanmean(np.stack(per_seed_learned[metric]), axis=0)
        donor_scores = np.nanmean(np.stack(per_seed_donor[metric]), axis=0)
        diff = learned_scores - donor_scores
        observed, lo, hi, p_value, n_patients = _cluster(
            diff,
            subjects,
            nboot,
            boot_seed + metric_index,
        )
        print(
            f"DONORTEST estimator=per_instance_seed_mean seeds={len(seeds)} "
            f"metric={metric} n_inst={len(common_indices)} n_patients={n_patients} "
            f"learned={np.nanmean(learned_scores):.4f} "
            f"learned_seed_sd={np.std(learned_seed_means, ddof=1):.4f} "
            f"donor={np.nanmean(donor_scores):.4f} "
            f"donor_seed_sd={np.std(donor_seed_means, ddof=1):.4f} "
            f"mean_delta={observed:+.4f} "
            f"ci=[{lo:+.4f},{hi:+.4f}] patient_wilcoxon_p={p_value:.6g}",
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="align.pt")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--split-seed", type=int, default=0, dest="split_seed")
    parser.add_argument(
        "--donor-ranking-seed",
        type=int,
        default=20260818,
        dest="donor_ranking_seed",
        help="Fixed seed for donor ordering; independent of optimizer seeds.",
    )
    parser.add_argument("--boot", type=int, default=10000)
    parser.add_argument("--boot-seed", type=int, default=20260812, dest="boot_seed")
    arguments = parser.parse_args()
    run(
        arguments.cache,
        arguments.epochs,
        [int(value) for value in arguments.seeds.split(",")],
        arguments.split_seed,
        arguments.donor_ranking_seed,
        arguments.boot,
        arguments.boot_seed,
    )

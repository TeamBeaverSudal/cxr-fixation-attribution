"""Structured cue components used by the reported comparisons.

The helpers combine training-derived anatomy, target-record scanpath support,
directional mention terms, and a temporal gate. They share the renderer used
by the learned selector and contain no historical experiment drivers.
"""

from __future__ import annotations

import argparse

import numpy as np

from finding_level_gaze_targets.baselines.anatomy import LEFT, LOWER, RIGHT, UPPER
from finding_level_gaze_targets.maps.core import (
    EVAL_RES,
    HEAT_RES,
    iou,
    raster,
    splat,
    temporal_window_heat,
)
from finding_level_gaze_targets.settings import FUSION, POSITION_ENCODING


_AXIS = (np.arange(EVAL_RES) + 0.5) / EVAL_RES
_X_GRID = np.broadcast_to(_AXIS, (EVAL_RES, EVAL_RES))
_Y_GRID = _X_GRID.T


def complete_scanpath_heat(fixations, sigma):
    """Render duration-weighted density from every recorded fixation."""
    weights = np.asarray(fixations[:, 3], dtype=np.float32)
    if weights.sum() <= 0:
        weights = np.ones(len(fixations), dtype=np.float32)
    return splat(fixations[:, 0], fixations[:, 1], weights, sigma)


def scanpath_support_heat(fixations, anatomical_map, sigma):
    """Sample an anatomical map at fixation locations and render the weights."""
    if anatomical_map is None:
        weights = np.zeros(len(fixations), dtype=np.float32)
    else:
        row = np.clip(
            (fixations[:, 1] * anatomical_map.shape[0]).astype(int),
            0,
            anatomical_map.shape[0] - 1,
        )
        column = np.clip(
            (fixations[:, 0] * anatomical_map.shape[1]).astype(int),
            0,
            anatomical_map.shape[1] - 1,
        )
        weights = anatomical_map[row, column]
    if weights.sum() <= 0:
        weights = np.ones(len(fixations), dtype=np.float32)
    return splat(fixations[:, 0], fixations[:, 1], weights, sigma)


def temporal_gate_weights(fixations, mentions, anatomical_map=None):
    """Return duration weights inside the mention window, optionally gated by anatomy."""
    fixation_start = fixations[:, 2] - fixations[:, 3] / 2
    fixation_end = fixations[:, 2] + fixations[:, 3] / 2
    selected = np.zeros(len(fixations), dtype=bool)
    for gate_start, _sentence_start, gate_end, *_ in mentions:
        selected |= (fixation_start < gate_end) & (fixation_end > gate_start)
    if not selected.any():
        selected[:] = True

    weights = np.zeros(len(fixations), dtype=np.float32)
    weights[selected] = fixations[selected, 3]
    if anatomical_map is not None:
        row = np.clip(
            (fixations[:, 1] * anatomical_map.shape[0]).astype(int),
            0,
            anatomical_map.shape[0] - 1,
        )
        column = np.clip(
            (fixations[:, 0] * anatomical_map.shape[1]).astype(int),
            0,
            anatomical_map.shape[1] - 1,
        )
        anatomical_weights = anatomical_map[row, column]
        if (weights * anatomical_weights).sum() > 0:
            weights *= anatomical_weights
    return weights


def combined_structured_heat(fixations, mentions, anatomical_map, sigma):
    """Render the anatomy × scanpath × temporal-gate structured selector."""
    weights = temporal_gate_weights(fixations, mentions, anatomical_map)
    nonzero = weights > 0
    return splat(
        fixations[nonzero, 0],
        fixations[nonzero, 1],
        weights[nonzero],
        sigma,
    )


def directional_statistics(training_instances):
    """Estimate half-plane orientation for left/right and upper/lower terms."""
    output = []
    for (first, second), grid, coordinate in (
        ((LEFT, RIGHT), _X_GRID, 0),
        ((UPPER, LOWER), _Y_GRID, 1),
    ):
        centers = [
            np.mean(
                [
                    np.mean(
                        [
                            (ellipse[coordinate] + ellipse[coordinate + 2]) / 2
                            for ellipse in item[2]
                        ]
                    )
                    for item in training_instances
                    if item[4][active] == 1
                    and item[4][inactive] == 0
                    and item[2]
                ]
                or [np.nan]
            )
            for active, inactive in ((first, second), (second, first))
        ]
        if np.isfinite(centers).all():
            output.append(
                ((first, second), grid, centers[0] < centers[1], *centers)
            )
    return output


def apply_directional_terms(anatomical_map, word_features, statistics):
    """Restrict anatomy to the named half-plane for unopposed directional terms."""
    keep = np.ones(anatomical_map.shape, dtype=bool)
    for (first, second), grid, first_is_low, _first_center, _second_center in statistics:
        if word_features[first] == 1 and word_features[second] == 0:
            keep &= (grid < 0.5) if first_is_low else (grid > 0.5)
        elif word_features[second] == 1 and word_features[first] == 0:
            keep &= (grid > 0.5) if first_is_low else (grid < 0.5)
    restricted = anatomical_map * keep
    return restricted if restricted.max() > 0 else anatomical_map


def self_check():
    """Exercise coordinate orientation, gating, and the reported learned scorer."""
    import torch

    from finding_level_gaze_targets.maps.core import WORD_DIM, align_feats
    from finding_level_gaze_targets.models.selector import make_net, pos_feat

    anatomical_map = np.zeros((EVAL_RES, EVAL_RES), dtype=np.float32)
    anatomical_map[: EVAL_RES // 2, : EVAL_RES // 2] = 1
    fixations = np.array(
        [
            [0.2, 0.2, 0.0, 1.0, 0.0],
            [0.2, 0.8, 1.0, 1.0, 0.0],
            [0.8, 0.2, 2.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    heat = scanpath_support_heat(fixations, anatomical_map, 1.0)
    peak = np.unravel_index(np.argmax(heat), heat.shape)
    assert peak[0] < HEAT_RES / 2 and peak[1] < HEAT_RES / 2

    rng = np.random.default_rng(0)
    synthetic = np.zeros((40, 5), dtype=np.float32)
    synthetic[:20, :2] = rng.uniform(0.05, 0.45, (20, 2))
    synthetic[20:, :2] = rng.uniform(0.55, 0.95, (20, 2))
    synthetic[:, 3] = 1.0
    target = raster([(0.05, 0.05, 0.45, 0.45)])
    weighted_iou = iou(
        scanpath_support_heat(synthetic, anatomical_map, 1.5), target, 0.3
    )
    unweighted_iou = iou(complete_scanpath_heat(synthetic, 1.5), target, 0.3)
    assert weighted_iou > unweighted_iou

    mentions = [(5.0, 5.0, 7.0)]
    uniform = np.ones((EVAL_RES, EVAL_RES), dtype=np.float32)
    assert np.allclose(
        combined_structured_heat(synthetic, mentions, uniform, 1.5),
        temporal_window_heat(synthetic, mentions, 1.5),
        atol=1e-6,
    )

    gated_fixations = np.array(
        [
            [0.20, 0.50, 5.5, 0.3, 0.1],
            [0.30, 0.50, 6.0, 0.3, 0.1],
            [0.70, 0.50, 6.2, 0.3, 0.1],
            [0.80, 0.50, 6.5, 0.3, 0.1],
            [0.95, 0.50, 1.0, 0.3, 0.1],
            [0.97, 0.50, 15.0, 0.3, 0.1],
        ],
        dtype=np.float32,
    )
    empty = np.zeros((EVAL_RES, EVAL_RES), dtype=np.float32)
    for uninformative in (uniform, empty):
        assert np.allclose(
            combined_structured_heat(
                gated_fixations, mentions, uninformative, 2.0
            ),
            temporal_window_heat(gated_fixations, mentions, 2.0),
            atol=1e-6,
        )

    def horizontal_center(heatmap):
        column_mass = heatmap.sum(axis=0)
        return float(
            (column_mass * np.arange(len(column_mass))).sum()
            / column_mass.sum()
            / len(column_mass)
        )

    left_half = np.zeros((EVAL_RES, EVAL_RES), dtype=np.float32)
    left_half[:, : EVAL_RES // 2] = 1.0
    assert horizontal_center(
        combined_structured_heat(gated_fixations, mentions, left_half, 1.0)
    ) < horizontal_center(
        temporal_window_heat(gated_fixations, mentions, 1.0)
    ) - 0.15

    feature_rows = np.concatenate(
        [
            rng.random((6, 2)),
            np.sort(rng.uniform(0, 10, (6, 1)), axis=0),
            rng.random((6, 2)),
        ],
        axis=1,
    ).astype(np.float32)
    word_features = np.zeros(WORD_DIM, dtype=np.float32)
    word_features[LEFT] = 1.0
    residuals = {}
    for fusion in (FUSION, "concat"):
        torch.manual_seed(0)
        network = make_net(
            ["finding"],
            use_position=True,
            use_text=True,
            fusion=fusion,
            pos_mode=POSITION_ENCODING,
        )

        def log_attention(rows):
            with torch.no_grad():
                return np.log(
                    network.attn(
                        torch.from_numpy(align_feats(rows, [(1.0, 2.0, 2.5)])),
                        0,
                        pos_feat(rows[:, :2], POSITION_ENCODING),
                        word_features,
                    ).numpy()
                )

        swapped = feature_rows.copy()
        swapped[[0, 1], :2] = feature_rows[[1, 0], :2]
        before, after = log_attention(feature_rows), log_attention(swapped)
        residuals[fusion] = abs(
            (after[0] + after[1] - before[0] - before[1])
            - 2 * (after[2] - before[2])
        )
    assert residuals[FUSION] < 1e-4
    assert residuals["concat"] > 1e-3
    print(
        "self-check ok: "
        f"weighted {weighted_iou:.3f} > unweighted {unweighted_iou:.3f}; "
        f"{FUSION} additivity residual {residuals[FUSION]:.2e}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    arguments = parser.parse_args()
    if not arguments.self_check:
        parser.error("this module exposes helpers; pass --self-check to validate them")
    self_check()

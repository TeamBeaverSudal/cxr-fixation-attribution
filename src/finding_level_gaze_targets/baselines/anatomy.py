"""Training-derived anatomical priors and patient-level partitions."""

import numpy as np
from scipy.ndimage import gaussian_filter

from finding_level_gaze_targets.maps.core import EVAL_RES, HEAT_RES, raster


# Indices of directional indicators in maps.core.WORD_TERMS.
LEFT, RIGHT, UPPER, LOWER = 0, 1, 3, 4


def split(records, split_seed=0):
    """Return the deterministic patient-disjoint split used by the study."""
    subjects = np.array(sorted({record["subject"] for record in records}))
    rng = np.random.default_rng(split_seed)
    rng.shuffle(subjects)
    n_subjects = len(subjects)
    validation = set(subjects[: int(n_subjects * 0.15)])
    test = set(subjects[int(n_subjects * 0.15) : int(n_subjects * 0.45)])

    def partition(record):
        if record["subject"] in validation:
            return "val"
        if record["subject"] in test:
            return "test"
        return "train"

    return partition


def label_prior(items):
    """Mean rasterized ellipse mask per finding, estimated from training items."""
    per_label = {}
    for ellipses, label in items:
        per_label.setdefault(label, []).append(raster(ellipses))
    return {
        label: np.mean(masks, axis=0).astype(np.float32)
        for label, masks in per_label.items()
    }


def prior_heat(prior, sigma):
    """Blur and max-normalize an anatomical prior on the common output grid."""
    if prior is None:
        return np.zeros((EVAL_RES, EVAL_RES), np.float32)
    heatmap = (
        gaussian_filter(prior, sigma * EVAL_RES / HEAT_RES)
        if sigma > 0
        else prior
    )
    maximum = heatmap.max()
    return heatmap / maximum if maximum > 0 else heatmap

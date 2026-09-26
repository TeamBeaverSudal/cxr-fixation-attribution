"""Cache construction, fixation features, rendering, and evaluation metrics."""

import argparse
import re
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, zoom

from finding_level_gaze_targets.data import reflacx
from finding_level_gaze_targets.linking.rules import (
    EVAL_RES,
    HEAT_RES,
    KEYWORDS,
    LOOKBACK,
    NEG,
    _col,
    assert_covers_phase3,
    last_mention_end,
    sentences_with_words,
)


TUNE_SIGMAS = [0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
WORD_TERMS = [
    r"\bleft\b",
    r"\bright\b",
    r"bilateral",
    r"upper|apic|apex",
    r"lower|bas(e|al|ilar)",
    r"middle|mid ",
    r"retrocardiac",
    r"small|mild|subtle|question|possibl|trace|minimal",
    r"large|severe|extensive|marked|significant",
    r"diffuse|patchy",
]
WORD_DIM = len(WORD_TERMS)


def tune_thresholds():
    """Threshold grid shared by every structured and learned method."""
    return np.concatenate(
        [np.linspace(0.002, 0.05, 25), np.linspace(0.06, 0.95, 30)]
    )


def raster(ellipses, res=EVAL_RES):
    """Rasterize normalized ellipse annotations onto the evaluation grid."""
    yy, xx = np.mgrid[0:res, 0:res]
    mask = np.zeros((res, res), bool)
    for x_min, y_min, x_max, y_max in ellipses:
        center_x = (x_min + x_max) / 2 * res
        center_y = (y_min + y_max) / 2 * res
        radius_x = max((x_max - x_min) / 2 * res, 0.5)
        radius_y = max((y_max - y_min) / 2 * res, 0.5)
        mask |= (
            ((xx - center_x) / radius_x) ** 2
            + ((yy - center_y) / radius_y) ** 2
            <= 1
        )
    return mask


def _normalized_ellipses(rows, height, width):
    return [
        (
            row["xmin"] / width,
            row["ymin"] / height,
            row["xmax"] / width,
            row["ymax"] / height,
        )
        for _, row in rows.iterrows()
    ]


def extract(root, limit):
    """Build the study cache from a credentialed local REFLACX copy."""
    import pandas as pd

    metadata_files = sorted(Path(root).rglob("metadata_phase_3.csv"))
    metadata = pd.read_csv(metadata_files[0]) if metadata_files else None
    if metadata is not None and "eye_tracking_data_discarded" in metadata.columns:
        metadata = metadata[~metadata["eye_tracking_data_discarded"].astype(bool)]

    subjects, image_sizes, eligible = {}, {}, None
    if metadata is not None:
        subjects = {
            row[_col(metadata, "id")]: str(row[_col(metadata, "subject_id")])
            for _, row in metadata.iterrows()
        }
        image_sizes = {
            row[_col(metadata, "id")]: (
                float(row[_col(metadata, "image_size_y")]),
                float(row[_col(metadata, "image_size_x")]),
            )
            for _, row in metadata.iterrows()
        }
        eligible = set(subjects)

    required = (
        "fixations.csv",
        "anomaly_location_ellipses.csv",
        "timestamps_transcription.csv",
    )
    source_records = sorted(
        (record_id, files)
        for record_id, files in reflacx.find_records(root).items()
        if all(name in files for name in required)
    )
    records, observed_labels = [], set()
    for record_id, files in source_records:
        if len(records) >= limit:
            break
        if eligible is not None and record_id not in eligible:
            continue
        try:
            fixations = pd.read_csv(files["fixations.csv"])
            if len(fixations) < 8:
                continue
            height, width = image_sizes.get(record_id, (None, None))
            if height is None:
                height = float(
                    fixations[_col(fixations, "ymax_shown_from_image")].max()
                )
                width = float(
                    fixations[_col(fixations, "xmax_shown_from_image")].max()
                )
            x_coordinate = fixations[
                _col(fixations, "x_position", "average_x_position")
            ].to_numpy(float) / width
            y_coordinate = fixations[
                _col(fixations, "y_position", "average_y_position")
            ].to_numpy(float) / height
            start = reflacx._seconds(
                fixations[_col(fixations, "timestamp_start_fixation")].to_numpy(float)
            )
            end = reflacx._seconds(
                fixations[_col(fixations, "timestamp_end_fixation")].to_numpy(float)
            )
            center = (start + end) / 2
            duration = end - start
            velocity = np.zeros(len(fixations))
            if len(fixations) > 1:
                distance = np.hypot(np.diff(x_coordinate), np.diff(y_coordinate))
                interval = np.clip(np.diff(center), 1e-3, None)
                velocity[1:] = distance / interval
            fixation_features = np.stack(
                [x_coordinate, y_coordinate, center, duration, velocity], axis=1
            ).astype(np.float32)

            ellipse_table = pd.read_csv(files["anomaly_location_ellipses.csv"])
            labels = [
                column
                for column in ellipse_table.columns
                if column not in {"xmin", "ymin", "xmax", "ymax", "certainty"}
                and ellipse_table[column]
                .dropna()
                .isin([True, False, 0, 1, 0.0, 1.0])
                .all()
            ]
            observed_labels.update(labels)
            sentences = sentences_with_words(
                pd.read_csv(files["timestamps_transcription.csv"])
            )
            sentence_times = [(text, first, last) for text, first, last, _ in sentences]
            findings = []
            for label in labels:
                rows = ellipse_table[ellipse_table[label].astype(bool)]
                if not len(rows):
                    continue
                mentions, mention_text = [], []
                pattern = KEYWORDS.get(label)
                if pattern:
                    matcher = re.compile(pattern, re.I)
                    for index, (text, sentence_start, sentence_end, words) in enumerate(
                        sentences
                    ):
                        if matcher.search(text) and not NEG.search(text):
                            previous_start = (
                                sentence_times[index - 1][1]
                                if index
                                else sentence_start
                            )
                            mention_end = last_mention_end(words, matcher)
                            mentions.append(
                                (
                                    max(sentence_start - LOOKBACK, previous_start),
                                    sentence_start,
                                    sentence_end,
                                    sentence_end if mention_end is None else mention_end,
                                )
                            )
                            mention_text.append(text)
                findings.append(
                    {
                        "label": label,
                        "ellipses": _normalized_ellipses(rows, height, width),
                        "mentions": mentions,
                        "mtext": mention_text,
                    }
                )
            if findings:
                records.append(
                    {
                        "rid": record_id,
                        "subject": subjects.get(record_id, record_id),
                        "fix": fixation_features,
                        "labels": findings,
                    }
                )
        except Exception as error:
            print(f"skip {record_id}: {type(error).__name__}: {error}")

    assert_covers_phase3(observed_labels)
    return records, sorted(observed_labels)


def splat(x_coordinates, y_coordinates, weights, sigma=1.5):
    """Render weighted normalized coordinates on the shared heatmap grid."""
    heatmap = np.zeros((HEAT_RES, HEAT_RES), np.float32)
    x_indices = np.clip((x_coordinates * HEAT_RES).astype(int), 0, HEAT_RES - 1)
    y_indices = np.clip((y_coordinates * HEAT_RES).astype(int), 0, HEAT_RES - 1)
    np.add.at(heatmap, (y_indices, x_indices), weights)
    heatmap = gaussian_filter(heatmap, sigma)
    maximum = heatmap.max()
    return heatmap / maximum if maximum > 0 else heatmap


def inside_ellipses(fixations, ellipses):
    """Identify fixations located inside at least one target ellipse."""
    x_coordinate, y_coordinate = fixations[:, 0], fixations[:, 1]
    inside = np.zeros(len(fixations), bool)
    for x_min, y_min, x_max, y_max in ellipses:
        center_x, center_y = (x_min + x_max) / 2, (y_min + y_max) / 2
        radius_x = max((x_max - x_min) / 2, 1e-3)
        radius_y = max((y_max - y_min) / 2, 1e-3)
        inside |= (
            ((x_coordinate - center_x) / radius_x) ** 2
            + ((y_coordinate - center_y) / radius_y) ** 2
            <= 1
        )
    return inside


def iou(heatmap, target, threshold):
    upscaled = zoom(heatmap, EVAL_RES / heatmap.shape[0], order=0)
    prediction = upscaled >= threshold
    intersection = (prediction & target).sum()
    union = prediction.sum() + target.sum() - intersection
    return intersection / union if union > 0 else np.nan


def pointing(heatmap, target):
    upscaled = zoom(heatmap, EVAL_RES / heatmap.shape[0], order=0)
    if not target.any():
        return np.nan
    return float(target[np.unravel_index(np.argmax(upscaled), upscaled.shape)])


def b1_at(fixations, mentions, sigma):
    """Render the fixed-window temporal baseline at a specified bandwidth."""
    start = fixations[:, 2] - fixations[:, 3] / 2
    end = fixations[:, 2] + fixations[:, 3] / 2
    if mentions:
        selected = np.zeros(len(fixations), bool)
        for gate_start, _sentence_start, gate_end, *_ in mentions:
            selected |= (start < gate_end) & (end > gate_start)
        if selected.any():
            return splat(
                fixations[selected, 0],
                fixations[selected, 1],
                fixations[selected, 3],
                sigma,
            )
    return splat(fixations[:, 0], fixations[:, 1], fixations[:, 3], sigma)


def b1_heat(fixations, mentions):
    return b1_at(fixations, mentions, sigma=1.5)


def word_feat(mention_text):
    features = np.zeros(WORD_DIM, np.float32)
    if mention_text:
        text = " ".join(mention_text).lower()
        for index, term in enumerate(WORD_TERMS):
            if re.search(term, text):
                features[index] = 1.0
    return features


def temporal_dim():
    return 5


def align_feats(fixations, mentions):
    """Compute temporal and kinematic features for each fixation."""
    fixation_times = fixations[:, 2]
    if mentions:
        mention_times = np.array([mention[1] for mention in mentions])
        offsets = fixation_times[:, None] - mention_times[None, :]
        nearest = np.abs(offsets).argmin(axis=1)
        signed_offset = offsets[np.arange(len(fixation_times)), nearest]
    else:
        signed_offset = np.zeros(len(fixation_times))
    return np.stack(
        [
            signed_offset,
            np.abs(signed_offset),
            (signed_offset < 0).astype(float),
            fixations[:, 3],
            fixations[:, 4],
        ],
        axis=1,
    ).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", help="credentialed REFLACX root")
    parser.add_argument("--cache", default="align.pt")
    parser.add_argument("--limit", type=int, default=3000)
    arguments = parser.parse_args()

    import torch

    records, labels = extract(arguments.root, arguments.limit)
    torch.save((records, labels), arguments.cache)
    print(f"extracted {len(records)} Phase-3 records -> {arguments.cache}")


if __name__ == "__main__":
    main()

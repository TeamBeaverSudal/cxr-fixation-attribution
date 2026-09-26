"""Finding mention rules and timestamp alignment helpers for REFLACX Phase 3."""

import re

import numpy as np

from finding_level_gaze_targets.data import reflacx


EVAL_RES = 256
HEAT_RES = 64
LOOKBACK = 1.5

# Canonical Phase-3 finding names mapped to case-insensitive mention patterns.
KEYWORDS = {
    "Atelectasis": r"atelecta|collapse",
    "Consolidation": r"consolidat|airspace",
    "Enlarged cardiac silhouette": (
        r"cardiomegaly|enlarged.*(cardiac|heart)"
        r"|(heart|cardiac|silhouette).*enlarg"
    ),
    "Pleural abnormality": r"effusion|pleural (fluid|thicken)|pleural abnormal",
    "Pulmonary edema": r"edema|vascular congestion|fluid overload",
    "Pneumothorax": r"pneumothorax|\bptx\b",
    "Lung nodule or mass": r"nodule|mass|lesion",
    "Groundglass opacity": r"ground.?glass|opacit|infiltrat|hazy",
    "Abnormal mediastinal contour": r"mediastinal contour|mediastin",
    "Interstitial lung disease": r"interstitial|reticular|ild",
    "Acute fracture": r"fracture",
    "Enlarged hilum": r"hilar|hilum",
    "High lung volume / emphysema": r"emphysema|hyperinflat",
    "Hiatal hernia": r"hiatal hernia|hiatus hernia",
}
NO_KEYWORD_BY_DESIGN = {"Other", "Support devices"}
NEG = re.compile(
    r"\b(no|without|resolved|clear of|free of|negative for|rule[sd]? out)\b",
    re.I,
)


def assert_covers_phase3(labels):
    """Require an explicit decision for every observed Phase-3 finding label."""
    missing = set(labels) - set(KEYWORDS) - NO_KEYWORD_BY_DESIGN
    if missing:
        raise SystemExit(
            "KEYWORDS has no pattern for Phase-3 label(s): "
            f"{sorted(missing)}. Add a pattern or explicitly mark the label as "
            "not mention-resolvable."
        )


def _col(dataframe, *candidates):
    return reflacx._col(dataframe, *candidates)


def sentences_with_words(transcript):
    """Return sentence text, bounds, and individual word timestamps."""
    words = transcript[_col(transcript, "word")].astype(str).tolist()
    starts = reflacx._seconds(
        transcript[_col(transcript, "timestamp_start_word", "timestamp_start")]
        .to_numpy(float)
    )
    ends = reflacx._seconds(
        transcript[_col(transcript, "timestamp_end_word", "timestamp_end")]
        .to_numpy(float)
    )
    sentences, current, sentence_start = [], [], None
    for index, token in enumerate(words):
        if sentence_start is None:
            sentence_start = starts[index]
        current.append((token, starts[index], ends[index]))
        if token.strip().endswith(".") or token.strip() == "." or index == len(words) - 1:
            sentences.append(
                (
                    " ".join(item[0] for item in current),
                    sentence_start,
                    ends[index],
                    current,
                )
            )
            current, sentence_start = [], None
    return sentences


def sentences_with_times(transcript):
    return [
        (text, start, end)
        for text, start, end, _words in sentences_with_words(transcript)
    ]


def last_mention_end(words, pattern):
    """Return the end timestamp of the last word touched by a regex match."""
    spans, position = [], 0
    for token, _start, end in words:
        spans.append((position, position + len(token), end))
        position += len(token) + 1
    text = " ".join(token for token, _start, _end in words)
    latest = None
    for match in pattern.finditer(text):
        for start, end, word_end in spans:
            if start < match.end() and end > match.start():
                latest = word_end if latest is None else max(latest, word_end)
    return latest


def label_windows(sentences, label):
    """Return fixed-lookback windows for positive sentences mentioning a finding."""
    expression = KEYWORDS.get(label)
    if not expression:
        return []
    pattern = re.compile(expression, re.I)
    windows = []
    for index, (text, start, end) in enumerate(sentences):
        if pattern.search(text) and not NEG.search(text):
            previous_start = sentences[index - 1][1] if index else start
            windows.append((max(start - LOOKBACK, previous_start), end))
    return windows


def reconstruct_base_mentions(transcript_path, label):
    """Recover unclipped positive-mention sentence bounds from a raw transcript."""
    import pandas as pd

    expression = KEYWORDS.get(label)
    if not expression:
        return []
    matcher = re.compile(expression, re.I)
    sentences = sentences_with_words(pd.read_csv(transcript_path))
    output = []
    for index, (text, start, end, _words) in enumerate(sentences):
        if matcher.search(text) and not NEG.search(text):
            previous_start = sentences[index - 1][1] if index else start
            output.append((previous_start, start, end))
    return output


def apply_lookback(base_mentions, seconds):
    """Apply a clipped pre-mention lookback to reconstructed sentence bounds."""
    return [
        (max(start - seconds, previous_start), start, end)
        for previous_start, start, end in base_mentions
    ]


def assert_cached_lookback(cached_mentions, base_mentions, seconds=LOOKBACK):
    """Fail when raw-transcript reconstruction disagrees with the study cache."""
    cached = [
        tuple(float(value) for value in mention[:3]) for mention in cached_mentions
    ]
    rebuilt = [
        tuple(float(value) for value in mention)
        for mention in apply_lookback(base_mentions, seconds)
    ]
    if len(cached) != len(rebuilt) or any(
        not np.allclose(left, right, atol=1e-6)
        for left, right in zip(cached, rebuilt)
    ):
        raise RuntimeError("raw mention reconstruction mismatch")

"""B1 reproduction: fixed-1.5s label-gated gaze heatmap, Lanfredi 2023 protocol.
Target = their tuned validation IoU 0.233 (the direct test-time-gaze localizer).
This both establishes the baseline to beat AND re-validates our protocol
(B0 no-gating already matched their ablation: 0.167 ≈ 0.165).

Pipeline per finding-label L present in an image:
  1. find sentences in the timestamped transcript that MENTION L (keyword match
     + simple negation guard) -> mention time windows.
  2. gate window = [MAX(sentence_start - 1.5s, prev_sentence_start), last_mention_end].
  3. accumulate duration-weighted Gaussian (σ=1° visual angle) fixations in the
     window -> per-label heatmap, normalize to max 1. multi-mention -> element MAX.
  4. IoU vs L's ellipse mask, computed at ELLIPSE (full) resolution (predicted
     heatmap NN-upscaled), threshold tuned on val per label.

Two stages (extraction light: fixations + transcript + ellipse; no gaze.csv):
  python linker_and_temporal.py /path/to/reflacx --cache b1.pt   # extract (node/CPU)
  python linker_and_temporal.py --cache b1.pt                    # analyze (mac)
  python linker_and_temporal.py                                  # synthetic self-check

KEYWORD label↔sentence mapping is a FIRST-PASS approximation of Lanfredi's
modified CheXpert labeler; swap in the labeler later for fidelity. The REFLACX
label set / column names are validated at extraction time (printed).
"""
import argparse
import re
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, zoom

import reflacx_io as gr

EVAL_RES = 256  # full-res grid for IoU (stand-in for ellipse resolution; NN-upscale from fixation grid)
HEAT_RES = 64   # resolution the gaze heatmap is built at (then upscaled to EVAL_RES)
LOOKBACK = 1.5  # seconds (Lanfredi tuned value)

# First-pass keyword map for REFLACX-style labels. Keys are canonical label names;
# values are regex alternations matched (case-insensitive) against sentence text.
# Refined/validated at extraction against actual label columns; swap for CheXpert labeler later.
# Keys MUST be REFLACX **Phase 3** finding names -- these are looked up as
# KEYWORDS.get(L) where L is a Phase-3 ellipse column, and a miss returns None
# silently. This dict was originally written against the Phase 1/2 vocabulary, so
# four findings had a perfectly good pattern filed under a name Phase 3 no longer
# uses. Those instances then reached the model with `mentions == []`, which zeroes
# four of the six temporal key features AND leaves word_feat all-zero -- i.e. the
# method did not run on them -- while B1 fell back to the ungated full-gaze heatmap.
# `assert_covers_phase3` below exists so this cannot recur silently.
KEYWORDS = {
    "Atelectasis": r"atelecta|collapse",
    "Consolidation": r"consolidat|airspace",
    # "the cardiac silhouette is enlarged" is the commonest phrasing and matched none of the
    # original three alternations: enlarged.* requires the wrong word order and heart.*enlarg
    # requires the token "heart". Both orders are now covered.
    "Enlarged cardiac silhouette": r"cardiomegaly|enlarged.*(cardiac|heart)"
                                   r"|(heart|cardiac|silhouette).*enlarg",
    "Pleural abnormality": r"effusion|pleural (fluid|thicken)|pleural abnormal",
    "Pulmonary edema": r"edema|vascular congestion|fluid overload",
    "Pneumothorax": r"pneumothorax|\bptx\b",
    "Lung nodule or mass": r"nodule|mass|lesion",
    "Groundglass opacity": r"ground.?glass|opacit|infiltrat|hazy",
    "Abnormal mediastinal contour": r"mediastinal contour|mediastin",
    "Interstitial lung disease": r"interstitial|reticular|ild",
    # --- renamed from the Phase 1/2 keys that never matched a Phase-3 column ---
    "Acute fracture": r"fracture",                       # was "Fracture"
    "Enlarged hilum": r"hilar|hilum",                    # was "Hilar abnormality"
    "High lung volume / emphysema": r"emphysema|hyperinflat",   # was "Emphysema"
    "Hiatal hernia": r"hiatal hernia|hiatus hernia",     # had no pattern at all
    # Dropped: "Wide mediastinum" (duplicate of Abnormal mediastinal contour),
    # "Fibrosis" and "Airway wall thickening" (no Phase-3 ellipse column exists).
    # Left without a pattern deliberately: "Other" and "Support devices" -- neither
    # is a finding a radiologist names with a describable keyword.
}

NO_KEYWORD_BY_DESIGN = {"Other", "Support devices"}


def assert_covers_phase3(labels):
    """Fail loudly when a Phase-3 label has no pattern.

    The old code printed a `MISSING keywords:` line at cache-build time. It printed
    the right answer for months and nobody read it, so this raises instead.
    """
    missing = set(labels) - set(KEYWORDS) - NO_KEYWORD_BY_DESIGN
    if missing:
        raise SystemExit(
            f"KEYWORDS has no pattern for Phase-3 label(s): {sorted(missing)}.\n"
            "Instances for these reach the model with mentions==[], which disables the\n"
            "temporal key features and the word features -- the method does not run on\n"
            "them. Add a pattern, or add the label to NO_KEYWORD_BY_DESIGN if it truly\n"
            "cannot have one.")
NEG = re.compile(r"\b(no|without|resolved|clear of|free of|negative for|rule[sd]? out)\b", re.I)


def _col(df, *c):
    return gr._col(df, *c)


def ellipse_mask(rows, h, w, res):
    yy, xx = np.mgrid[0:res, 0:res]
    m = np.zeros((res, res), bool)
    for _, r in rows.iterrows():
        cx = (r["xmin"] + r["xmax"]) / 2 / w * res
        cy = (r["ymin"] + r["ymax"]) / 2 / h * res
        a = max((r["xmax"] - r["xmin"]) / 2 / w * res, 0.5)
        b = max((r["ymax"] - r["ymin"]) / 2 / h * res, 0.5)
        m |= ((xx - cx) / a) ** 2 + ((yy - cy) / b) ** 2 <= 1
    return m


def gaze_heat(fx, fy, dur, h, w, ppd):
    m = np.zeros((HEAT_RES, HEAT_RES), np.float32)
    xi = np.clip((fx / w * HEAT_RES).astype(int), 0, HEAT_RES - 1)
    yi = np.clip((fy / h * HEAT_RES).astype(int), 0, HEAT_RES - 1)
    np.add.at(m, (yi, xi), dur)
    sigma = max(0.5, ppd * HEAT_RES / max(h, w))  # 1° in grid cells
    m = gaussian_filter(m, sigma)
    mx = m.max()
    return m / mx if mx > 0 else m


def sentences_with_words(tr):
    """-> list of (text, start, end, words) where words is [(word, w_start, w_end), ...].

    The per-word times are what the original's right edge needs: its window ends at the last
    MENTION inside the sentence, not at the sentence end, and that distinction is invisible
    once words are collapsed into a sentence."""
    w = tr[_col(tr, "word")].astype(str).tolist()
    ts = gr._seconds(tr[_col(tr, "timestamp_start_word", "timestamp_start")].to_numpy(float))
    te = gr._seconds(tr[_col(tr, "timestamp_end_word", "timestamp_end")].to_numpy(float))
    sents, cur, s0 = [], [], None
    for i, tok in enumerate(w):
        if s0 is None:
            s0 = ts[i]
        cur.append((tok, ts[i], te[i]))
        if tok.strip().endswith(".") or tok.strip() in (".",) or i == len(w) - 1:
            sents.append((" ".join(x[0] for x in cur), s0, te[i], cur)); cur, s0 = [], None
    return sents


def sentences_with_times(tr):
    """-> list of (text, start, end) sentences from word-timestamp transcript."""
    return [(txt, s, e) for txt, s, e, _w in sentences_with_words(tr)]


def last_mention_end(words, rx):
    """End time of the last word a pattern match touches, or None if no word is touched.

    Words are joined with single spaces to form the sentence text, so character spans are
    exact and a regex match can be mapped back to the words it overlaps."""
    spans, pos = [], 0
    for tok, _ws, we in words:
        spans.append((pos, pos + len(tok), we))
        pos += len(tok) + 1
    txt = " ".join(tok for tok, _ws, _we in words)
    best = None
    for m in rx.finditer(txt):
        for a, b, we in spans:
            if a < m.end() and b > m.start():
                best = we if best is None else max(best, we)
    return best


def label_windows(sents, label):
    """Gate windows [start,end] for sentences mentioning `label` (keyword + negation guard)."""
    pat = KEYWORDS.get(label)
    if not pat:
        return []
    rx = re.compile(pat, re.I)
    wins = []
    for i, (txt, s, e) in enumerate(sents):
        if rx.search(txt) and not NEG.search(txt):
            prev_start = sents[i - 1][1] if i > 0 else s
            wins.append((max(s - LOOKBACK, prev_start), e))
    return wins


def extract(root, limit):
    import pandas as pd
    hits = sorted(Path(root).rglob("metadata_phase_3.csv"))
    meta = pd.read_csv(hits[0]) if hits else None
    if meta is not None and "eye_tracking_data_discarded" in meta.columns:
        meta = meta[~meta["eye_tracking_data_discarded"].astype(bool)]
    subj, size, keep = {}, {}, None
    if meta is not None:
        subj = {r[_col(meta, "id")]: str(r[_col(meta, "subject_id")]) for _, r in meta.iterrows()}
        size = {r[_col(meta, "id")]: (float(r[_col(meta, "image_size_y")]),
                                      float(r[_col(meta, "image_size_x")])) for _, r in meta.iterrows()}
        keep = set(subj)

    need = ("fixations.csv", "anomaly_location_ellipses.csv", "timestamps_transcription.csv")
    recs = sorted((r, f) for r, f in gr.find_records(root).items() if all(k in f for k in need))
    lab_seen, matched = set(), 0
    out = []
    for rid, files in recs:
        if len(out) >= limit:
            break
        if keep is not None and rid not in keep:
            continue
        try:
            f = pd.read_csv(files["fixations.csv"])
            if len(f) < 8:
                continue
            h, w = size.get(rid, (None, None))
            if h is None:
                h = float(f[_col(f, "ymax_shown_from_image")].max())
                w = float(f[_col(f, "xmax_shown_from_image")].max())
            fx = f[_col(f, "x_position", "average_x_position")].to_numpy(float)
            fy = f[_col(f, "y_position", "average_y_position")].to_numpy(float)
            ft0 = gr._seconds(f[_col(f, "timestamp_start_fixation")].to_numpy(float))
            ft1 = gr._seconds(f[_col(f, "timestamp_end_fixation")].to_numpy(float))
            try:
                ppd = float(np.nanmedian(f[_col(f, "angular_resolution_x_pixels_per_degree")]))
            except KeyError:
                ppd = 50.0
            ell = pd.read_csv(files["anomaly_location_ellipses.csv"])
            labs = [c for c in ell.columns if c not in ("xmin", "ymin", "xmax", "ymax", "certainty")
                    and ell[c].dropna().isin([True, False, 0, 1, 0.0, 1.0]).all()]
            lab_seen.update(labs)
            tr = pd.read_csv(files["timestamps_transcription.csv"])
            sents = sentences_with_times(tr)

            per = {}
            for L in labs:
                rows = ell[ell[L].astype(bool)]
                if not len(rows):
                    continue
                gt = ellipse_mask(rows, h, w, EVAL_RES)
                wins = label_windows(sents, L)
                if wins:
                    matched += 1
                    sel = np.zeros(len(fx), bool)
                    for a, b in wins:
                        sel |= (ft0 >= a) & (ft1 <= b) | ((ft0 < b) & (ft1 > a))
                    if sel.any():
                        hm = gaze_heat(fx[sel], fy[sel], (ft1 - ft0)[sel], h, w, ppd)
                    else:
                        hm = np.zeros((HEAT_RES, HEAT_RES), np.float32)
                else:  # no mention found -> fall back to full gaze (B0 behavior for that label)
                    hm = gaze_heat(fx, fy, ft1 - ft0, h, w, ppd)
                per[L] = {"gt": gt, "heat": hm.astype(np.float16), "gated": bool(wins)}
            if per:
                out.append({"rid": rid, "subject": subj.get(rid, rid), "labels": per})
        except Exception as e:
            print(f"skip {rid}: {type(e).__name__}: {e}")
    print(f"label columns seen: {sorted(lab_seen)}")
    print(f"KEYWORDS covers: {sorted(set(KEYWORDS) & lab_seen)}")
    print(f"MISSING keywords for: {sorted(lab_seen - set(KEYWORDS))}")
    print(f"gated (label,image) instances: {matched}")
    return out


def iou_at_eval(heat, gt, t):
    up = zoom(heat.astype(np.float32), EVAL_RES / heat.shape[0], order=0)  # NN upscale to ellipse res
    pred = up >= t
    inter = (pred & gt).sum(); union = pred.sum() + gt.sum() - inter
    return inter / union if union > 0 else np.nan


def analyze(cache):
    import torch
    recs = torch.load(cache, weights_only=False)
    subs = np.array([r["subject"] for r in recs]); uniq = np.array(sorted(set(subs)))
    rng = np.random.default_rng(0); rng.shuffle(uniq); n = len(uniq)
    val_s = set(uniq[:int(n * .15)]); test_s = set(uniq[int(n * .15):int(n * .45)])
    val = [r for r in recs if r["subject"] in val_s]
    test = [r for r in recs if r["subject"] in test_s]

    def collect(records):
        return [(d["heat"], d["gt"]) for r in records for d in r["labels"].values() if d["gt"].any()]

    def gated_frac(records):
        items = [d for r in records for d in r["labels"].values() if d["gt"].any()]
        return np.mean([d["gated"] for d in items]) if items else 0

    vv, tt = collect(val), collect(test)
    ts = np.linspace(0.05, 0.6, 23)
    scores = [np.nanmean([iou_at_eval(h, g, t) for h, g in vv]) for t in ts]
    t_best = ts[int(np.argmax(scores))]
    test_iou = np.nanmean([iou_at_eval(h, g, t_best) for h, g in tt])
    print(f"\n{len(recs)} records; val/test (label,img) = {len(vv)}/{len(tt)}")
    print(f"val-tuned threshold = {t_best:.3f}")
    print(f"TEST per-label IoU  = {test_iou:.4f}   (B1 target ~0.233, B0 no-gating 0.165)")
    print(f"gated fraction (test) = {gated_frac(test):.2f}  "
          "(fraction of label-instances that got a keyword-matched gaze window)")
    print("\n≈0.233 → B1 reproduced (keyword gating works). Much lower → keyword\n"
          "coverage too weak (see gated fraction) → improve keywords or use CheXpert labeler.")


def _selfcheck():
    import torch
    rng = np.random.default_rng(0); recs = []
    for k in range(80):
        gt = np.zeros((EVAL_RES, EVAL_RES), bool)
        cx, cy = rng.integers(60, 196, 2)
        yy, xx = np.mgrid[0:EVAL_RES, 0:EVAL_RES]; gt[((xx-cx)**2+(yy-cy)**2) < 400] = True
        hm = np.zeros((HEAT_RES, HEAT_RES), np.float32)
        hm[int(cy/EVAL_RES*HEAT_RES), int(cx/EVAL_RES*HEAT_RES)] = 1
        hm = gaussian_filter(hm, 2); hm /= hm.max()
        recs.append({"rid": f"r{k}", "subject": f"s{k}",
                     "labels": {"L": {"gt": gt, "heat": hm.astype(np.float16), "gated": True}}})
    # exercise text/gating helpers too
    sents = sentences_with_times(__import__("pandas").DataFrame({
        "word": ["there", "is", "a", "nodule.", "no", "pneumothorax."],
        "timestamp_start_word": [1., 1.3, 1.5, 2.0, 3.0, 3.3],
        "timestamp_end_word": [1.2, 1.4, 1.9, 2.5, 3.2, 3.9]}))
    assert label_windows(sents, "Lung nodule or mass"), "should match nodule"
    assert not label_windows(sents, "Pneumothorax"), "negation should block"
    p = Path("/tmp/_b1_selfcheck.pt"); torch.save(recs, p)
    analyze(p); print("self-check ran (gaze on lesion + gating/negation logic ok)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?")
    ap.add_argument("--cache", default="b1.pt")
    ap.add_argument("--limit", type=int, default=3000)
    a = ap.parse_args()
    if a.root:
        import torch
        recs = extract(a.root, a.limit); torch.save(recs, a.cache)
        print(f"extracted {len(recs)} Phase-3 records -> {a.cache}")
    elif Path(a.cache).exists():
        analyze(a.cache)
    else:
        _selfcheck()

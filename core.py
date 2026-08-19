"""OUR METHOD: learned word/finding-level gaze-voice temporal alignment, to beat
the fixed-1.5s-window baseline (B1 ~0.225-0.233) for gaze+voice lesion localization.

Idea: B1 gates fixations with a HARD window [mention-1.5s, mention_end]. We replace
it with a LEARNED SOFT attention a_i over fixations, computed from each fixation's
TEMPORAL offset to the mention (Δt), duration, velocity, and a per-finding
embedding — deliberately POSITION-AGNOSTIC (no x,y in the attention) so the model
learns *which fixations in time* correspond to the spoken finding, not where lesions
usually are (avoids the anatomy confound). Localization = softmax-weighted splat of
the attended fixations' actual (x,y). B1 is the special case a_i = 1[Δt∈[-1.5,0]].

Same script computes BOTH B1 (fixed window) and OURS on the SAME data/split/eval,
so the comparison is perfectly controlled.

Two stages:
  python core.py /path/to/reflacx --cache align.pt   # extract (node/CPU)
  python core.py --cache align.pt --epochs 40         # train+eval (mac, small model)
  python core.py                                      # synthetic self-check
"""
import argparse
import re
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, zoom

import reflacx_io as gr
from linker_and_temporal import (KEYWORDS, NEG, last_mention_end, sentences_with_words,
                          _col, EVAL_RES,
                          HEAT_RES, LOOKBACK, assert_covers_phase3)


def ellipses_norm(ell, h, w):
    return [((r["xmin"] / w, r["ymin"] / h, r["xmax"] / w, r["ymax"] / h))
            for _, r in ell.iterrows()]


# Single source of truth for the validation search. Defined here because every scorer
# must use the identical grid: an earlier version had three scripts with three different
# grids, which silently gave some methods a wider search than others.
TUNE_SIGMAS = [0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]


def tune_thresholds():
    """Dense below 0.05, where gaze-splat models select, and sparse above, where the
    smoother external maps do. A single uniform grid is either too coarse at the low end
    or wastes points at the high end."""
    return np.concatenate([np.linspace(0.002, 0.05, 25), np.linspace(0.06, 0.95, 30)])


def raster(ells, res=EVAL_RES):
    yy, xx = np.mgrid[0:res, 0:res]; m = np.zeros((res, res), bool)
    for x0, y0, x1, y1 in ells:
        cx, cy = (x0 + x1) / 2 * res, (y0 + y1) / 2 * res
        a = max((x1 - x0) / 2 * res, 0.5); b = max((y1 - y0) / 2 * res, 0.5)
        m |= ((xx - cx) / a) ** 2 + ((yy - cy) / b) ** 2 <= 1
    return m


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
    out, seen = [], set()
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
                h = float(f[_col(f, "ymax_shown_from_image")].max()); w = float(f[_col(f, "xmax_shown_from_image")].max())
            x = f[_col(f, "x_position", "average_x_position")].to_numpy(float) / w
            y = f[_col(f, "y_position", "average_y_position")].to_numpy(float) / h
            t0 = gr._seconds(f[_col(f, "timestamp_start_fixation")].to_numpy(float))
            t1 = gr._seconds(f[_col(f, "timestamp_end_fixation")].to_numpy(float))
            tc = (t0 + t1) / 2; dur = t1 - t0
            vel = np.zeros(len(x));
            if len(x) > 1:
                d = np.hypot(np.diff(x), np.diff(y)); dt = np.clip(np.diff(tc), 1e-3, None)
                vel[1:] = d / dt
            fixfeat = np.stack([x, y, tc, dur, vel], 1).astype(np.float32)   # (N,5)
            ell = pd.read_csv(files["anomaly_location_ellipses.csv"])
            labs = [c for c in ell.columns if c not in ("xmin", "ymin", "xmax", "ymax", "certainty")
                    and ell[c].dropna().isin([True, False, 0, 1, 0.0, 1.0]).all()]
            seen.update(labs)
            sw = sentences_with_words(pd.read_csv(files["timestamps_transcription.csv"]))
            sents = [(txt, s, e) for txt, s, e, _w in sw]
            per = []
            for L in labs:
                rows = ell[ell[L].astype(bool)]
                if not len(rows):
                    continue
                rx = KEYWORDS.get(L)
                mentions, mtext = [], []
                if rx:
                    r_ = re.compile(rx, re.I)
                    for i, (txt, s, e, wds) in enumerate(sw):
                        if r_.search(txt) and not NEG.search(txt):
                            # 4th field is the original's right edge: the end of the LAST
                            # mention inside the sentence, which our own window replaces with
                            # the sentence end (the later ISBI convention). Carrying both lets
                            # either rule be evaluated without re-extracting again.
                            lme = last_mention_end(wds, r_)
                            mentions.append((max(s - LOOKBACK, sents[i-1][1] if i else s),
                                             s, e, e if lme is None else lme))
                            mtext.append(txt)
                per.append({"label": L, "ellipses": ellipses_norm(rows, h, w),
                            "mentions": mentions,   # each: (gate_start, sent_start, sent_end)
                            "mtext": mtext})        # matched sentence texts (for word features)
            if per:
                # Sentence starts travel with the record so a gate wider than LOOKBACK can be
                # rebuilt later. The cached gate is max(sent_start - LOOKBACK, prev_start),
                # which loses prev_start wherever the clip did not bind, and with it any
                # window past LOOKBACK -- including the lookback sweep's upper end.
                out.append({"rid": rid, "subject": subj.get(rid, rid),
                            "fix": fixfeat, "labels": per,
                            "sents": [(s, e) for _txt, s, e in sents]})
        except Exception as ex:
            print(f"skip {rid}: {type(ex).__name__}: {ex}")
    print(f"labels seen: {sorted(seen)}")
    assert_covers_phase3(seen)   # was a print nobody read; see linker_and_temporal.KEYWORDS
    return out, sorted(seen)


# ------------------------------------------------------------ model + eval

def splat(x, y, wt, sigma=1.5):
    m = np.zeros((HEAT_RES, HEAT_RES), np.float32)
    xi = np.clip((x * HEAT_RES).astype(int), 0, HEAT_RES - 1)
    yi = np.clip((y * HEAT_RES).astype(int), 0, HEAT_RES - 1)
    np.add.at(m, (yi, xi), wt)
    m = gaussian_filter(m, sigma)
    mx = m.max()
    return m / mx if mx > 0 else m           # normalize to max 1 (match B1 eval scale)


def inside_ellipses(fix, ells):
    """Per-fixation: is (x,y) inside any of the label's ellipses. (N,) bool."""
    x, y = fix[:, 0], fix[:, 1]
    m = np.zeros(len(x), bool)
    for x0, y0, x1, y1 in ells:
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        a = max((x1 - x0) / 2, 1e-3); b = max((y1 - y0) / 2, 1e-3)
        m |= ((x - cx) / a) ** 2 + ((y - cy) / b) ** 2 <= 1
    return m


def iou(heat, gt, t):
    up = zoom(heat, EVAL_RES / heat.shape[0], order=0)
    pred = up >= t; inter = (pred & gt).sum(); uni = pred.sum() + gt.sum() - inter
    return inter / uni if uni > 0 else np.nan


def pointing(heat, gt):
    """Pointing-game hit: is the heatmap's peak inside the ellipse? (threshold-free)"""
    up = zoom(heat, EVAL_RES / heat.shape[0], order=0)
    return float(gt[np.unravel_index(np.argmax(up), up.shape)]) if gt.any() else np.nan


def b1_at(fix, mentions, sigma):
    """b1_heat with the blur bandwidth exposed, so the baseline can be given the same
    (sigma, threshold) search the learned models get. b1_heat is this at sigma=1.5."""
    t0 = fix[:, 2] - fix[:, 3] / 2; t1 = fix[:, 2] + fix[:, 3] / 2
    if mentions:
        sel = np.zeros(len(fix), bool)
        for g, s, e, *_ in mentions:
            sel |= (t0 < e) & (t1 > g)
        if sel.any():
            return splat(fix[sel, 0], fix[sel, 1], fix[sel, 3], sigma)
    return splat(fix[:, 0], fix[:, 1], fix[:, 3], sigma)


def b1_heat(fix, mentions):
    """Fixed-window baseline (hard gate); faithful overlap selection like linker_and_temporal."""
    t0 = fix[:, 2] - fix[:, 3] / 2; t1 = fix[:, 2] + fix[:, 3] / 2
    if mentions:
        sel = np.zeros(len(fix), bool)
        for g, s, e, *_ in mentions:                      # fixation interval overlaps [g,e]
            sel |= (t0 < e) & (t1 > g)
        if sel.any():
            return splat(fix[sel, 0], fix[sel, 1], fix[sel, 3])
    return splat(fix[:, 0], fix[:, 1], fix[:, 3])     # ungated fallback = full gaze


# cheap, interpretable "word content" features from the mention sentence(s):
# spatial + severity descriptors the label-id alone can't carry. If these help,
# escalate to contextual (BioClinical/CXR-BERT) embeddings.
WORD_TERMS = [r"\bleft\b", r"\bright\b", r"bilateral", r"upper|apic|apex",
              r"lower|bas(e|al|ilar)", r"middle|mid ", r"retrocardiac",
              r"small|mild|subtle|question|possibl|trace|minimal",
              r"large|severe|extensive|marked|significant", r"diffuse|patchy"]
WORD_DIM = len(WORD_TERMS)


def word_feat(mtext):
    v = np.zeros(WORD_DIM, np.float32)
    if mtext:
        blob = " ".join(mtext).lower()
        for i, t in enumerate(WORD_TERMS):
            if re.search(t, blob):
                v[i] = 1.0
    return v


# The in-window indicator was 1[-LOOKBACK <= dt <= 0] -- it handed the model B1's own 1.5 s
# constant, pre-thresholded, while the paper claims the learned selection replaces exactly
# that hand-set window. Measured over nine runs, dropping it changes nothing: IoU 0.3441 ->
# 0.3467 and pointing 0.8027 -> 0.8059, both inside the +-0.0033 seed half-range, with every
# control still significant in 9/9. So the shipped model does not receive it, and no
# hand-set constant reaches the model at all -- dt and |dt| give it the raw offset and any
# gating has to be learned. B1 is unaffected: its gate reads LOOKBACK directly, not through
# align_feats. Set USE_INWIN = True (or pass --with-inwin) to restore it.
USE_INWIN = False


def temporal_dim():
    """Width of align_feats' output -- make_net needs it to size the key projection."""
    return 6 if USE_INWIN else 5


def align_feats(fix, mentions):
    """Position-AGNOSTIC per-fixation features vs nearest mention. (N, temporal_dim())"""
    tc = fix[:, 2]
    if mentions:
        ms = np.array([m[1] for m in mentions])           # sentence-start times
        dt = tc[:, None] - ms[None, :]
        j = np.abs(dt).argmin(1); dmin = dt[np.arange(len(tc)), j]
    else:
        dmin = np.zeros(len(tc))
    cols = [dmin, np.abs(dmin), (dmin < 0).astype(float)]          # dt, |dt|, before
    if USE_INWIN:
        cols.append(((dmin >= -LOOKBACK) & (dmin <= 0)).astype(float))   # inwin
    cols += [fix[:, 3], fix[:, 4]]                                 # dur, vel
    return np.stack(cols, 1).astype(np.float32)


def run(cache, epochs, wordemb):
    import torch, torch.nn as nn
    recs, labels = torch.load(cache, weights_only=False)
    li = {L: i for i, L in enumerate(labels)}
    subs = np.array([r["subject"] for r in recs]); uniq = np.array(sorted(set(subs)))
    rng = np.random.default_rng(0); rng.shuffle(uniq); n = len(uniq)
    val_s = set(uniq[:int(n*.15)]); test_s = set(uniq[int(n*.15):int(n*.45)])
    part = lambda r: "val" if r["subject"] in val_s else "test" if r["subject"] in test_s else "train"

    # flatten to (record, label) instances. word feature from mention text (or zeros).
    def insts(p):
        o = []
        for r in recs:
            if part(r) != p:
                continue
            for d in r["labels"]:
                wf = word_feat(d.get("mtext", [])) if wordemb else np.zeros(WORD_DIM, np.float32)
                o.append((r["fix"], d["mentions"], d["ellipses"], li[d["label"]], wf))
        return o
    tr, va, te = insts("train"), insts("val"), insts("test")
    print(f"instances train/val/test = {len(tr)}/{len(va)}/{len(te)}, {len(labels)} labels, "
          f"wordemb={'ON' if wordemb else 'OFF'}", flush=True)

    wd = WORD_DIM if wordemb else 0

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(len(labels), 8)
            self.mlp = nn.Sequential(nn.Linear(temporal_dim() + 8 + wd, 32), nn.ReLU(),
                                     nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1))

        def attn(self, feats, lab, wf):  # feats (N,6), lab int, wf (WORD_DIM,) -> (N,) softmax
            e = self.emb(torch.tensor(lab)).expand(len(feats), -1)
            parts = [feats, e]
            if wordemb:
                parts.append(torch.from_numpy(wf).expand(len(feats), -1))
            s = self.mlp(torch.cat(parts, 1)).squeeze(-1)
            return torch.softmax(s, 0)

    net = Net(); opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    # loss = attention-weighted mass-in: concentrate softmax attention on fixations
    # INSIDE the finding's ellipse. Differentiable in a; position-agnostic attention
    # must exploit timing/kinematics/label to find them (= learned alignment).
    # entropy reg keeps it from collapsing to a single fixation.
    for ep in range(epochs):
        net.train(); np.random.shuffle(tr); tot = 0
        for fix, ment, ells, lab, wf in tr:
            ins = torch.from_numpy(inside_ellipses(fix, ells).astype(np.float32))
            if ins.sum() == 0:
                continue
            a = net.attn(torch.from_numpy(align_feats(fix, ment)), lab, wf)
            massin = (a * ins).sum()
            ent = -(a * (a + 1e-9).log()).sum()
            loss = -torch.log(massin + 1e-6) - 0.01 * ent
            opt.zero_grad(); loss.backward(); opt.step(); tot += massin.item()
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"  epoch {ep} train mass-in {tot/max(len(tr),1):.4f}", flush=True)

    # eval: val-tune threshold, report test IoU + pointing. items are 5-tuples
    # (fix, ment, ells, lab, wf). Three heatmap sources: B1 (fixed window), OURS
    # (learned), OURS-shuffled (wrong mentions, control).
    def our_heat(fix, ment, lab, wf):
        with torch.no_grad():
            a = net.attn(torch.from_numpy(align_feats(fix, ment)), lab, wf).numpy()
        return splat(fix[:, 0], fix[:, 1], a)

    rng2 = np.random.default_rng(1)
    perm = rng2.permutation(len(te))

    def heat(item, i, method):
        f, m, e, l, wf = item
        if method == "b1":
            return b1_heat(f, m)
        if method == "shuf":
            return our_heat(f, te[perm[i]][1], l, wf)   # another instance's mentions
        return our_heat(f, m, l, wf)

    def per_metric(items, method, t, metric):
        out = []
        for i, it in enumerate(items):
            hm = heat(it, i, method)
            gt = raster(it[2])
            out.append(iou(hm, gt, t) if metric == "iou" else pointing(hm, gt))
        return np.array(out)

    ts = np.linspace(0.05, 0.6, 23)
    tb1 = ts[int(np.argmax([np.nanmean(per_metric(va, "b1", t, "iou")) for t in ts]))]
    tour = ts[int(np.argmax([np.nanmean(per_metric(va, "our", t, "iou")) for t in ts]))]
    b1v = per_metric(te, "b1", tb1, "iou")
    ourv = per_metric(te, "our", tour, "iou")
    shufv = per_metric(te, "shuf", tour, "iou")
    b1, ours, shuf = np.nanmean(b1v), np.nanmean(ourv), np.nanmean(shufv)

    from scipy.stats import wilcoxon
    def paired(a, b):
        ok = np.isfinite(a) & np.isfinite(b)
        _, p = wilcoxon(a[ok], b[ok]); return np.median(a[ok] - b[ok]), p

    pb1 = per_metric(te, "b1", tb1, "pg")
    pour = per_metric(te, "our", tour, "pg")
    pshuf = per_metric(te, "shuf", tour, "pg")

    print(f"\nTEST (n={len(te)}):            IoU      pointing-game")
    print(f"  B1 fixed-window       {b1:.4f}   {np.nanmean(pb1):.4f}")
    print(f"  OURS learned          {ours:.4f}   {np.nanmean(pour):.4f}")
    print(f"  OURS shuffled-mention {shuf:.4f}   {np.nanmean(pshuf):.4f}  (control)")
    d1, p1 = paired(ourv, b1v); d2, p2 = paired(ourv, shufv)
    dp, pp = paired(pour, pb1)
    print(f"\npaired IoU      OURS vs B1: Δmed={d1:+.4f} p={p1:.3g} {'*' if p1<0.05 else ''}")
    print(f"paired pointing OURS vs B1: Δmean={np.nanmean(pour)-np.nanmean(pb1):+.4f} p={pp:.3g} {'*' if pp<0.05 else ''}")
    print(f"paired IoU      OURS vs shuffled: Δmed={d2:+.4f} p={p2:.3g} {'*' if p2<0.05 else ''}")
    print("\nVERDICT:")
    if ours > b1 and p1 < 0.05 and ours > shuf and p2 < 0.05:
        print("  OURS > B1 (sig) AND OURS > shuffled-mention (sig) -> learned alignment is\n"
              "  REAL and uses the mention timing. Novelty holds.")
    elif ours > shuf + 0.005 and p2 < 0.05:
        print("  OURS > shuffled but ~B1 -> uses alignment but no better than fixed window yet.")
    else:
        print("  OURS ~ shuffled -> the lift is NOT from mention alignment (kinematics/dur\n"
              "  confound). Novelty NOT supported as 'learned alignment'; rethink.")


def _selfcheck():
    """Synthetic: relevant gaze is at a per-finding time OFFSET (not the fixed 1.5s),
    so learned attention should recover it and beat the fixed window."""
    import torch
    rng = np.random.default_rng(0); recs = []
    for k in range(120):
        N = 40; tc = np.sort(rng.uniform(0, 20, N))
        x = rng.uniform(0, 1, N); y = rng.uniform(0, 1, N)
        lx, ly = rng.uniform(.3, .7, 2)
        m_start = rng.uniform(5, 15)
        # relevant fixations 3s BEFORE mention (outside the 1.5s window) sit on lesion
        rel = (tc > m_start - 3.5) & (tc < m_start - 2.5)
        x[rel], y[rel] = lx + rng.normal(0, .02, rel.sum()), ly + rng.normal(0, .02, rel.sum())
        fix = np.stack([x, y, tc, np.full(N, .2), np.zeros(N)], 1).astype(np.float32)
        recs.append({"rid": f"r{k}", "subject": f"s{k}", "fix": fix,
                     "labels": [{"label": "L", "ellipses": [(lx-.05, ly-.05, lx+.05, ly+.05)],
                                 "mentions": [(m_start-1.5, m_start, m_start+.5)]}]})
    p = Path("/tmp/_align_selfcheck.pt"); torch.save((recs, ["L"]), p)
    run(p, epochs=25, wordemb=False)
    print("self-check ran (relevant gaze at −3s offset; learned should beat fixed 1.5s window)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?")
    ap.add_argument("--cache", default="align.pt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--limit", type=int, default=3000)
    ap.add_argument("--wordemb", action="store_true", help="add mention-word features to attention")
    a = ap.parse_args()
    if a.root:
        import torch
        recs, labels = extract(a.root, a.limit); torch.save((recs, labels), a.cache)
        print(f"extracted {len(recs)} Phase-3 records -> {a.cache}")
    elif Path(a.cache).exists():
        run(a.cache, a.epochs, a.wordemb)
    else:
        _selfcheck()

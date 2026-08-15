"""Two controls the current paper is missing, both raised in review.

(A) COUNTERFACTUAL SPATIAL SEMANTICS. Spatial-word masking shows the model *depends*
    on spatial words; it does not show it maps them to the right place. Here the word is
    swapped rather than removed -- left<->right, upper<->lower -- on the same frozen model,
    and the predicted heat map's centroid is tracked. Because this model never sees the
    image, no image flip is needed: the text is the only thing that changes.

    The test is run in two stages so it cannot be passed by accident:
      1. Does the model separate the words at all? Compare mean centroid for instances
         whose mention says only "left" against those that say only "right". If these
         coincide, the word carries no spatial meaning to the model and stage 2 is moot.
      2. Does swapping move the prediction *toward* the other word's region? For each
         left-only instance, swap to right and measure how far the centroid travels along
         the axis separating the two populations, as a fraction of that separation.

    Note on convention: in a frontal radiograph the patient's left is the image's right.
    The direction the model learned is measured, not assumed, so this also reveals whether
    it picked up the inversion.

(B) LABEL-ONLY ANATOMICAL PRIOR. A baseline with no gaze and no text: for each finding,
    the mean training ellipse mask. Findings sit in characteristic places -- our best
    per-label result is enlarged cardiac silhouette, the most positionally predictable one
    -- so without this the size of the gaze contribution is unknown. It receives the same
    (bandwidth, threshold) validation search every other method gets.

    This does not threaten the causal claim: a pure prior is unaffected by position-shuffle,
    and shuffle degrades the model in 9/9 runs. It bounds the *magnitude* of the claim.

  python prior_and_swap.py --cache align.pt --epochs 40
"""
import argparse

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import wilcoxon

from core import (iou, pointing, raster, word_feat, EVAL_RES,
                         HEAT_RES, TUNE_SIGMAS, tune_thresholds)
from selector import train_model, predict_raw, blur_norm
from evaluate import FUSION, POS_MODE

LEFT, RIGHT, UPPER, LOWER = 0, 1, 3, 4      # indices into WORD_TERMS


def split(recs, split_seed=0):
    subs = np.array([r["subject"] for r in recs]); uniq = np.array(sorted(set(subs)))
    rng = np.random.default_rng(split_seed); rng.shuffle(uniq); n = len(uniq)
    val_s = set(uniq[:int(n * .15)]); test_s = set(uniq[int(n * .15):int(n * .45)])
    return lambda r: ("val" if r["subject"] in val_s
                      else "test" if r["subject"] in test_s else "train")


def label_prior(items):
    """Mean rasterized ellipse mask per label. items: iterable of (ellipses, label),
    and must come from the TRAIN split only. Lifted out of (B) below unchanged so the
    training-free prior-intersect-scanpath baseline uses this construction rather than
    a second copy of it that could drift."""
    per = {}
    for ells, L in items:
        per.setdefault(L, []).append(raster(ells))
    return {L: np.mean(v, 0).astype(np.float32) for L, v in per.items()}


def prior_heat(p, sigma):
    """Blur + max-normalize a label prior mask. p is at EVAL_RES; sigma is quoted in
    HEAT_RES cells like every other bandwidth here, so it is rescaled."""
    if p is None:
        return np.zeros((EVAL_RES, EVAL_RES), np.float32)
    m = gaussian_filter(p, sigma * EVAL_RES / HEAT_RES) if sigma > 0 else p
    mx = m.max()
    return m / mx if mx > 0 else m


def centroid(hm):
    """Intensity-weighted centre of the predicted heat map, in normalized coords."""
    m = hm / hm.sum() if hm.sum() > 0 else hm
    ys, xs = np.mgrid[0:hm.shape[0], 0:hm.shape[1]]
    return float((m * xs).sum() / hm.shape[1]), float((m * ys).sum() / hm.shape[0])


def swap(wf, a, b):
    w = wf.copy(); w[a], w[b] = wf[b], wf[a]; return w


def run(cache, epochs, seed=0, split_seed=0):
    import torch
    recs, labels = torch.load(cache, weights_only=False)
    li = {L: i for i, L in enumerate(labels)}
    part = split(recs, split_seed)

    def insts(p):
        return [(r["fix"], d["mentions"], d["ellipses"], li[d["label"]],
                 word_feat(d.get("mtext", [])), d["label"])
                for r in recs if part(r) == p for d in r["labels"]]
    tr, va, te = insts("train"), insts("val"), insts("test")
    print(f"train/val/test = {len(tr)}/{len(va)}/{len(te)}", flush=True)

    net = train_model([t[:5] for t in tr], labels, use_position=True, epochs=epochs,
                      use_text=True, fusion=FUSION, pos_mode=POS_MODE, seed=seed)

    def raw_of(item, wf):
        f, m, l = item[0], item[1], item[3]
        return predict_raw(net, f, m, l, True, use_text=True, wf=wf, pos_mode=POS_MODE)

    # The Final model gets the same (bandwidth, threshold) validation search the prior gets
    # below. Hardcoding them here would reintroduce exactly the asymmetry this project
    # removed for B1 and for RadZero -- and the tuned pair is not constant across runs.
    _ts = tune_thresholds()
    _best = (TUNE_SIGMAS[0], _ts[0], -1.0)
    for _sg in TUNE_SIGMAS:
        _c = [(blur_norm(raw_of(it, it[4]), _sg), raster(it[2])) for it in va]
        for _t in _ts:
            _sc = np.nanmean([iou(h, g, _t) for h, g in _c])
            if _sc > _best[2]:
                _best = (_sg, _t, _sc)
        del _c
    SIGMA_FN, THR_FN = _best[0], _best[1]
    print(f"val-tuned Final: sigma={SIGMA_FN}, threshold={THR_FN:.3f}", flush=True)

    def hm_of(item, wf):
        return blur_norm(raw_of(item, wf), SIGMA_FN)

    # ---------------------------------------------------------------- (A)
    print("\n" + "=" * 70)
    print("(A) COUNTERFACTUAL SPATIAL SEMANTICS")
    print("=" * 70)

    for name, (A, B), axis in (("left / right", (LEFT, RIGHT), 0),
                               ("upper / lower", (UPPER, LOWER), 1)):
        only_a = [it for it in te if it[4][A] == 1 and it[4][B] == 0]
        only_b = [it for it in te if it[4][B] == 1 and it[4][A] == 0]
        if len(only_a) < 15 or len(only_b) < 15:
            print(f"\n{name}: too few exclusive instances "
                  f"({len(only_a)}/{len(only_b)}), skipping")
            continue

        ca = np.array([centroid(hm_of(it, it[4]))[axis] for it in only_a])
        cb = np.array([centroid(hm_of(it, it[4]))[axis] for it in only_b])
        sep = cb.mean() - ca.mean()
        ax_name = "x (0 = image left)" if axis == 0 else "y (0 = image top)"
        print(f"\n{name}   n = {len(only_a)} vs {len(only_b)}   axis {ax_name}")
        print(f"  stage 1 -- are the words separated at all?")
        print(f"    mean centroid: word A {ca.mean():.4f}   word B {cb.mean():.4f}"
              f"   separation {sep:+.4f}")
        _, p_sep = __import__("scipy.stats", fromlist=["mannwhitneyu"]).mannwhitneyu(ca, cb)
        print(f"    Mann-Whitney p = {p_sep:.3g}"
              + ("  -> separated" if p_sep < 0.05 else "  -> NOT separated; stage 2 is moot"))
        if abs(sep) < 1e-6:
            continue

        # stage 2: swap A->B on the A-instances; how far does the centroid travel
        # toward B's mean, as a fraction of the separation?
        moved = np.array([centroid(hm_of(it, swap(it[4], A, B)))[axis] for it in only_a])
        frac = (moved - ca) / sep
        _, p_move = wilcoxon(moved, ca)
        print(f"  stage 2 -- does swapping the word move the prediction toward the other?")
        print(f"    centroid after swap {moved.mean():.4f} (was {ca.mean():.4f}), "
              f"shift {moved.mean()-ca.mean():+.4f}")
        print(f"    as a fraction of the A->B separation: {frac.mean():+.3f} "
              f"(1.0 = fully relocated, 0 = no effect, negative = wrong direction)")
        print(f"    paired Wilcoxon p = {p_move:.3g}")
        print(f"    instances moving the correct way: "
              f"{100*(frac > 0).mean():.0f}%")
        tag = "LR" if axis == 0 else "UD"
        print(f"CF seed={seed} split={split_seed} pair={tag} n_a={len(only_a)} n_b={len(only_b)} "
              f"ca={ca.mean():.4f} cb={cb.mean():.4f} sep={sep:+.4f} p_sep={p_sep:.3g} "
              f"frac={frac.mean():+.4f} p_move={p_move:.3g} correct={100*(frac>0).mean():.1f}")

    # ------------------------------------------------ (A2) the discriminating test
    # (A) shows the words move the prediction in the right direction. It does NOT
    # separate "the model places the word" from "the word is a token correlated with
    # a region the radiologist already looked at": a model that had merely learned
    # left-word -> left-fixations would produce (A) exactly. Conditioning on how
    # bilaterally the gaze is spread does separate them. Where the scanpath never
    # crossed the midline there is nothing to select BETWEEN, so a shift that persists
    # there is the model imposing a direction rather than choosing among fixations.
    print("\n" + "=" * 70)
    print("(A2) DOES THE SHIFT SURVIVE WHEN THE GAZE IS ONE-SIDED?")
    print("=" * 70)

    def side_mass(fix):
        """duration-weighted fraction of fixation mass left of the midline."""
        w = fix[:, 3]
        return float((w * (fix[:, 0] < 0.5)).sum() / max(w.sum(), 1e-9))

    A, B, axis = LEFT, RIGHT, 0
    only_a = [it for it in te if it[4][A] == 1 and it[4][B] == 0]
    if len(only_a) >= 30:
        m = np.array([side_mass(it[0]) for it in only_a])
        # unilateral = >=90% of gaze mass on one side of the midline
        uni = (m >= 0.90) | (m <= 0.10)
        ca = np.array([centroid(hm_of(it, it[4]))[axis] for it in only_a])
        mv = np.array([centroid(hm_of(it, swap(it[4], A, B)))[axis] for it in only_a])
        d = mv - ca
        print(f"  left-only instances: {len(only_a)}  "
              f"({uni.sum()} one-sided, {(~uni).sum()} bilateral)")
        for tag, sel in (("bilateral", ~uni), ("one-sided", uni)):
            if sel.sum() < 10:
                print(f"    {tag:10s} n={sel.sum()} -- too few to test")
                continue
            _, p = wilcoxon(mv[sel], ca[sel])
            print(f"    {tag:10s} n={sel.sum():3d}  centroid {ca[sel].mean():.4f} -> "
                  f"{mv[sel].mean():.4f}   shift {d[sel].mean():+.4f}  p={p:.3g}"
                  + ("  *" if p < 0.05 else "   (n.s.)"))
            print(f"BILAT seed={seed} split={split_seed} group={tag} n={sel.sum()} "
                  f"c0={ca[sel].mean():.4f} c1={mv[sel].mean():.4f} "
                  f"shift={d[sel].mean():+.4f} p={p:.3g}")
        if uni.sum() >= 10 and (~uni).sum() >= 10:
            _, p_diff = __import__("scipy.stats", fromlist=["mannwhitneyu"]).mannwhitneyu(
                d[uni], d[~uni])
            print(f"    shift larger in bilateral than one-sided? "
                  f"Mann-Whitney p = {p_diff:.3g}")
            print(f"BILATDIFF seed={seed} split={split_seed} d_uni={d[uni].mean():+.4f} "
                  f"d_bi={d[~uni].mean():+.4f} p={p_diff:.3g}")
        print("  Reading: a shift that survives in the one-sided group is the model\n"
              "  supplying a direction, not selecting among fixations that differ in side.")
    else:
        print(f"  only {len(only_a)} left-only instances; not enough to split")

    # ---------------------------------------------------------------- (B)
    print("\n" + "=" * 70)
    print("(B) LABEL-ONLY ANATOMICAL PRIOR  (no gaze, no text)")
    print("=" * 70)

    prior = label_prior((it[2], it[5]) for it in tr)

    def prior_hm(item, sigma):
        return prior_heat(prior.get(item[5]), sigma)

    ts = tune_thresholds()
    best = (TUNE_SIGMAS[0], ts[0], -1.0)
    for sg in [0.0] + list(TUNE_SIGMAS):
        cached = [(prior_hm(it, sg), raster(it[2])) for it in va]
        for t in ts:
            sc = np.nanmean([iou(h, g, t) for h, g in cached])
            if sc > best[2]:
                best = (sg, t, sc)
        del cached
    sp, tp = best[0], best[1]
    print(f"  val-tuned: sigma {sp}, threshold {tp:.3f}")

    te_pr = [(prior_hm(it, sp), raster(it[2])) for it in te]
    pr_iou = np.array([iou(h, g, tp) for h, g in te_pr])
    pr_pg = np.array([pointing(h, g) for h, g in te_pr])

    fn_iou, fn_pg = [], []
    for it in te:
        h, g = hm_of(it, it[4]), raster(it[2])
        fn_iou.append(iou(h, g, THR_FN)); fn_pg.append(pointing(h, g))
    fn_iou, fn_pg = np.array(fn_iou), np.array(fn_pg)

    print(f"\n  {'':22s} {'IoU':>8} {'pointing':>10}")
    print(f"  {'label-only prior':22s} {np.nanmean(pr_iou):8.4f} {np.nanmean(pr_pg):10.4f}")
    print(f"  {'Final (gaze+text)':22s} {np.nanmean(fn_iou):8.4f} {np.nanmean(fn_pg):10.4f}")
    # Effect size over the same paired-complete subset the test uses, not over each
    # array's own non-NaN entries. Currently identical (no NaNs), but the two diverge the
    # moment one appears.
    ok = np.isfinite(fn_iou) & np.isfinite(pr_iou)
    _, p_i = wilcoxon(fn_iou[ok], pr_iou[ok]); d_i = np.mean(fn_iou[ok] - pr_iou[ok])
    ok2 = np.isfinite(fn_pg) & np.isfinite(pr_pg)
    _, p_p = wilcoxon(fn_pg[ok2], pr_pg[ok2]); d_p = np.mean(fn_pg[ok2] - pr_pg[ok2])
    print(f"  {'margin':22s} {d_i:+8.4f} {d_p:+10.4f}   (paired n={int(ok.sum())})")
    print(f"  paired p: IoU {p_i:.3g}   pointing {p_p:.3g}")
    labs = np.array([it[5] for it in te])
    print(f"PRIOR seed={seed} split={split_seed} pr_iou={np.nanmean(pr_iou):.4f} "
          f"pr_pg={np.nanmean(pr_pg):.4f} fn_iou={np.nanmean(fn_iou):.4f} "
          f"fn_pg={np.nanmean(fn_pg):.4f} d_iou={d_i:+.4f} "
          f"d_pg={d_p:+.4f} p_iou={p_i:.3g} p_pg={p_p:.3g}")
    for L in sorted(set(labs)):
        m = labs == L
        if m.sum() >= 20:
            print(f"PRIORLAB seed={seed} split={split_seed} label={L.replace(' ','_')} "
                  f"n={m.sum()} prior={np.nanmean(pr_iou[m]):.4f} fn={np.nanmean(fn_iou[m]):.4f} "
                  f"d={np.nanmean(fn_iou[m]-pr_iou[m]):+.4f}")

    print(f"\n  per-label -- where the prior is strong, our margin is what matters:")
    print(f"    {'finding':34s} {'n':>4} {'prior':>7} {'Final':>7} {'Δ':>8}")
    for L in sorted(set(labs), key=lambda L: -np.nanmean(fn_iou[labs == L] - pr_iou[labs == L])):
        m = labs == L
        if m.sum() < 20:
            continue
        print(f"    {L[:34]:34s} {m.sum():4d} {np.nanmean(pr_iou[m]):7.4f} "
              f"{np.nanmean(fn_iou[m]):7.4f} {np.nanmean(fn_iou[m]-pr_iou[m]):+8.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="align.pt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-seed", type=int, default=0, dest="split_seed")
    a = ap.parse_args()
    run(a.cache, a.epochs, a.seed, a.split_seed)

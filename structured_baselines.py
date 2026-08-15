"""How much of the margin over B1 survives with no learned mention-timing association?

The position-shuffle control permutes which fixation holds which position, at evaluation time, with weights frozen.
The shipped scorer is cross-attention, so both projections are a single Linear and the
score of fixation i decomposes additively for a query q fixed per instance:

    s_i = alpha(q)·tau_i + beta(q)·phi(x_i) + c(q)

predict_raw permutes fix[:, :2] and splats onto those same permuted coordinates, while
align_feats reads only fix[:, 2:5]. So beta(q)·phi(x) stays attached to its position and
only the temporal term is scrambled: the shuffled model is a learned label-and-language
to position map evaluated on the real scanpath, carrying permuted logit noise. That is
why it still scores above B1 (0.2945 vs 0.2670 IoU), and it is not evidence of anything
temporal. _additivity_check() below verifies the decomposition numerically on the shipped
architecture, and shows the same identity failing under concat fusion so the check is not
vacuous.

The clean, training-free version of that same mechanism is this script's baseline:

  PRIORSCAN    for each test instance take the mean rasterized ellipse mask for the
               finding's label, built from the TRAIN split only (prior_and_swap's
               gaze-free anatomical prior, same construction, now shared rather than
               copied). Read that mask at each fixation's coordinates, use the values as
               weights, and splat them onto the fixations' REAL coordinates -- the same
               localization head every other method uses. It uses the label and the gaze
               positions. It has no parameters and never sees mention timing.

  PRIORSCAN_W  the same, with the mask first restricted by the mention's spatial words
               (left/right mirror, upper/lower half) -- a second variant, reported
               separately and never merged into the first. Which image half a word points
               to is MEASURED on the train ellipse centres, not assumed: on a frontal
               radiograph the patient's left is the image's right, and quietly getting
               that backwards would understate the variant.

Both get the identical (bandwidth, threshold) validation search on IoU that every other
method receives, and both are compared per-instance to B1, to the gaze-free anatomical
prior, to the position-shuffled model and to the shipped model, with the same paired
Wilcoxon and mean paired difference used throughout.

Every comparison is reported three ways -- all test instances, the mention-matched subset,
and the no-mention subset -- because B1's gate needs a matched mention and these baselines
never do. Same predicate evaluate's B1SUB row uses.

PRIORSCAN_W reads the same spatial-word bits the shipped model receives, so it is
TIMING-FREE, not language-free; PRIORSCAN alone is both.

The baseline is untrained, so on a fixed split its own numbers do not move with --seed;
only the models it is compared against do. Five seeds on one split are five paired
comparisons, not five replications of the baseline.

  python structured_baselines.py --epochs 40 --seed 0 [--split-seed 0]
"""
import argparse

import numpy as np
from scipy.ndimage import zoom
from scipy.stats import wilcoxon

from core import (EVAL_RES, HEAT_RES, LOOKBACK, TUNE_SIGMAS, b1_at, inside_ellipses,
                         iou, pointing,
                         raster, splat, tune_thresholds, word_feat)
from prior_and_swap import (LEFT, RIGHT, UPPER, LOWER, label_prior, prior_heat,
                                    split)
from selector import blur_norm, predict_raw, train_model
from evaluate import FUSION, POS_MODE

TS = tune_thresholds()

# Pixel-centre coordinate grids at EVAL_RES, indexed [row=y, col=x] like every mask here.
_AX = (np.arange(EVAL_RES) + 0.5) / EVAL_RES
GX = np.broadcast_to(_AX, (EVAL_RES, EVAL_RES))
GY = GX.T


def scan_heat(fix, mask, sigma):
    """The baseline's heatmap: prior value at each fixation as that fixation's weight,
    splatted at its real coordinate."""
    w = (np.zeros(len(fix), np.float32) if mask is None
         else mask[np.clip((fix[:, 1] * mask.shape[0]).astype(int), 0, mask.shape[0] - 1),
                   np.clip((fix[:, 0] * mask.shape[1]).astype(int), 0, mask.shape[1] - 1)])
    if w.sum() <= 0:
        w = np.ones(len(fix), np.float32)   # prior has no support on this scanpath: fall
    return splat(fix[:, 0], fix[:, 1], w, sigma)   # back to the ungated map, as b1_at and
                                                   # oracle_at do, rather than emit zeros


def isbi_weights(fix, sents, ment, psi=1.5):
    """Ghelichkhan & Tasdizen (ISBI 2025), §2.2 step 1, as published: the window runs from PSI
    seconds before a sentence starts to when that sentence ENDS -- unclipped, unlike the
    Frontiers rule -- and each fixation is assigned to the FIRST sentence in report order whose
    window covers it. The finding's weight is the duration of the fixations that partition gave
    to its mentioning sentence(s). Label-blind by construction: two findings named in one
    sentence get identical weights, which is a property of their rule, not of this port."""
    t0 = fix[:, 2] - fix[:, 3] / 2; t1 = fix[:, 2] + fix[:, 3] / 2
    owner = np.full(len(fix), -1)
    for j, (s, e) in enumerate(sents):
        take = (owner < 0) & (t0 < e) & (t1 > s - psi)
        owner[take] = j
    mine = [j for j, (s, _e) in enumerate(sents)
            if any(abs(s - ms) < 1e-6 for _g, ms, *_ in ment)]
    sel = np.isin(owner, mine) if mine else np.zeros(len(fix), bool)
    if not sel.any():
        sel[:] = True                      # same ungated fallback every other method takes
    w = np.zeros(len(fix), np.float32)
    w[sel] = fix[sel, 3]
    return w


def b1_weights(fix, mentions, mask=None, right="sent"):
    """Per-fixation weight behind the gated methods, as one source of truth: the temporal gate
    decides membership, duration weights it as B1 does, and the label prior (word-modulated or
    not) modulates it. Zero outside the gate. `mask=None` reproduces b1_at's weighting, so the
    heatmap and the fixation-mass diagnostic always read the same selection."""
    t0 = fix[:, 2] - fix[:, 3] / 2; t1 = fix[:, 2] + fix[:, 3] / 2
    sel = np.zeros(len(fix), bool)
    if mentions:
        for g, s, e, *rest in mentions:
            # right="mention" is the published Frontiers edge (end of the last mention inside
            # the sentence); "sent" is the sentence end our own B1 uses, which is the later
            # ISBI convention. The 4th field is absent in caches built before this was added.
            hi = rest[0] if (right == "mention" and rest) else e
            sel |= (t0 < hi) & (t1 > g)
    if not sel.any():
        sel[:] = True                      # ungated fallback, exactly as b1_at does
    w = np.zeros(len(fix), np.float32)
    w[sel] = fix[sel, 3]
    if mask is not None:
        m = mask[np.clip((fix[:, 1] * mask.shape[0]).astype(int), 0, mask.shape[0] - 1),
                 np.clip((fix[:, 0] * mask.shape[1]).astype(int), 0, mask.shape[1] - 1)]
        if (w * m).sum() > 0:
            w = w * m                      # else the prior has no support: keep B1's weights
    return w


def _wsplat(fix, w, sigma):
    """Splat a per-fixation weight vector at the fixations' real coordinates."""
    nz = w > 0
    return splat(fix[nz, 0], fix[nz, 1], w[nz], sigma)


def mass_inside(fix, ells, w):
    """Share of a method's per-fixation weight that lands on fixations inside the annotation.
    No threshold and no bandwidth, so nothing here is tuned. NOT a superiority metric: the
    shipped model is trained on an objective directly related to this quantity, while the
    rule-based methods are not. It characterizes assignment behaviour, nothing more."""
    tot = float(np.sum(w))
    if tot <= 0 or not len(ells):
        return np.nan
    return float(np.sum(w[inside_ellipses(fix, ells)]) / tot)


def b1_scan_heat(fix, mentions, mask, sigma):
    """All three signals combined by rule rather than learned: B1's temporal gate picks
    which fixations, the word-modulated label prior weights them, duration weights them as
    B1 already does. This is the label+timing+spatial cell that no published method
    occupies -- the reference that separates *learning* the association from merely
    *having* the three inputs."""
    w = b1_weights(fix, mentions, mask)
    nz = w > 0
    return splat(fix[nz, 0], fix[nz, 1], w[nz], sigma)


def word_dirs(tr):
    """For each spatial word pair, which half of the image it points to, measured on the
    train split's ellipse centres. -> list of ((bitA, bitB), coord_grid, A_is_low_half)."""
    out = []
    for (A, B), g, ax in (((LEFT, RIGHT), GX, 0), ((UPPER, LOWER), GY, 1)):
        c = [np.mean([np.mean([(e[ax] + e[ax + 2]) / 2 for e in it[2]])
                      for it in tr if it[4][a] == 1 and it[4][b] == 0 and it[2]] or [np.nan])
             for a, b in ((A, B), (B, A))]
        if np.isfinite(c).all():
            out.append(((A, B), g, c[0] < c[1], c[0], c[1]))
    return out


def modulate(mask, wf, dirs):
    """Restrict the prior to the half-plane the mention's spatial word names. A word only
    restricts when its bit is set and its opposite is not, so 'bilateral left and right'
    correctly restricts nothing."""
    keep = np.ones(mask.shape, bool)
    for (A, B), g, a_low, _, _ in dirs:
        if wf[A] == 1 and wf[B] == 0:
            keep &= (g < 0.5) if a_low else (g > 0.5)
        elif wf[B] == 1 and wf[A] == 0:
            keep &= (g > 0.5) if a_low else (g < 0.5)
    m = mask * keep
    return m if m.max() > 0 else mask


def _iou_pre(up, gt, t):
    """iou() with the upscale hoisted out -- it depends on sigma, not on the threshold."""
    pred = up >= t
    inter = (pred & gt).sum(); uni = pred.sum() + gt.sum() - inter
    return inter / uni if uni > 0 else np.nan


def paired(a, b):
    """Mean paired difference + paired Wilcoxon over the paired-complete subset."""
    ok = np.isfinite(a) & np.isfinite(b)
    d = a[ok] - b[ok]
    if d.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(d)), (1.0 if not np.any(d) else float(wilcoxon(a[ok], b[ok])[1]))


def run(cache, epochs, seed, split_seed):
    import torch
    recs, labels = torch.load(cache, weights_only=False)
    li = {L: i for i, L in enumerate(labels)}
    part = split(recs, split_seed)

    def insts(p):
        return [(r["fix"], d["mentions"], d["ellipses"], li[d["label"]],
                 word_feat(d.get("mtext", [])), d["label"], r.get("sents", []))
                for r in recs if part(r) == p for d in r["labels"]]
    tr, va, te = insts("train"), insts("val"), insts("test")
    print(f"train/val/test = {len(tr)}/{len(va)}/{len(te)}", flush=True)

    # Training-derived quantities use the cohort the task is defined on: instances with a
    # resolved positive mention.
    tr_prior = [it for it in tr if len(it[1])]
    print(f"prior estimated on {len(tr_prior)} of {len(tr)} training instances", flush=True)
    prior = label_prior((it[2], it[5]) for it in tr_prior)
    dirs = word_dirs(tr_prior)
    for (A, _B), _g, a_low, ca, cb in dirs:
        nm = "left/right (x)" if A == LEFT else "upper/lower (y)"
        print(f"  measured direction {nm}: train ellipse centres {ca:.3f} vs {cb:.3f} "
              f"-> first word = {'low' if a_low else 'high'} half")

    mod = {}

    def masks(items, use_words):
        out = []
        for it in items:
            m = prior.get(it[5])
            if m is not None and use_words:
                k = (it[5], tuple(it[4][[LEFT, RIGHT, UPPER, LOWER]]))
                out.append(mod.setdefault(k, modulate(m, it[4], dirs)))
            else:
                out.append(m)
        return out

    va_pm, te_pm = masks(va, False), masks(te, False)
    va_wm, te_wm = masks(va, True), masks(te, True)
    n_nomask = sum(m is None for m in te_pm)
    n_fallback = sum(1 for it, m in zip(te, te_pm)
                     if m is None or scan_heat(it[0], m, 1.0).max() == 0)
    print(f"  test instances with no train prior for their label: {n_nomask}; "
          f"falling back to the ungated map: {n_fallback}/{len(te)}")

    gt = {"va": [raster(it[2]) for it in va], "te": [raster(it[2]) for it in te]}

    def tune(heat, sigmas=TUNE_SIGMAS, with_score=False):
        """The same joint search every method gets, selected on validation IoU."""
        best = (sigmas[0], TS[0], -1.0)
        for sg in sigmas:
            ups = []
            for i, it in enumerate(va):
                h = heat(i, it, sg)
                ups.append(zoom(h, EVAL_RES / h.shape[0], order=0))
            for t in TS:
                sc = np.nanmean([_iou_pre(u, g, t) for u, g in zip(ups, gt["va"])])
                if sc > best[2]:
                    best = (sg, t, sc)
            del ups
        if best[0] in (sigmas[0], sigmas[-1]) or best[1] in (TS[0], TS[-1]):
            print(f"  WARNING: tuned ({best[0]}, {best[1]:.3f}) on a grid boundary")
        return best if with_score else best[:2]

    def score(heat, sg, th):
        hs = [heat(i, it, sg) for i, it in enumerate(te)]
        return (np.array([iou(h, g, th) for h, g in zip(hs, gt["te"])]),
                np.array([pointing(h, g) for h, g in zip(hs, gt["te"])]))

    # The shipped model, trained here so the paired comparison is against this exact
    # split/seed rather than against a number copied from another log.
    print(f"training FINAL (fusion={FUSION}, pos_mode={POS_MODE}, seed={seed})...", flush=True)
    net = train_model(tr, labels, use_position=True, epochs=epochs, use_text=True,
                      fusion=FUSION, pos_mode=POS_MODE, seed=seed)

    def fin_raw(items, shuf, rng=None):
        return [predict_raw(net, it[0], it[1], it[3], use_position=True, use_text=True,
                            wf=it[4], pos_mode=POS_MODE, shuffle_pos=shuf, rng=rng)
                for it in items]
    va_fn, te_fn = fin_raw(va, False), fin_raw(te, False)
    # rng(2) fresh and consumed only here, over te in order -- same as evaluate,
    # so this reproduces that script's shuffled row rather than a differently-permuted one.
    te_sh = fin_raw(te, True, np.random.default_rng(2))

    res = {}
    for nm, hv, ht, sg_grid in (
            ("PRIORSCAN", lambda i, it, sg: scan_heat(it[0], va_pm[i], sg),
             lambda i, it, sg: scan_heat(it[0], te_pm[i], sg), TUNE_SIGMAS),
            ("PRIORSCAN_W", lambda i, it, sg: scan_heat(it[0], va_wm[i], sg),
             lambda i, it, sg: scan_heat(it[0], te_wm[i], sg), TUNE_SIGMAS),
            # sigma 0 is in the anatomical prior's grid because prior_and_swap gives
            # it that grid; the prior is already smooth and may not want blurring at all.
            ("PRIOR", lambda i, it, sg: prior_heat(va_pm[i], sg),
             lambda i, it, sg: prior_heat(te_pm[i], sg), [0.0] + list(TUNE_SIGMAS)),
            # the prior carrying the spatial words but NOT restricted to the scanpath:
            # isolates what gaze coverage adds to the strongest training-free reference.
            ("PRIOR_W", lambda i, it, sg: prior_heat(va_wm[i], sg),
             lambda i, it, sg: prior_heat(te_wm[i], sg), [0.0] + list(TUNE_SIGMAS)),
            # B1P is B1PW without the spatial words: the two differ by exactly the word
            # modulation, so B1P -> B1PW isolates what the mention's spatial language adds
            # inside the gated path. Without it, B1 -> B1PW confounds language with anatomy.
            ("B1P", lambda i, it, sg: b1_scan_heat(it[0], it[1], va_pm[i], sg),
             lambda i, it, sg: b1_scan_heat(it[0], it[1], te_pm[i], sg), TUNE_SIGMAS),
            ("B1PW", lambda i, it, sg: b1_scan_heat(it[0], it[1], va_wm[i], sg),
             lambda i, it, sg: b1_scan_heat(it[0], it[1], te_wm[i], sg), TUNE_SIGMAS),
            # the two published rules, transferred onto this evaluator. Everything downstream
            # of the window -- the held-out threshold, the bandwidth search, the pointing game
            # -- is ours, and neither original computes it; see REFRAME.md 10.4.
            ("B1_FRONT", lambda i, it, sg: _wsplat(it[0], b1_weights(it[0], it[1],
                                                                     right="mention"), sg),
             lambda i, it, sg: _wsplat(it[0], b1_weights(it[0], it[1], right="mention"), sg),
             TUNE_SIGMAS),
            ("B1_ISBI", lambda i, it, sg: _wsplat(it[0], isbi_weights(it[0], it[6], it[1]), sg),
             lambda i, it, sg: _wsplat(it[0], isbi_weights(it[0], it[6], it[1]), sg),
             TUNE_SIGMAS),
            ("B1", lambda i, it, sg: b1_at(it[0], it[1], sg),
             lambda i, it, sg: b1_at(it[0], it[1], sg), TUNE_SIGMAS),
            ("FINAL", lambda i, it, sg: blur_norm(va_fn[i], sg),
             lambda i, it, sg: blur_norm(te_fn[i], sg), TUNE_SIGMAS)):
        print(f"tuning {nm}...", flush=True)
        s, t = tune(hv, sg_grid)
        res[nm] = score(ht, s, t) + (s, t)
        print(f"  {nm:12s} sigma={s} thr={t:.4f}  IoU {np.nanmean(res[nm][0]):.4f}  "
              f"pointing {np.nanmean(res[nm][1]):.4f}", flush=True)
    # The shuffle control reuses FINAL's own (sigma, threshold): same eval settings, only
    # the input differs. That is the established discipline for every control here.
    s_fn, t_fn = res["FINAL"][2], res["FINAL"][3]
    res["SHUF"] = score(lambda i, it, sg: blur_norm(te_sh[i], sg), s_fn, t_fn) + (s_fn, t_fn)
    print(f"  {'SHUF':12s} sigma={s_fn} thr={t_fn:.4f}  IoU {np.nanmean(res['SHUF'][0]):.4f}  "
          f"pointing {np.nanmean(res['SHUF'][1]):.4f}", flush=True)

    # Fixation-level diagnostic: of the weight each rule puts on fixations, how much lands on
    # fixations inside the annotation? No threshold and no bandwidth, so nothing is tuned. This
    # is a characterization of assignment behaviour, NOT a ranking: FINAL is trained on an
    # objective directly related to this quantity and the rule-based methods are not. It exists
    # because "we attribute individual fixations" should be measured at the fixation level
    # somewhere, rather than only asserted about a splatted heatmap.
    fm = {}
    for nm, wf_ in (("B1", lambda i, it: b1_weights(it[0], it[1])),
                    ("B1P", lambda i, it: b1_weights(it[0], it[1], te_pm[i])),
                    ("B1PW", lambda i, it: b1_weights(it[0], it[1], te_wm[i])),
                    ("FINAL", lambda i, it: predict_raw(
                        net, it[0], it[1], it[3], use_position=True, use_text=True,
                        wf=it[4], pos_mode=POS_MODE, return_weights=True))):
        fm[nm] = np.array([mass_inside(it[0], it[2], wf_(i, it))
                           for i, it in enumerate(te)], float)
    print("\nFIXATION-MASS DIAGNOSTIC -- share of per-fixation weight inside the ellipse.")
    print("  Untuned. FINAL trains on a related objective, so read as behaviour, not ranking.")
    for nm in ("B1", "B1P", "B1PW", "FINAL"):
        v = fm[nm]
        print(f"FIXMASS split={split_seed} seed={seed} method={nm} "
              f"mean={np.nanmean(v):.4f} median={np.nanmedian(v):.4f} "
              f"n={int(np.isfinite(v).sum())}", flush=True)

    # B1 and B1PW are untrained, so on a fixed split the sweep returns the same curve for
    # every --seed. The nine-run population is five seeds on split 0 plus five splits at
    # seed 0, so gating on seed 0 runs the sweep exactly once per distinct split and drops
    # four identical repeats. No swept quantity depends on the seed, so nothing is lost.
    if seed == 0:
        # --- B1's temporal offset, given the same validation search every other knob gets ----
        # The cached gate start is max(sent_start - LOOKBACK, prev_sentence_start), so
        # max(sent_start - d, gate_start) reproduces the clipped window exactly for d <= LOOKBACK:
        # where the clip bound, gate_start IS prev_sentence_start; where it did not, gate_start is
        # sent_start - LOOKBACK, which the max() then overrides. Past LOOKBACK the previous
        # sentence's start is not recoverable from the cache, so the grid stops at 1.5 and the
        # printed clip fraction says how much of the window was ever decided by the clip at all.
        def regate(ments, d):
            # keeps any trailing fields (the original's last-mention edge) intact
            return [(max(s - d, g), s, e, *r) for g, s, e, *r in ments]

        clip = [g > s - LOOKBACK + 1e-6 for it in te for g, s, e, *_ in it[1]]
        print(f"\nB1 offset sweep (clip bound on {100 * np.mean(clip):.1f}% of "
              f"{len(clip)} test mentions; grid cannot exceed LOOKBACK={LOOKBACK})", flush=True)
        # B1PW inherits B1's gate, so the sweep has to reach it too. Tuning the offset for the
        # baseline while leaving our own strongest rule-based reference pinned at the shipped
        # 1.5 s would be an asymmetry in our favour -- the class of thing this protocol exists to
        # avoid. It also exposes the residual: a better gate lifts B1PW and leaves FINAL, which
        # has no gate at all, exactly where it was.
        b1o, bwo = [], []
        for d in (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5):
            for nm, hv_, ht_, acc in (
                    ("B1OFF",
                     lambda i, it, sg, d=d: b1_at(it[0], regate(it[1], d), sg),
                     lambda i, it, sg, d=d: b1_at(it[0], regate(it[1], d), sg), b1o),
                    ("B1PWOFF",
                     lambda i, it, sg, d=d: b1_scan_heat(it[0], regate(it[1], d), va_wm[i], sg),
                     lambda i, it, sg, d=d: b1_scan_heat(it[0], regate(it[1], d), te_wm[i], sg),
                     bwo)):
                sg_, th_, v_ = tune(hv_, TUNE_SIGMAS, with_score=True)
                iv, pv = score(ht_, sg_, th_)
                acc.append((v_, d, np.nanmean(iv), np.nanmean(pv), sg_, th_))
                print(f"{nm} split={split_seed} seed={seed} delta={d} val_iou={v_:.4f} "
                      f"sigma={sg_} thr={th_:.4f} iou={np.nanmean(iv):.4f} "
                      f"pg={np.nanmean(pv):.4f}", flush=True)
        for nm, acc in (("B1OFFBEST", b1o), ("B1PWOFFBEST", bwo)):
            bv, bd, bi, bp, bs, bt = max(acc)
            print(f"{nm} split={split_seed} seed={seed} delta={bd} val_iou={bv:.4f} "
                  f"iou={bi:.4f} pg={bp:.4f} sigma={bs} thr={bt:.4f}  "
                  f"(shipped 1.5 -> iou={acc[-1][2]:.4f} pg={acc[-1][3]:.4f})", flush=True)

    order = ("B1", "B1_FRONT", "B1_ISBI", "PRIOR", "PRIOR_W", "PRIORSCAN", "PRIORSCAN_W",
             "B1P", "B1PW", "SHUF", "FINAL")
    pairs = (("PRIORSCAN", "B1"), ("PRIORSCAN", "PRIOR"), ("PRIORSCAN", "SHUF"),
             ("PRIORSCAN", "FINAL"), ("PRIORSCAN_W", "B1"), ("PRIORSCAN_W", "PRIOR"),
             ("PRIORSCAN_W", "SHUF"), ("PRIORSCAN_W", "FINAL"),
             ("PRIORSCAN_W", "PRIORSCAN"),
             ("PRIOR_W", "PRIOR"), ("PRIORSCAN_W", "PRIOR_W"),
             ("B1PW", "B1"), ("B1PW", "PRIORSCAN_W"), ("B1PW", "FINAL"),
             ("B1P", "B1"), ("B1PW", "B1P"), ("B1P", "FINAL"),
             ("B1_FRONT", "B1"), ("B1_ISBI", "B1"), ("B1_FRONT", "B1_ISBI"),
             ("B1_FRONT", "FINAL"), ("B1_ISBI", "FINAL"))

    print(f"\nTEST (n={len(te)})            IoU    pointing")
    for nm in order:
        i_v, p_v, s, t = res[nm]
        print(f"  {nm:14s} {np.nanmean(i_v):8.4f} {np.nanmean(p_v):9.4f}   "
              f"(sigma={s}, thr={t:.4f})")
        print(f"PSCAN split={split_seed} seed={seed} method={nm} "
              f"iou={np.nanmean(i_v):.4f} pg={np.nanmean(p_v):.4f} sigma={s} thr={t:.4f}")

    # The keyword matcher finds no mention for a minority of instances. There B1 has
    # nothing to gate on and falls back to the ungated full-gaze map, while the
    # prior-scanpath baselines never need a mention at all -- so part of any edge over B1
    # is B1's labeler failing, not the mechanism. Split every comparison exactly the way
    # evaluate splits its own B1SUB row: same field, same predicate.
    has_m = np.array([len(it[1]) > 0 for it in te])
    subs = (("all", np.ones(len(te), bool)), ("ment", has_m), ("noment", ~has_m))
    assert int(has_m.sum()) + int((~has_m).sum()) == len(te)
    print(f"\nMENTION SPLIT: matched {has_m.sum()}/{len(te)} ({100*has_m.mean():.1f}%), "
          f"no-mention {(~has_m).sum()}  (sums to {len(te)})")

    for sname, msk in subs:
        n = int(msk.sum())
        print(f"\n=== subset {sname} (n={n}"
              + (", UNDERPOWERED" if n < 30 else "") + f")            IoU    pointing")
        for nm in order:
            print(f"  {nm:14s} {np.nanmean(res[nm][0][msk]):8.4f} "
                  f"{np.nanmean(res[nm][1][msk]):9.4f}")
        print(f"PSSUM split={split_seed} seed={seed} sub={sname} n={n} "
              + " ".join(f"{k.lower()}_iou={np.nanmean(res[k][0][msk]):.4f} "
                         f"{k.lower()}_pg={np.nanmean(res[k][1][msk]):.4f}" for k in order)
              + (f" fallback={n_fallback}" if sname == "all" else ""))
        print(f"  paired, positive = the training-free baseline is better:")
        for a, b in pairs:
            di, pi = paired(res[a][0][msk], res[b][0][msk])
            dp, pp = paired(res[a][1][msk], res[b][1][msk])
            print(f"    {a:12s} vs {b:12s} IoU {di:+.4f} p={pi:.3g} {'*' if pi < 0.05 else ' '}"
                  f"  pointing {dp:+.4f} p={pp:.3g} {'*' if pp < 0.05 else ' '}")
            print(f"PSCMP split={split_seed} seed={seed} sub={sname} n={n} a={a} b={b} "
                  f"d_iou={di:+.4f} p_iou={pi:.3g} d_pg={dp:+.4f} p_pg={pp:.3g}")


def _selfcheck():
    """Three things that would silently corrupt the result if wrong."""
    import torch
    from core import WORD_DIM, align_feats
    from selector import make_net, pos_feat

    # 1. mask lookup orientation. A mask that is 1 only in the TOP-LEFT quadrant must
    #    weight a top-left fixation and not a bottom-left one; [x, y] indexing would pass
    #    a symmetric test and fail this one.
    mask = np.zeros((EVAL_RES, EVAL_RES), np.float32); mask[:EVAL_RES // 2, :EVAL_RES // 2] = 1
    fix = np.array([[0.2, 0.2, 0., 1., 0.], [0.2, 0.8, 1., 1., 0.],
                    [0.8, 0.2, 2., 1., 0.]], np.float32)
    h = scan_heat(fix, mask, 1.0)
    peak = np.unravel_index(np.argmax(h), h.shape)
    assert peak[0] < HEAT_RES / 2 and peak[1] < HEAT_RES / 2, f"mask lookup transposed: {peak}"

    # 2. the weighting must actually select. Half the fixations sit in the prior, half
    #    outside; the weighted map must beat the unweighted one against the prior region.
    rng = np.random.default_rng(0)
    f2 = np.zeros((40, 5), np.float32)
    f2[:20, :2] = rng.uniform(0.05, 0.45, (20, 2)); f2[20:, :2] = rng.uniform(0.55, 0.95, (20, 2))
    f2[:, 3] = 1.0
    g = raster([(0.05, 0.05, 0.45, 0.45)])
    w = iou(scan_heat(f2, mask, 1.5), g, 0.3)
    u = iou(splat(f2[:, 0], f2[:, 1], np.ones(40, np.float32), 1.5), g, 0.3)
    assert w > u, f"prior weighting did not select: {w:.3f} vs ungated {u:.3f}"

    # 3. the additivity the whole framing rests on. Swapping the positions of fixations
    #    i and j must leave s_i + s_j unchanged; in log-softmax that is
    #    (l'_i + l'_j - l_i - l_j) - 2(l'_k - l_k) = 0 for any untouched k.
    f3 = np.concatenate([rng.random((6, 2)), np.sort(rng.uniform(0, 10, (6, 1)), 0),
                         rng.random((6, 2))], 1).astype(np.float32)
    ment = [(1.0, 2.0, 2.5)]
    wf = np.zeros(WORD_DIM, np.float32); wf[LEFT] = 1.0
    i, j, k = 0, 1, 2
    resid = {}
    for fusion in ("crossattn", "concat"):
        torch.manual_seed(0)
        net = make_net(["L"], use_position=True, use_text=True, fusion=fusion,
                       pos_mode=POS_MODE)

        def la(f):
            with torch.no_grad():
                return np.log(net.attn(torch.from_numpy(align_feats(f, ment)), 0,
                                       pos_feat(f[:, :2], POS_MODE), wf).numpy())
        f4 = f3.copy(); f4[[i, j], :2] = f3[[j, i], :2]
        l0, l1 = la(f3), la(f4)
        resid[fusion] = abs((l1[i] + l1[j] - l0[i] - l0[j]) - 2 * (l1[k] - l0[k]))
    assert resid["crossattn"] < 1e-4, f"shipped scorer is NOT additive: {resid['crossattn']:.2e}"
    assert resid["concat"] > 1e-3, f"concat additive too ({resid['concat']:.2e}): test vacuous"

    # B1PW composes B1's gate with the word-modulated prior: check it stays faithful to B1
    # where the prior is uninformative, and that each component still bites.
    def cx_of(h):                       # splat writes m[y, x]; sum rows -> per-x mass
        c = h.sum(0)
        return (c * np.arange(len(c))).sum() / c.sum() / len(c)
    fb = np.array([[.20, .5, 5.5, .3, .1], [.30, .5, 6.0, .3, .1],     # gated, left half
                   [.70, .5, 6.2, .3, .1], [.80, .5, 6.5, .3, .1],     # gated, right half
                   [.95, .5, 1.0, .3, .1], [.97, .5, 15., .3, .1]])    # outside the gate
    mb = [(5.0, 5.0, 7.0)]
    ones = np.ones((64, 64), np.float32)
    for mask, why in ((ones, "uniform prior"), (np.zeros((64, 64), np.float32), "empty prior")):
        assert np.allclose(b1_scan_heat(fb, mb, mask, 2.0), b1_at(fb, mb, 2.0), atol=1e-6), \
            f"B1PW under a {why} must reduce to B1"
    assert np.allclose(b1_scan_heat(fb, [], ones, 2.0), b1_at(fb, [], 2.0), atol=1e-6), \
        "B1PW without a mention must take B1's own ungated fallback"
    left = np.zeros((64, 64), np.float32); left[:, :32] = 1.0
    assert cx_of(b1_scan_heat(fb, mb, left, 1.0)) < cx_of(b1_at(fb, mb, 1.0)) - 0.15, \
        "the spatial word did not move mass into the half it names"
    right = np.zeros((64, 64), np.float32); right[:, 32:] = 1.0
    assert cx_of(b1_scan_heat(fb, mb, right, 1.0)) < 0.90, \
        "fixations outside the temporal gate leaked in where the mask was 1"
    print(f"self-check ok: weighted {w:.3f} > ungated {u:.3f}; additivity residual "
          f"crossattn {resid['crossattn']:.2e} vs concat {resid['concat']:.2e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="align.pt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-seed", type=int, default=0, dest="split_seed")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    _selfcheck()
    if not a.selfcheck:
        run(a.cache, a.epochs, a.seed, a.split_seed)

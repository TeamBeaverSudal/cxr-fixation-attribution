"""Final candidate validation: cross-attention + Fourier position + text, trained
and evaluated as ONE model, then BOTH causal controls (position-shuffle,
spatial-word masking) re-run on that EXACT model -- not on an intermediate
config. Per the agreed reasoning: cross-attention makes word the QUERY that
directly selects spatial keys, so position-shuffle alone can no longer rule out
a learned word->canonical-position shortcut; masking is promoted from optional
to REQUIRED alongside shuffle for this architecture.

Accept the core causal claim only if all three hold on THIS exact model:
  1. beats Temporal-only (real improvement)
  2. collapses under position-shuffle (genuine per-instance position use)
  3. degrades under spatial-word masking (genuine spatial-content use, not a
     word->canonical-position lookup memorized independent of correspondence)

Usage (reuses align.pt; no new extraction):
  python evaluate.py --cache align.pt --epochs 40
  python evaluate.py                            # structural self-check
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from core import (iou, pointing, raster, word_feat, b1_at, WORD_TERMS,
                         TUNE_SIGMAS, tune_thresholds)
from selector import train_model, predict_raw, blur_norm
from masking_control import mask_spatial, SPATIAL_IDX

# Cross-attention with FOURIER position, not raw. The 2x2 ablation -- once completed and
# swept over seeds -- put crossattn+raw behind crossattn+fourier on both metrics in 5/5
# seeds at an identical tuned bandwidth (IoU +0.019, pointing +0.079). raw was originally
# adopted from a one-factor comparison under concat fusion, where a two-layer MLP can
# build a nonlinear function of raw coordinates; cross-attention's key projection is a
# single Linear and cannot, so it needs the encoding that concat did not.
FUSION, POS_MODE = "crossattn", "fourier"


def run(cache, epochs, seed=0, external=None, split_seed=0, train_frac=None, chain_seed=0):
    import torch
    recs, labels = torch.load(cache, weights_only=False)
    li = {L: i for i, L in enumerate(labels)}
    subs = np.array([r["subject"] for r in recs]); uniq = np.array(sorted(set(subs)))
    # split_seed changes WHICH patients are held out; seed changes only training.
    # Varying them separately distinguishes 'robust to training noise' from
    # 'robust to this particular patient sample' -- different questions.
    rng = np.random.default_rng(split_seed); rng.shuffle(uniq); n = len(uniq)
    val_s = set(uniq[:int(n * .15)]); test_s = set(uniq[int(n * .15):int(n * .45)])
    part = lambda r: "val" if r["subject"] in val_s else "test" if r["subject"] in test_s else "train"

    def insts(p):
        out = []
        for r in recs:
            if part(r) != p:
                continue
            for d in r["labels"]:
                out.append((r["fix"], d["mentions"], d["ellipses"], li[d["label"]],
                            word_feat(d.get("mtext", [])), (r["rid"], d["label"])))
        return out
    tr, va, te = insts("train"), insts("val"), insts("test")
    # The selector is fit on the cohort the task is defined over: instances with a resolved
    # positive mention. Evaluation keeps every annotated instance so that linker coverage and
    # the fallback behaviour stay visible.
    tr_fit = [it for it in tr if len(it[1])]
    if train_frac is not None:
        # Training-set-size sensitivity. Each fraction is a prefix of one shuffled patient
        # order, so within a chain the subsamples nest: 10% subset of 25% subset of 50%. The
        # unit is the patient, not the instance, so no patient straddles the boundary.
        sub_of = {r["rid"]: r["subject"] for r in recs}
        # eligible = patients contributing a mention-resolved training instance, the set the
        # fractions are defined over; the training split holds more patients than that.
        tp = np.array(sorted({sub_of[it[5][0]] for it in tr_fit}))
        np.random.default_rng(chain_seed).shuffle(tp)
        keep = set(tp[:max(1, int(round(train_frac * len(tp))))])
        tr_fit = [it for it in tr_fit if sub_of[it[5][0]] in keep]
        print(f"train_frac={train_frac} chain={chain_seed}: {len(keep)} of {len(tp)} "
              f"eligible training patients, {len(tr_fit)} instances", flush=True)
    print(f"fitting on {len(tr_fit)} of {len(tr)} training instances", flush=True)
    print(f"instances train/val/test = {len(tr)}/{len(va)}/{len(te)}, {len(labels)} labels "
          f"(split_seed={split_seed}, train seed={seed})", flush=True)

    # NOTE: the patient split (default_rng(0)) and the shuffle-control permutation
    # (default_rng(2)) stay FIXED across --seed values. Only training varies, so every
    # seed is scored on the identical test set and the identical perturbation -- that's
    # what makes cross-seed comparison of the controls meaningful.
    print(f"training Temporal-only model (seed={seed})...", flush=True)
    # fusion=FUSION so the ablation differs from the full model in exactly one thing:
    # the inputs, and takes the shipped fusion explicitly rather than a default, since
    # Ours - Temporal-only mixed the input change with a scorer change and could not be
    # read as the contribution of position and language. Measured across nine runs, the
    # scorer alone is worth +0.0025 IoU and +0.0098 pointing here, significant in 0/9 --
    # small, but the comparison is now clean rather than merely nearly clean.
    net_r1 = train_model(tr_fit, labels, use_position=False, epochs=epochs, use_text=False,
                         fusion=FUSION, seed=seed)
    print(f"training FINAL candidate (fusion={FUSION}, pos_mode={POS_MODE}, seed={seed})...", flush=True)
    net_final = train_model(tr_fit, labels, use_position=True, epochs=epochs, use_text=True,
                            fusion=FUSION, pos_mode=POS_MODE, seed=seed)

    shuf_rng = np.random.default_rng(2)
    # A second fixed stream, so permuting the temporal rows does not consume the position
    # control's draws and shift a comparison that is supposed to be identical across seeds.
    tshuf_rng = np.random.default_rng(3)

    # The reduced query keeps the four directional indicators and drops the six that qualify
    # extent, size or character. WORD_TERMS order: left, right, bilateral, upper, lower,
    # middle, retrocardiac, small, large, diffuse.
    Q4 = (0, 1, 3, 4)

    def keep4(wf):
        v = np.zeros_like(wf)
        v[list(Q4)] = wf[list(Q4)]
        return v

    print(f"training four-indicator candidate (query = {[WORD_TERMS[i] for i in Q4]})...",
          flush=True)
    net_q4 = train_model([(f, m, e, l, keep4(wf), k) for f, m, e, l, wf, k in tr_fit],
                         labels, use_position=True, epochs=epochs, use_text=True,
                         fusion=FUSION, pos_mode=POS_MODE, seed=seed)

    def raw_for(method, item):
        f, m, e, l, wf, _key = item
        if method == "result1":
            return predict_raw(net_r1, f, m, l, use_position=False)
        if method == "final":
            return predict_raw(net_final, f, m, l, use_position=True, use_text=True,
                               wf=wf, pos_mode=POS_MODE)
        if method == "final_posshuf":
            return predict_raw(net_final, f, m, l, use_position=True, use_text=True,
                               wf=wf, pos_mode=POS_MODE, shuffle_pos=True, rng=shuf_rng)
        if method == "final_wordmask":
            return predict_raw(net_final, f, m, l, use_position=True, use_text=True,
                               wf=mask_spatial(wf), pos_mode=POS_MODE)
        if method == "final_tempshuf":
            return predict_raw(net_final, f, m, l, use_position=True, use_text=True,
                               wf=wf, pos_mode=POS_MODE, shuffle_temporal=True, rng=tshuf_rng)
        if method == "q4":
            return predict_raw(net_q4, f, m, l, use_position=True, use_text=True,
                               wf=keep4(wf), pos_mode=POS_MODE)

    def cache_raw(items, method):
        return [(raw_for(method, it), raster(it[2])) for it in items]

    def metric_at(cached, t, sigma, metric):
        out = []
        for raw, gt in cached:
            hm = blur_norm(raw, sigma)
            out.append(iou(hm, gt, t) if metric == "iou" else pointing(hm, gt))
        return np.array(out)

    ts = tune_thresholds()
    sigmas = TUNE_SIGMAS

    def _upscaled(cached, sigma):
        from scipy.ndimage import zoom
        from core import EVAL_RES
        return [(zoom(blur_norm(raw, sigma), EVAL_RES / raw.shape[0], order=0), gt)
                for raw, gt in cached]

    def _iou_pre(up, gt, t):
        pred = up >= t
        inter = (pred & gt).sum(); uni = pred.sum() + gt.sum() - inter
        return inter / uni if uni > 0 else np.nan

    def _search(cached):
        """Blur and upsample depend only on sigma, not on the threshold, but the previous
        form redid both for every threshold -- 23x redundant work, which is precisely why
        the grids were kept small enough to clip the optimum. Doing them once per sigma
        makes a grid wide and fine enough to be correct affordable, and holds only one
        sigma's arrays at a time instead of all of them."""
        best = (sigmas[0], ts[0], -1.0)
        for sg in sigmas:
            ups = _upscaled(cached, sg)
            for t in ts:
                sc = np.nanmean([_iou_pre(u, gt, t) for u, gt in ups])
                if sc > best[2]:
                    best = (sg, t, sc)
            del ups
        return best

    def tune(va_cached):
        best = _search(va_cached)
        if best[0] in (sigmas[0], sigmas[-1]) or best[1] in (ts[0], ts[-1]):
            print(f"  WARNING: tuned (sigma={best[0]}, thr={best[1]:.3f}) sits on a grid "
                  f"boundary -- widen the grid; this method is being under-scored.")
        return best[0], best[1]

    print("caching + tuning...", flush=True)
    va_r1, te_r1 = cache_raw(va, "result1"), cache_raw(te, "result1")
    va_fn, te_fn = cache_raw(va, "final"), cache_raw(te, "final")
    te_shuf = cache_raw(te, "final_posshuf")
    te_mask = cache_raw(te, "final_wordmask")
    te_tshuf = cache_raw(te, "final_tempshuf")
    va_q4, te_q4 = cache_raw(va, "q4"), cache_raw(te, "q4")

    # B1 (Lanfredi's fixed-1.5s gate) is scored HERE, on the same test instances, with
    # the SAME (sigma, threshold) search the learned models get, rather than pinned
    # to sigma=1.5 while our models searched seven values; granting it the identical
    # search raises it from IoU 0.2082 to 0.2650 (p=1e-69) and halves our margin over the
    # published baseline. Any tuning budget we give ourselves must be given to the
    # baseline too, or the comparison measures our search, not our method.
    best_b1 = (sigmas[0], ts[0], -1.0)
    for sg in sigmas:
        vb = [(b1_at(f, m, sg), raster(e)) for f, m, e, l, wf, _k in va]
        for t in ts:
            sc = np.nanmean([iou(hm, gt, t) for hm, gt in vb])
            if sc > best_b1[2]:
                best_b1 = (sg, t, sc)
        del vb
    sb1, tb1 = best_b1[0], best_b1[1]
    if sb1 in (sigmas[0], sigmas[-1]) or tb1 in (ts[0], ts[-1]):
        print(f"  WARNING: B1 tuned (sigma={sb1}, thr={tb1:.3f}) sits on a grid boundary.")
    te_b1 = [(b1_at(f, m, sb1), raster(e)) for f, m, e, l, wf, _k in te]
    b1v = np.array([iou(hm, gt, tb1) for hm, gt in te_b1])
    pb1 = np.array([pointing(hm, gt) for hm, gt in te_b1])

    s_r1, t_r1 = tune(va_r1)
    s_fn, t_fn = tune(va_fn)
    print(f"val-tuned (sigma,thr): Temporal-only={s_r1,t_r1}  final={s_fn,t_fn}  "
          f"B1={sb1,tb1}")

    r1v = metric_at(te_r1, t_r1, s_r1, "iou"); pr1 = metric_at(te_r1, t_r1, s_r1, "pg")
    fnv = metric_at(te_fn, t_fn, s_fn, "iou"); pfn = metric_at(te_fn, t_fn, s_fn, "pg")
    # controls reuse the FINAL model's own tuned (sigma,threshold) -- same eval
    # settings, only the input differs, per the established discipline.
    shv = metric_at(te_shuf, t_fn, s_fn, "iou"); psh = metric_at(te_shuf, t_fn, s_fn, "pg")
    mkv = metric_at(te_mask, t_fn, s_fn, "iou"); pmk = metric_at(te_mask, t_fn, s_fn, "pg")
    tsv = metric_at(te_tshuf, t_fn, s_fn, "iou"); pts = metric_at(te_tshuf, t_fn, s_fn, "pg")
    # the reduced query is a trained model, not a perturbation, so it gets its own search
    s_q4, t_q4 = tune(va_q4)
    q4v = metric_at(te_q4, t_q4, s_q4, "iou"); pq4 = metric_at(te_q4, t_q4, s_q4, "pg")
    print(f"val-tuned four-indicator (sigma,thr)={s_q4, t_q4}")

    # ---- record substitution -------------------------------------------------------------
    # Does the target's own gaze-mention record carry information that a coarse finding and
    # directional match cannot supply? Each test instance is re-scored on other patients'
    # records that match it exactly on finding and on the mention indicators. If the match
    # were sufficient, substitution would cost nothing.
    import hashlib

    sub_of = {r["rid"]: r["subject"] for r in recs}

    def group_key(it):
        # Matched on finding and on the location indicators the package already names as
        # spatial. The size and character terms describe the finding rather than where it is,
        # so including them splits the groups on something the substitution is not about.
        return (it[3], tuple(it[4][SPATIAL_IDX].tolist()))

    pool = {}
    for it in tr_fit:
        pool.setdefault(group_key(it), []).append(it)
    # A fixed hash, so which donors are used does not depend on dict or file order.
    for v in pool.values():
        v.sort(key=lambda d: hashlib.sha256(str(d[5][0]).encode()).hexdigest())

    def donors(it, K):
        same = pool.get(group_key(it), [])
        out = [d for d in same if sub_of.get(d[5][0]) != sub_of.get(it[5][0])]
        return out[:min(K, len(out))]

    def sub_raw(it, K):
        ds = donors(it, K)
        if not ds:
            return None
        f, m, e, l, wf, _k = it
        # the donor supplies the record -- fixations and mention timing -- while the query
        # stays the target's; they agree by construction on finding and indicators anyway
        maps = [predict_raw(net_final, d[0], d[1], l, use_position=True, use_text=True,
                            wf=wf, pos_mode=POS_MODE) for d in ds]
        return np.mean(maps, 0)

    K = 8   # selected on validation from {1, 2, 4, 8} by pointing accuracy
    va_ex = [it for it in va if len(it[1]) and donors(it, K)]
    te_ex = [it for it in te if len(it[1]) and donors(it, K)]
    print(f"\nrecord substitution: {len(te_ex)}/{len(te)} test and {len(va_ex)}/{len(va)} "
          f"validation instances have an exact-match donor (K={K})", flush=True)

    va_sub = [(sub_raw(it, K), raster(it[2])) for it in va_ex]
    te_sub = [(sub_raw(it, K), raster(it[2])) for it in te_ex]
    s_sub, t_sub = tune(va_sub)
    sub_iou = metric_at(te_sub, t_sub, s_sub, "iou")
    sub_pg = metric_at(te_sub, t_sub, s_sub, "pg")
    # the target row is the shipped selector on the same instances, so the pair differs in
    # the record alone rather than in the cohort as well
    tgt = [(raw_for("final", it), raster(it[2])) for it in te_ex]
    tgt_iou = metric_at(tgt, t_fn, s_fn, "iou")
    tgt_pg = metric_at(tgt, t_fn, s_fn, "pg")
    print(f"SUBST n={len(te_ex)} target={np.nanmean(tgt_pg):.4f}/{np.nanmean(tgt_iou):.4f} "
          f"donors={np.nanmean(sub_pg):.4f}/{np.nanmean(sub_iou):.4f} "
          f"(sigma,thr) target={s_fn, t_fn} donors={s_sub, t_sub}", flush=True)

    def paired_iou(a, b):
        ok = np.isfinite(a) & np.isfinite(b)
        _, p = wilcoxon(a[ok], b[ok]); return np.median(a[ok] - b[ok]), p

    def paired_pg(a, b):
        ok = np.isfinite(a) & np.isfinite(b)
        _, p = wilcoxon(a[ok], b[ok]); return np.mean(a[ok] - b[ok]), p

    def paired_iou_mean(a, b):
        # For the masking comparison the paired IoU MEDIAN is structurally 0 (most
        # instances are near-zero IoU under both conditions), so the median cannot
        # express the effect even when Wilcoxon is highly significant. Report the mean
        # for that cell -- same median-vs-mean distinction already applied to pointing.
        ok = np.isfinite(a) & np.isfinite(b)
        _, p = wilcoxon(a[ok], b[ok]); return np.mean(a[ok] - b[ok]), p

    # Macro (per-label) averages alongside micro (per-instance). Bigolin Lanfredi et al.
    # 2023 report per-label IoU, and REFLACX label frequency is extremely skewed
    # (support devices ~45% of cases, hiatal hernia ~1%), so a per-instance mean is
    # dominated by the common labels and is NOT the quantity the published number refers
    # to. Reporting both keeps our headline comparable to prior art and is the fairer
    # aggregation for imbalanced multi-label data.
    labs_of = np.array([k[1] for k in (it[5] for it in te)])

    def macro(v, min_n=1):
        """Unweighted mean over labels. min_n guards the degenerate case: this test split
        has labels with n = 1, 3, 7 and 8, and an unrestricted macro average gives a
        single-instance label the same weight as one with 219. Report both."""
        per = [np.nanmean(v[labs_of == L]) for L in np.unique(labs_of)
               if (labs_of == L).sum() >= min_n and np.isfinite(v[labs_of == L]).any()]
        return float(np.mean(per)) if per else float("nan")

    nlab = len(np.unique(labs_of))
    nlab20 = sum(1 for L in np.unique(labs_of) if (labs_of == L).sum() >= 20)
    print(f"\nPER-LABEL (macro) -- comparable to Lanfredi's per-label IoU. "
          f"all {nlab} labels, and the {nlab20} with n>=20:")
    for nm, iv, pv in (("B1", b1v, pb1), ("Temporal-only", r1v, pr1), ("FINAL", fnv, pfn),
                       ("FINAL shuffled", shv, psh), ("FINAL masked", mkv, pmk)):
        print(f"  {nm:16s} IoU {macro(iv):.4f} / {macro(iv,20):.4f}   "
              f"pointing {macro(pv):.4f} / {macro(pv,20):.4f}")

    print(f"\n  per-label detail (FINAL):")
    for L in np.unique(labs_of):
        m = labs_of == L
        if not np.isfinite(fnv[m]).any():
            continue
        print(f"PERLAB split={split_seed} seed={seed} label={L.replace(' ', '_')} "
              f"n={m.sum()} fn_iou={np.nanmean(fnv[m]):.4f} fn_pg={np.nanmean(pfn[m]):.4f} "
              f"b1_iou={np.nanmean(b1v[m]):.4f} b1_pg={np.nanmean(pb1[m]):.4f}")
        print(f"    {L[:38]:38s} n={m.sum():4d}  IoU {np.nanmean(fnv[m]):.4f}  "
              f"pg {np.nanmean(pfn[m]):.4f}  (B1 {np.nanmean(b1v[m]):.4f}/"
              f"{np.nanmean(pb1[m]):.4f})")

    print(f"\nFINAL MODEL VALIDATION (n={len(te)}):        IoU      pointing-game")
    print(f"  B1 fixed-window (Lanfredi 2023)  {np.nanmean(b1v):.4f}   {np.nanmean(pb1):.4f}")
    print(f"  Temporal-only                    {np.nanmean(r1v):.4f}   {np.nanmean(pr1):.4f}")
    print(f"  FINAL ({FUSION}+{POS_MODE}+gaze+text){'':<3}{np.nanmean(fnv):.4f}   {np.nanmean(pfn):.4f}")
    print(f"    FINAL position-shuffled        {np.nanmean(shv):.4f}   {np.nanmean(psh):.4f}  (control 1)")
    print(f"    FINAL spatial-word-masked      {np.nanmean(mkv):.4f}   {np.nanmean(pmk):.4f}  (control 2)")

    d1, p1 = paired_iou(fnv, r1v); dp1, pp1 = paired_pg(pfn, pr1)
    d2, p2 = paired_iou(fnv, shv); dp2, pp2 = paired_pg(pfn, psh)
    d3, p3 = paired_iou(fnv, mkv); dp3, pp3 = paired_pg(pfn, pmk)

    print(f"\n1) beats Temporal-only?        IoU Δmed ={d1:+.4f} p={p1:.3g} {'*' if p1 < 0.05 else ' '}  |  "
          f"pointing Δmean={dp1:+.4f} p={pp1:.3g} {'*' if pp1 < 0.05 else ''}")
    print(f"2) collapses on shuffle?  IoU Δmed ={d2:+.4f} p={p2:.3g} {'*' if p2 < 0.05 else ' '}  |  "
          f"pointing Δmean={dp2:+.4f} p={pp2:.3g} {'*' if pp2 < 0.05 else ''}")
    d3m, _ = paired_iou_mean(fnv, mkv)
    print(f"3) degrades on masking?   IoU Δmed ={d3:+.4f} (Δmean={d3m:+.4f}) p={p3:.3g} "
          f"{'*' if p3 < 0.05 else ' '}  |  "
          f"pointing Δmean={dp3:+.4f} p={pp3:.3g} {'*' if pp3 < 0.05 else ''}")

    # Against the PUBLISHED baseline. Deliberately NOT folded into the three
    # acceptance criteria: those test the causal claim (does position+language
    # earn its place), whereas this establishes standing vs prior art. Both are
    # needed in the paper, but conflating them would let a win here paper over a
    # failure there.
    d4, p4 = paired_iou(fnv, b1v); dp4, pp4 = paired_pg(pfn, pb1)
    d5, p5 = paired_iou(r1v, b1v); dp5, pp5 = paired_pg(pr1, pb1)
    # The IoU cells above are paired MEDIANS; pointing is a paired MEAN. Reporting one
    # as the other -- or comparing a median here against the mean used for the
    # mention-matched subset below -- reverses which comparison looks larger. Emit both
    # so the paper can state a like-for-like number and say which statistic it is.
    d1m, _ = paired_iou_mean(fnv, r1v)
    d2m, _ = paired_iou_mean(fnv, shv)
    d4m, _ = paired_iou_mean(fnv, b1v)
    d5m, _ = paired_iou_mean(r1v, b1v)
    # Split the B1 comparison by whether the keyword matcher found a mention at all.
    # Where it did not, B1's gate has nothing to gate on and it falls back to the full
    # gaze heatmap -- i.e. to roughly B0 -- while our model still has position and text.
    # Any margin concentrated in that subset comes from the label-mapping step, not from
    # the alignment mechanism, and would not survive a stronger labeler (Bigolin Lanfredi
    # et al. use a modified CheXpert labeler where we use keyword matching).
    has_m = np.array([len(it[1]) > 0 for it in te])
    print(f"\nB1 FIDELITY CHECK -- margin split by whether a mention was matched:")
    print(f"  instances with a matched mention: {has_m.sum()}/{len(has_m)} "
          f"({100*has_m.mean():.1f}%)")
    for nm, m in (("with mention", has_m), ("NO mention (B1 falls back)", ~has_m)):
        if m.sum() < 10:
            print(f"  {nm}: n={m.sum()}, too few to test"); continue
        di, pi = paired_iou_mean(fnv[m], b1v[m])
        dp, pp = paired_pg(pfn[m], pb1[m])
        print(f"  {nm:28s} n={m.sum():4d}  B1 {np.nanmean(b1v[m]):.4f}/{np.nanmean(pb1[m]):.4f}"
              f"  FINAL {np.nanmean(fnv[m]):.4f}/{np.nanmean(pfn[m]):.4f}"
              f"  ->  IoU Δ{di:+.4f} p={pi:.3g}  pg Δ{dp:+.4f} p={pp:.3g}")

    # Greppable so the mention-matched subset -- the paper's PRIMARY B1 comparison -- can be
    # aggregated over a sweep instead of resting on one run.
    if has_m.sum() >= 10:
        _di, _pi = paired_iou_mean(fnv[has_m], b1v[has_m])
        _dp, _pp = paired_pg(pfn[has_m], pb1[has_m])
        print(f"B1SUB split={split_seed} seed={seed} n={int(has_m.sum())} "
              f"frac={has_m.mean():.4f} b1_iou={np.nanmean(b1v[has_m]):.4f} "
              f"b1_pg={np.nanmean(pb1[has_m]):.4f} fn_iou={np.nanmean(fnv[has_m]):.4f} "
              f"fn_pg={np.nanmean(pfn[has_m]):.4f} d_iou={_di:+.4f} p_iou={_pi:.3g} "
              f"d_pg={_dp:+.4f} p_pg={_pp:.3g}")

    print(f"\nvs published fixed-window baseline B1 (prior art, not an acceptance criterion):")
    print(f"  FINAL vs B1            IoU Δmed ={d4:+.4f} p={p4:.3g} {'*' if p4 < 0.05 else ' '}  |  "
          f"pointing Δmean={dp4:+.4f} p={pp4:.3g} {'*' if pp4 < 0.05 else ''}")
    print(f"  Temporal-only vs B1    IoU Δmed ={d5:+.4f} p={p5:.3g} {'*' if p5 < 0.05 else ' '}  |  "
          f"pointing Δmean={dp5:+.4f} p={pp5:.3g} {'*' if pp5 < 0.05 else ''}")

    c1 = (d1 > 0 and p1 < 0.05) or (dp1 > 0 and pp1 < 0.05)
    c2 = (d2 > 0 and p2 < 0.05) and (dp2 > 0 and pp2 < 0.05)
    # c3 gates on the IoU MEAN, not the median. For this comparison specifically the
    # median is structurally 0 -- most instances sit at near-zero IoU under both
    # conditions -- so the median cannot register the effect no matter how large it is.
    # Gating on it discarded evidence that is significant in 10/10 runs (5 training
    # seeds x 5 patient splits, p <= 6e-20). This is the same median-vs-mean correction
    # the binary pointing metric already uses.
    c3 = (d3m > 0 and p3 < 0.05) or (dp3 > 0 and pp3 < 0.05)
    print(f"\n--- VERDICT: (1) beats Temporal-only = {c1}, (2) collapses on shuffle = {c2}, "
          f"(3) degrades on masking = {c3} ---")
    if c1 and c2 and c3:
        print(f"ALL THREE HOLD -> core causal claim substantially supported for the FINAL\n"
              f"  ({FUSION} + {POS_MODE} position) model. Ready to write up.")
    else:
        print("NOT all three hold -> do not finalize this configuration yet; see which\n"
              "  criterion failed and reconsider (e.g. fall back to concat/Fourier, which\n"
              "  already passed all analogous checks in Stage 1/2).")

    if external:
        # Join an external baseline's per-instance scores on (rid, label) and run the
        # SAME paired tests. Joining by key rather than by position matters: the
        # external run may drop instances (missing image, inference failure), so
        # positional zip would silently misalign and compare unrelated pairs.
        import csv as _csv
        ext = {}
        with open(external) as f:
            for row in _csv.DictReader(f):
                # pg_alt is the external model at its OWN pointing optimum, a bandwidth
                # selected on validation pointing rather than on IoU. Carrying it lets the
                # sensitivity analysis be a measurement instead of an argument: the main
                # analysis gives every method one operating point, and this asks whether the
                # conclusion survives when the baseline is allowed its best point per metric.
                ext[(row["rid"], row["label"])] = (float(row["iou"]), float(row["pg"]),
                                                   float(row.get("pg_alt", row["pg"])))
        keys = [it[5] for it in te]
        hit = [i for i, k in enumerate(keys) if k in ext]
        miss = len(keys) - len(hit)
        if len(hit) < 0.5 * len(keys):
            print(f"\nSKIPPING external comparison: only {len(hit)}/{len(keys)} test instances "
                  f"matched. The external CSV was scored on a different split -- re-run the "
                  f"external baseline for split_seed={split_seed} rather than comparing a "
                  f"biased subset.")
            external = None
    if external:
        e_iou = np.array([ext[keys[i]][0] for i in hit])
        e_pg = np.array([ext[keys[i]][1] for i in hit])
        e_pg_alt = np.array([ext[keys[i]][2] for i in hit])
        print(f"\nEXTERNAL BASELINE ({Path(external).name}): matched {len(hit)}/{len(keys)} "
              f"test instances" + (f" ({miss} unmatched, excluded)" if miss else ""))
        print(f"  external                         {np.nanmean(e_iou):.4f}   {np.nanmean(e_pg):.4f}")
        for name, mv, pv in (("FINAL", fnv[hit], pfn[hit]),
                             ("Temporal-only", r1v[hit], pr1[hit])):
            di, pi = paired_iou_mean(mv, e_iou)
            dp, pp = paired_pg(pv, e_pg)
            print(f"  {name:14s} vs external   IoU Δmean={di:+.4f} p={pi:.3g} "
                  f"{'*' if pi < 0.05 else ' '}  |  pointing Δmean={dp:+.4f} p={pp:.3g} "
                  f"{'*' if pp < 0.05 else ''}")
        print("  (positive Δ = ours better. A non-significant p means we have NOT shown\n"
              "   an advantage over this baseline -- report it as a tie, not a win.)")
        # Sensitivity analysis. The main analysis above scores every method at one
        # operating point, chosen on IoU; pointing is therefore not at its own optimum
        # for anyone. Here the baseline alone is allowed the bandwidth that maximises
        # its validation pointing, which is the convention this protocol declines. If
        # the conclusion survives that, the choice of convention does not carry it.
        dpa, ppa = paired_pg(pfn[hit], e_pg_alt)
        print(f"  SENSITIVITY: external at its own pointing optimum "
              f"{np.nanmean(e_pg_alt):.4f} (vs {np.nanmean(e_pg):.4f} at the shared point)")
        print(f"    FINAL vs external, pointing  Δmean={dpa:+.4f} p={ppa:.3g} "
              f"{'*' if ppa < 0.05 else '(n.s.)'}")
        print(f"EXTALT split={split_seed} seed={seed} e_pg={np.nanmean(e_pg):.4f} "
              f"e_pg_alt={np.nanmean(e_pg_alt):.4f} d_pg_alt={dpa:+.4f} p_pg_alt={ppa:.3g}")

    # One greppable line per run, for aggregating a --seed sweep.
    # Every paired statistic the reported tables need, on one greppable line, so the
    # headline comparisons can be aggregated across seeds without re-running.
    print(f"MEANIOU split={split_seed} seed={seed} vsR1={d1m:+.4f} vsSHUF={d2m:+.4f} "
          f"vsB1={d4m:+.4f} r1vsB1={d5m:+.4f}")
    print(f"SUMMARY split={split_seed} seed={seed} "
          f"b1_iou={np.nanmean(b1v):.4f} b1_pg={np.nanmean(pb1):.4f} "
          f"r1_iou={np.nanmean(r1v):.4f} r1_pg={np.nanmean(pr1):.4f} "
          f"fn_iou={np.nanmean(fnv):.4f} fn_pg={np.nanmean(pfn):.4f} "
          f"shuf_iou={np.nanmean(shv):.4f} shuf_pg={np.nanmean(psh):.4f} "
          f"mask_iou={np.nanmean(mkv):.4f} mask_pg={np.nanmean(pmk):.4f} "
          f"vsR1_d_iou={d1:+.4f} vsR1_p_iou={p1:.3g} vsR1_d_pg={dp1:+.4f} vsR1_p_pg={pp1:.3g} "
          f"vsSHUF_d_iou={d2:+.4f} vsSHUF_p_iou={p2:.3g} vsSHUF_d_pg={dp2:+.4f} vsSHUF_p_pg={pp2:.3g} "
          f"mask_d_iou_mean={d3m:+.4f} mask_p_iou={p3:.3g} mask_d_pg={dp3:+.4f} mask_p_pg={pp3:.3g} "
          f"vsB1_d_iou={d4:+.4f} vsB1_p_iou={p4:.3g} vsB1_d_pg={dp4:+.4f} vsB1_p_pg={pp4:.3g} "
          f"r1vsB1_d_iou={d5:+.4f} r1vsB1_p_iou={p5:.3g} r1vsB1_d_pg={dp5:+.4f} r1vsB1_p_pg={pp5:.3g} "
          f"sig_fn={s_fn},{t_fn} sig_r1={s_r1},{t_r1} sig_b1={sb1},{tb1} "
          f"macro_b1_iou={macro(b1v):.4f} macro_r1_iou={macro(r1v):.4f} "
          f"macro_fn_iou={macro(fnv):.4f} macro_b1_pg={macro(pb1):.4f} "
          f"macro_r1_pg={macro(pr1):.4f} macro_fn_pg={macro(pfn):.4f} "
          f"macro20_fn_iou={macro(fnv,20):.4f} macro20_r1_iou={macro(r1v,20):.4f} "
          f"macro20_b1_iou={macro(b1v,20):.4f} macro20_fn_pg={macro(pfn,20):.4f} "
          f"macro20_shuf_iou={macro(shv,20):.4f} macro20_shuf_pg={macro(psh,20):.4f} "
          f"c1={c1} c2={c2} c3={c3}")

    # Hand back the per-instance arrays and each instance's (rid, label) key, so a
    # caller can redo the paired tests at a different unit of analysis without a second
    # copy of the scoring path drifting away from this one (see cluster_stats.py). The
    # CLI ignores the return value; nothing above changes.
    return {"keys": [it[5] for it in te],
            "iou": {"b1": b1v, "r1": r1v, "fn": fnv, "shuf": shv, "mask": mkv,
                    "tshuf": tsv, "q4": q4v},
            "pg": {"b1": pb1, "r1": pr1, "fn": pfn, "shuf": psh, "mask": pmk,
                   "tshuf": pts, "q4": pq4},
            "subst": {"n": len(te_ex),
                      "target_pg": float(np.nanmean(tgt_pg)),
                      "target_iou": float(np.nanmean(tgt_iou)),
                      "donor_pg": float(np.nanmean(sub_pg)),
                      "donor_iou": float(np.nanmean(sub_iou))}}


def _selfcheck():
    """Structural check: confirm the crossattn+raw combination (never run before
    this script) trains/evaluates/shuffles/masks without error on a tiny synthetic."""
    import torch
    rng = np.random.default_rng(0); recs = []
    for k in range(60):
        N = 30
        tc = np.sort(rng.uniform(0, 10, N)); dur = np.full(N, 0.2); vel = np.zeros(N)
        lx, ly = rng.uniform(.3, .7, 2)
        x = rng.uniform(0, 1, N); y = rng.uniform(0, 1, N)
        near = rng.random(N) < 0.4
        x[near] = lx + rng.normal(0, .02, near.sum()); y[near] = ly + rng.normal(0, .02, near.sum())
        fix = np.stack([x, y, tc, dur, vel], 1).astype(np.float32)
        recs.append({"rid": f"r{k}", "subject": f"s{k}", "fix": fix,
                     "labels": [{"label": "L", "ellipses": [(lx - .05, ly - .05, lx + .05, ly + .05)],
                                 "mentions": [], "mtext": ["left lung nodule"]}]})
    p = Path("/tmp/_final_selfcheck.pt"); torch.save((recs, ["L"]), p)
    run(p, epochs=8)
    print("\nself-check ran (structural only -- crossattn+raw trains/evals/shuffles/masks cleanly)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="align.pt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--split-seed", type=int, default=0, dest="split_seed",
                    help="patient-split seed. Changes which patients are held out, so it "
                         "tests robustness to the sample rather than to training noise.")
    ap.add_argument("--external", default=None,
                    help="CSV of an external baseline's per-instance scores "
                         "(rid,label,iou,pg) -- e.g. radzero_test_scores.csv -- joined on "
                         "(rid,label) for paired significance tests.")
    ap.add_argument("--seed", type=int, default=0,
                    help="training seed (weight init + example order). Split and shuffle-"
                         "control permutation stay fixed, so seeds are directly comparable.")
    ap.add_argument("--with-inwin", action="store_true",
                    help="restore the in-1.5s-window indicator the shipped model drops. Only "
                         "for reproducing runs from before it was removed; see core.")
    a = ap.parse_args()
    if a.with_inwin:
        import core
        core.USE_INWIN = True
    import core as _am
    print(f"temporal features: {_am.temporal_dim()}-dim "
          f"(in-window indicator {'ON' if _am.USE_INWIN else 'OFF'})", flush=True)
    if Path(a.cache).exists():
        run(a.cache, a.epochs, a.seed, a.external, a.split_seed)
    else:
        _selfcheck()

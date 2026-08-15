"""Can the paper's Table I be reproduced by calibrating on the mention-resolved cohort?

Quantities derived from training data -- the label-conditional anatomical prior and the
directional half-plane orientation -- are estimated on the mention-resolved training cohort,
following the task definition. Estimating them on all training instances instead reproduces
the rows that use no prior exactly and misses every row that does, by a few thousandths.

This script builds Table I under both training cohorts side by side and prints the published
values underneath, so the choice can be checked rather than taken on trust. No training is
involved: these constructions have no learned parameters.

    python repro_check.py --cache align.pt
"""
import argparse

import numpy as np
from scipy.ndimage import zoom

from core import EVAL_RES, TUNE_SIGMAS, iou, pointing, raster, tune_thresholds, word_feat
from structured_baselines import b1_weights, modulate, scan_heat, word_dirs, _wsplat
from prior_and_swap import label_prior, prior_heat, split

TS = tune_thresholds()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="align.pt")
    ap.add_argument("--split-seed", type=int, default=0, dest="split_seed")
    a = ap.parse_args()

    import torch
    recs, labels = torch.load(a.cache, weights_only=False)
    li = {l: i for i, l in enumerate(labels)}
    part = split(recs, a.split_seed)

    def insts(p):
        return [(r["fix"], d["mentions"], d["ellipses"], li[d["label"]],
                 word_feat(d.get("mtext", [])), d["label"])
                for r in recs if part(r) == p for d in r["labels"]]

    tr, va, te = insts("train"), insts("val"), insts("test")
    va_m = [it for it in va if len(it[1])]
    te_m = [it for it in te if len(it[1])]
    print(f"val {len(va)} -> mention-resolved {len(va_m)}   "
          f"test {len(te)} -> mention-resolved {len(te_m)}", flush=True)

    # Two candidate training cohorts for the training-derived quantities. Rows that use no
    # prior matched the paper exactly while every prior-bearing row differed, which points at
    # the prior itself rather than at calibration or the evaluation cohort.
    tr_m = [it for it in tr if len(it[1])]
    print(f"train {len(tr)} -> mention-resolved {len(tr_m)}", flush=True)
    PRIORS = {"prior on ALL train": (label_prior((it[2], it[5]) for it in tr), word_dirs(tr)),
              "prior on RESOLVED train": (label_prior((it[2], it[5]) for it in tr_m),
                                          word_dirs(tr_m))}

    def masks(items, words, prior=None, dirs=None):
        out = []
        for it in items:
            m = prior.get(it[5])
            out.append(modulate(m, it[4], dirs) if (m is not None and words) else m)
        return out

    # every rule-based row of Table I, as the paper describes them
    VARIANTS = [
        ("Complete-scanpath density", lambda i, it, mk, sg: _wsplat(it[0], it[0][:, 3], sg), False),
        ("1.5-s temporal baseline",   lambda i, it, mk, sg: _wsplat(it[0], b1_weights(it[0], it[1]), sg), False),
        ("Anatomical prior",          lambda i, it, mk, sg: prior_heat(mk[i], sg), False),
        ("  + directional terms",     lambda i, it, mk, sg: prior_heat(mk[i], sg), True),
        ("Prior x scanpath",          lambda i, it, mk, sg: scan_heat(it[0], mk[i], sg), False),
        ("  + directional terms",     lambda i, it, mk, sg: scan_heat(it[0], mk[i], sg), True),
        ("Prior x scanpath x gate",   lambda i, it, mk, sg: _wsplat(it[0], b1_weights(it[0], it[1], mk[i]), sg), False),
        ("Combined structured",       lambda i, it, mk, sg: _wsplat(it[0], b1_weights(it[0], it[1], mk[i]), sg), True),
    ]

    def evaluate(fn, items, mk, sg, th):
        io, pg = [], []
        for i, it in enumerate(items):
            h = fn(i, it, mk, sg)
            up = zoom(h, EVAL_RES / h.shape[0], order=0)
            g = raster(it[2])
            io.append(iou(up, g, th)); pg.append(pointing(up, g))
        return float(np.nanmean(io)), float(np.nanmean(pg))

    def tune(fn, items, mk, sigmas):
        # Accumulate per instance rather than materializing every upsampled map at once, so
        # memory stays constant in the cohort size.
        best = (sigmas[0], TS[0], -1.0)
        for sg in sigmas:
            acc = np.zeros(len(TS)); cnt = np.zeros(len(TS))
            for i, it in enumerate(items):
                h = fn(i, it, mk, sg)          # prior_heat is already EVAL_RES; splats are 64
                up = h if h.shape[0] == EVAL_RES else zoom(h, EVAL_RES / h.shape[0], order=0)
                g = raster(it[2])
                for k, t in enumerate(TS):
                    v = iou(up, g, t)
                    if np.isfinite(v):
                        acc[k] += v; cnt[k] += 1
            m = np.where(cnt > 0, acc / np.maximum(cnt, 1), -1.0)
            k = int(np.argmax(m))
            if m[k] > best[2]:
                best = (sg, TS[k], m[k])
        return best[0], best[1]

    print(f"\n{'row':28s} " + " ".join(f"{k:>24s}" for k in PRIORS))
    for name, fn, words in VARIANTS:
        row = []
        for pname in PRIORS:
            prior, dirs = PRIORS[pname]
            cohort = va
            mk_v = masks(cohort, words, prior, dirs)
            sigmas = [0.0] + list(TUNE_SIGMAS) if "prior" in name.lower() and "scanpath" not in name.lower() else TUNE_SIGMAS
            sg, th = tune(fn, cohort, mk_v, sigmas)
            mk_t = masks(te_m, words, prior, dirs)
            i_, p_ = evaluate(fn, te_m, mk_t, sg, th)
            row.append(f"{p_:.4f}/{i_:.4f}")
        print(f"{name:28s} " + " ".join(f"{r:>24s}" for r in row), flush=True)

    print("\nPaper Table I (pointing/IoU): 0.4063/0.2008  0.6211/0.2773  0.5035/0.2736  "
          "0.5978/0.3248\n  0.6717/0.3056  0.7183/0.3408  0.7528/0.3201  0.7923/0.3439")


if __name__ == "__main__":
    main()

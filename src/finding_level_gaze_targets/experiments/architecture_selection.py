"""Validation-based comparison of selector fusion and position encodings.

The complete two-by-two design compares concatenation with cross-attention and
raw coordinates with Fourier features. Every configuration uses the same
patient partition, training inputs, validation grid, and optimizer seed.
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from finding_level_gaze_targets.maps.core import (iou, pointing, raster, word_feat, TUNE_SIGMAS,
                         tune_thresholds)
from finding_level_gaze_targets.models.selector import train_model, predict_raw, blur_norm


def run(cache, epochs, seed=0, split_seed=0):
    import torch
    recs, labels = torch.load(cache, weights_only=False)
    li = {L: i for i, L in enumerate(labels)}
    subs = np.array([r["subject"] for r in recs]); uniq = np.array(sorted(set(subs)))
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
                            word_feat(d.get("mtext", []))))
        return out
    tr, va, te = insts("train"), insts("val"), insts("test")
    print(f"instances train/val/test = {len(tr)}/{len(va)}/{len(te)}, {len(labels)} labels", flush=True)

    variants = {
        "concat+fourier": dict(fusion="concat", pos_mode="fourier"),
        "concat+raw": dict(fusion="concat", pos_mode="raw"),
        "crossattn+fourier": dict(fusion="crossattn", pos_mode="fourier"),
        "crossattn+raw": dict(fusion="crossattn", pos_mode="raw"),
    }
    nets = {}
    for name, cfg in variants.items():
        print(f"training +gaze+text [{name}] (seed={seed})...", flush=True)
        nets[name] = train_model(tr, labels, use_position=True, epochs=epochs,
                                 use_text=True, seed=seed, **cfg)

    def cache_raw(items, name):
        pos_mode = variants[name]["pos_mode"]   # fusion is baked into the trained net already
        return [(predict_raw(nets[name], f, m, l, use_position=True, use_text=True, wf=wf,
                             pos_mode=pos_mode), raster(e)) for f, m, e, l, wf in items]

    def metric_at(cached, t, sigma, metric):
        out = []
        for raw, gt in cached:
            hm = blur_norm(raw, sigma)
            out.append(iou(hm, gt, t) if metric == "iou" else pointing(hm, gt))
        return np.array(out)

    ts = tune_thresholds()
    sigmas = TUNE_SIGMAS

    def tune(va_cached):
        best = (1.5, 0.3, -1.0)
        for s in sigmas:
            scores = [np.nanmean(metric_at(va_cached, t, s, "iou")) for t in ts]
            j = int(np.argmax(scores))
            if scores[j] > best[2]:
                best = (s, ts[j], scores[j])
        if best[0] in (sigmas[0], sigmas[-1]) or best[1] in (ts[0], ts[-1]):
            print(f"  WARNING: tuned (sigma={best[0]}, thr={best[1]:.3f}) on a grid boundary")
        return best[0], best[1], best[2]      # best[2] = the VALIDATION score

    print("caching + tuning...", flush=True)
    vals, tuned, val_iou = {}, {}, {}
    for name in variants:
        va_c = cache_raw(va, name)
        tuned[name] = tune(va_c)
        s, t, vscore = tuned[name]
        val_iou[name] = vscore
        te_c = cache_raw(te, name)
        vals[name] = {"iou": metric_at(te_c, t, s, "iou"), "pg": metric_at(te_c, t, s, "pg")}

    print("\nval-tuned (sigma, threshold):")
    for name in variants:
        print(f"  {name:38s} sigma={tuned[name][0]} thr={tuned[name][1]:.3f} "
              f"VAL_IoU={val_iou[name]:.4f}")

    print(f"\nARCHITECTURE COMPARISON (n={len(te)}):{'':7s}IoU      pointing-game")
    for name in variants:
        v = vals[name]
        print(f"  {name:38s} {np.nanmean(v['iou']):.4f}   {np.nanmean(v['pg']):.4f}")

    def paired_iou(a, b):
        ok = np.isfinite(a) & np.isfinite(b)
        _, p = wilcoxon(a[ok], b[ok]); return np.median(a[ok] - b[ok]), p

    def paired_pg(a, b):
        ok = np.isfinite(a) & np.isfinite(b)
        _, p = wilcoxon(a[ok], b[ok]); return np.mean(a[ok] - b[ok]), p

    base = "concat+fourier"
    print(f"\n--- sensitivity checks (vs {base}) ---")
    for name in variants:
        if name == base:
            continue
        di, pi = paired_iou(vals[name]["iou"], vals[base]["iou"])
        dp, pp = paired_pg(vals[name]["pg"], vals[base]["pg"])
        print(f"{name:38s} IoU Δmed ={di:+.4f} p={pi:.3g} {'*' if pi < 0.05 else ' '}  |  "
              f"pointing Δmean={dp:+.4f} p={pp:.3g} {'*' if pp < 0.05 else ''}")

    for name in variants:
        v = vals[name]
        print(f"ARCH split={split_seed} seed={seed} variant={name!r} val_iou={val_iou[name]:.4f} "
              f"iou={np.nanmean(v['iou']):.4f} pg={np.nanmean(v['pg']):.4f} "
              f"sigma={tuned[name][0]} thr={tuned[name][1]:.3f}")

    win_val = max(val_iou, key=val_iou.get)
    win_test = max(variants, key=lambda n: np.nanmean(vals[n]["iou"]))
    print(f"\nSELECTION CHECK split={split_seed} seed={seed}: validation picks {win_val!r}; "
          f"test-best is {win_test!r}; agree={win_val == win_test}")

def _selfcheck():
    """Confirm that all four architecture variants train and evaluate."""
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
    p = Path("/tmp/_stage3_selfcheck.pt"); torch.save((recs, ["L"]), p)
    run(p, epochs=8)
    print("\nself-check ran (structural only -- all variants trained/evaluated cleanly)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="align.pt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-seed", type=int, default=0)
    a = ap.parse_args()
    if Path(a.cache).exists():
        run(a.cache, a.epochs, a.seed, a.split_seed)
    else:
        _selfcheck()

"""Stage 2 (agreed plan): add TEXT (word content) on top of Temporal-only and +gaze,
completing the 2x2 factorial:
    Temporal-only   +gaze (temporal+position)
    +text (temporal+word)      +gaze+text (temporal+position+word)

Fusion mechanism is FIXED (same concat-into-MLP as Stage 1; Stage 3 later asks
whether cross-attention beats this, as an isolated architecture ablation, not
mixed in here).

Text feature = the existing WORD_TERMS descriptor bag (core.WORD_DIM=10):
indices 0-6 are SPATIAL (left/right/bilateral/upper/lower/middle/retrocardiac),
7-9 are severity/extent (small/large/diffuse) -- not spatial.

Validation (eval-time masking, not retraining -- cheaper and more directly
causal, per the agreed plan): zero out the SPATIAL indices only, keep severity
indices, using the SAME trained model and SAME (sigma,threshold) as the
unmasked run. If performance doesn't drop, text's apparent gain isn't coming
from spatial descriptors specifically (could be label-redundant lexical content
instead -- e.g. mention text almost certainly contains the very keyword used to
assign the label).

Usage (reuses align.pt; no new extraction):
  python masking_control.py --cache align.pt --epochs 40
  python masking_control.py                                   # synthetic self-check
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from core import (iou, pointing, raster, word_feat)
from selector import (train_model, predict_raw, blur_norm)

SPATIAL_IDX = list(range(7))          # left,right,bilateral,upper,lower,middle,retrocardiac
SEVERITY_IDX = [7, 8, 9]              # small/mild.., large/severe.., diffuse/patchy


def mask_spatial(wf):
    wf = wf.copy()
    wf[SPATIAL_IDX] = 0.0
    return wf


def run(cache, epochs):
    import torch
    recs, labels = torch.load(cache, weights_only=False)
    li = {L: i for i, L in enumerate(labels)}
    subs = np.array([r["subject"] for r in recs]); uniq = np.array(sorted(set(subs)))
    rng = np.random.default_rng(0); rng.shuffle(uniq); n = len(uniq)
    val_s = set(uniq[:int(n * .15)]); test_s = set(uniq[int(n * .15):int(n * .45)])
    part = lambda r: "val" if r["subject"] in val_s else "test" if r["subject"] in test_s else "train"

    def insts(p):
        out = []
        for r in recs:
            if part(r) != p:
                continue
            for d in r["labels"]:
                wf = word_feat(d.get("mtext", []))
                out.append((r["fix"], d["mentions"], d["ellipses"], li[d["label"]], wf))
        return out
    tr, va, te = insts("train"), insts("val"), insts("test")
    print(f"instances train/val/test = {len(tr)}/{len(va)}/{len(te)}, {len(labels)} labels", flush=True)

    print("training Temporal-only model...", flush=True)
    net_r1 = train_model(tr, labels, use_position=False, epochs=epochs, use_text=False)
    print("training +gaze (temporal+position)...", flush=True)
    net_gz = train_model(tr, labels, use_position=True, epochs=epochs, use_text=False)
    print("training +text (temporal+word)...", flush=True)
    net_tx = train_model(tr, labels, use_position=False, epochs=epochs, use_text=True)
    print("training +gaze+text (temporal+position+word)...", flush=True)
    net_gt = train_model(tr, labels, use_position=True, epochs=epochs, use_text=True)

    NETS = {"result1": (net_r1, False, False), "gaze": (net_gz, True, False),
            "text": (net_tx, False, True), "gaze_text": (net_gt, True, True)}

    def raw_for(method, item, masked=False):
        f, m, e, l, wf = item
        net, use_pos, use_txt = NETS[method]
        w = mask_spatial(wf) if (masked and use_txt) else wf
        return predict_raw(net, f, m, l, use_position=use_pos, use_text=use_txt, wf=w)

    def cache_raw(items, method, masked=False):
        return [(raw_for(method, it, masked), raster(it[2])) for it in items]

    def metric_at(cached, t, sigma, metric):
        out = []
        for raw, gt in cached:
            hm = blur_norm(raw, sigma)
            out.append(iou(hm, gt, t) if metric == "iou" else pointing(hm, gt))
        return np.array(out)

    ts = np.linspace(0.05, 0.6, 23)
    sigmas = [0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]

    def tune(va_cached):
        best = (1.5, 0.3, -1.0)
        for s in sigmas:
            scores = [np.nanmean(metric_at(va_cached, t, s, "iou")) for t in ts]
            j = int(np.argmax(scores))
            if scores[j] > best[2]:
                best = (s, ts[j], scores[j])
        return best[0], best[1]

    print("caching + tuning (this trains nothing further, just eval sweeps)...", flush=True)
    tuned, te_cache = {}, {}
    for name in ("result1", "gaze", "text", "gaze_text"):
        va_c = cache_raw(va, name)
        tuned[name] = tune(va_c)
        te_cache[name] = cache_raw(te, name)
    te_cache_masked = {name: cache_raw(te, name, masked=True) for name in ("text", "gaze_text")}

    print("val-tuned (sigma, threshold):")
    for name in ("result1", "gaze", "text", "gaze_text"):
        print(f"  {name:10s} {tuned[name]}")

    vals = {}
    for name in ("result1", "gaze", "text", "gaze_text"):
        s, t = tuned[name]
        vals[name] = {"iou": metric_at(te_cache[name], t, s, "iou"),
                     "pg": metric_at(te_cache[name], t, s, "pg")}
    for name in ("text", "gaze_text"):
        s, t = tuned[name]
        vals[name + "_masked"] = {"iou": metric_at(te_cache_masked[name], t, s, "iou"),
                                   "pg": metric_at(te_cache_masked[name], t, s, "pg")}

    print(f"\nSTAGE 2 (n={len(te)}):                    IoU      pointing-game")
    order = [("result1", "Temporal-only"), ("gaze", "+gaze (position)"),
             ("text", "+text (word)"), ("text_masked", "  +text SPATIAL-MASKED"),
             ("gaze_text", "+gaze+text (full)"),
             ("gaze_text_masked", "  +gaze+text SPATIAL-MASKED")]
    for key, label in order:
        v = vals[key]
        print(f"  {label:28s} {np.nanmean(v['iou']):.4f}   {np.nanmean(v['pg']):.4f}")

    def paired_iou(a, b):
        ok = np.isfinite(a) & np.isfinite(b)
        _, p = wilcoxon(a[ok], b[ok]); return np.median(a[ok] - b[ok]), p

    def paired_pg(a, b):
        ok = np.isfinite(a) & np.isfinite(b)
        _, p = wilcoxon(a[ok], b[ok]); return np.mean(a[ok] - b[ok]), p

    print("\n--- main effects (vs Temporal-only) ---")
    for name, label in [("gaze", "+gaze"), ("text", "+text"), ("gaze_text", "+gaze+text")]:
        di, pi = paired_iou(vals[name]["iou"], vals["result1"]["iou"])
        dp, pp = paired_pg(vals[name]["pg"], vals["result1"]["pg"])
        print(f"{label:12s} IoU Δmed ={di:+.4f} p={pi:.3g} {'*' if pi < 0.05 else ' '}  |  "
              f"pointing Δmean={dp:+.4f} p={pp:.3g} {'*' if pp < 0.05 else ''}")

    print("\n--- does +gaze+text beat +gaze alone, and beat +text alone? (direct paired,\n"
          "    weak power at single-seed -- NOT a per-instance oracle-max, that's unfair) ---")
    for base in ("gaze", "text"):
        di, pi = paired_iou(vals["gaze_text"]["iou"], vals[base]["iou"])
        dp, pp = paired_pg(vals["gaze_text"]["pg"], vals[base]["pg"])
        print(f"  gaze_text vs {base:5s}: IoU Δmed ={di:+.4f} p={pi:.3g} {'*' if pi < 0.05 else ' '}  |  "
              f"pointing Δmean={dp:+.4f} p={pp:.3g} {'*' if pp < 0.05 else ''}")

    print("\n--- spatial-word masking control (does text use SPATIAL descriptors, "
          "not just label-redundant lexical content?) ---")
    for name, label in [("text", "+text"), ("gaze_text", "+gaze+text")]:
        di, pi = paired_iou(vals[name]["iou"], vals[name + "_masked"]["iou"])
        dp, pp = paired_pg(vals[name]["pg"], vals[name + "_masked"]["pg"])
        print(f"{label:12s} real vs spatial-masked: IoU Δmed ={di:+.4f} p={pi:.3g} "
              f"{'*' if pi < 0.05 else ' '}  |  pointing Δmean={dp:+.4f} p={pp:.3g} "
              f"{'*' if pp < 0.05 else ''}")
        if not ((di > 0 and pi < 0.05) or (dp > 0 and pp < 0.05)):
            print(f"  -> WARNING: masking spatial words barely changes {label}'s output.\n"
                  "     Its gain (if any) is likely NOT from spatial descriptor content --\n"
                  "     could be label-redundant lexical signal (severity words, or the\n"
                  "     mention text simply containing the keyword used to assign the label).")


def _selfcheck():
    """Synthetic: label alone is uninformative (single shared label), only the
    SPATIAL word ('left' vs 'right') distinguishes where the lesion is; gaze and
    timing carry no signal. +text should win; masking spatial words should kill it."""
    import torch
    rng = np.random.default_rng(0); recs = []
    for k in range(150):
        N = 30
        tc = np.linspace(0, 10, N); dur = np.full(N, 0.2); vel = np.zeros(N)
        left = rng.random() < 0.5
        lx, ly = (rng.uniform(.1, .3), rng.uniform(.3, .7)) if left else \
                 (rng.uniform(.7, .9), rng.uniform(.3, .7))
        x = rng.uniform(0, 1, N); y = rng.uniform(0, 1, N)   # gaze itself carries NO signal
        fix = np.stack([x, y, tc, dur, vel], 1).astype(np.float32)
        mtext = ["there is a nodule in the left lung."] if left else \
                ["there is a nodule in the right lung."]
        recs.append({"rid": f"r{k}", "subject": f"s{k}", "fix": fix,
                     "labels": [{"label": "L", "ellipses": [(lx - .05, ly - .05, lx + .05, ly + .05)],
                                 "mentions": [], "mtext": mtext}]})
    p = Path("/tmp/_stage2_selfcheck.pt"); torch.save((recs, ["L"]), p)
    run(p, epochs=15)
    print("\nself-check ran (only the spatial WORD carries signal, gaze is noise -> "
          "expect +text/+gaze+text to win and spatial-masking to hurt them)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="align.pt")
    ap.add_argument("--epochs", type=int, default=40)
    a = ap.parse_args()
    if Path(a.cache).exists():
        run(a.cache, a.epochs)
    else:
        _selfcheck()

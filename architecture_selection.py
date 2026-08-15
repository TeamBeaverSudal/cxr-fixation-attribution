"""Stage 3 (agreed plan): architecture sensitivity checks on top of the ALREADY
VALIDATED +gaze+text model (Stage 1/2 passed the substantive causal tests --
shuffle-control, spatial-word masking). This is NOT a new stopping gate; it asks
whether two literature-borrowed design choices actually earn their complexity:

  Axis A: raw (x,y) vs Fourier(x,y) position encoding, fusion held at 'concat'.
          Fourier was used from Stage 1 onward to avoid a false-negative at the
          stopping gate (MLP spectral bias); THIS is the deferred check of
          whether it actually mattered for the final numbers.

  Axis B: 'concat' (Stage 1/2's validated MLP-over-concatenated-features) vs
          'crossattn' (a genuine, minimal single-layer cross-attention: a
          word+label QUERY does scaled dot-product attention over per-fixation
          KEYS built from temporal+label+position -- TransVG's "let a simple
          homogeneous mechanism replace hand-engineered fusion" principle,
          empirically tested rather than assumed), pos_mode held at 'fourier'.

Both axes vary ONE thing at a time against the same +gaze+text baseline --
consistent with the whole staged-ablation discipline (never bundle >1 change).

Usage (reuses align.pt; no new extraction):
  python architecture_selection.py --cache align.pt --epochs 40
  python architecture_selection.py                                   # structural self-check
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from core import (iou, pointing, raster, word_feat, TUNE_SIGMAS,
                         tune_thresholds)
from selector import train_model, predict_raw, blur_norm


def run(cache, epochs, seed=0, split_seed=0):
    import torch
    recs, labels = torch.load(cache, weights_only=False)
    li = {L: i for i, L in enumerate(labels)}
    subs = np.array([r["subject"] for r in recs]); uniq = np.array(sorted(set(subs)))
    # Parameterised so the ablation can be repeated across patient splits. The paper
    # argues split variation dominates seed variation, then selected this architecture
    # on seeds alone; the margins between cells vary about 3x more across splits than
    # across seeds elsewhere in this project, so the choice needs both families.
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

    # Full 2x2. The original three variants changed one factor at a time from
    # concat+fourier and never included crossattn+raw -- which is the configuration
    # actually shipped. "raw beats fourier under concat" and "crossattn beats concat
    # under fourier" do not jointly establish that their combination is best, so the
    # adopted model was never compared against the alternatives it was chosen over.
    variants = {
        "concat+fourier (origin)": dict(fusion="concat", pos_mode="fourier"),
        "concat+raw": dict(fusion="concat", pos_mode="raw"),
        "crossattn+fourier (SHIPPED)": dict(fusion="crossattn", pos_mode="fourier"),
        "crossattn+raw (superseded)": dict(fusion="crossattn", pos_mode="raw"),
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

    print(f"\nSTAGE 3 (n={len(te)}):{'':18s}IoU      pointing-game")
    for name in variants:
        v = vals[name]
        print(f"  {name:38s} {np.nanmean(v['iou']):.4f}   {np.nanmean(v['pg']):.4f}")

    def paired_iou(a, b):
        ok = np.isfinite(a) & np.isfinite(b)
        _, p = wilcoxon(a[ok], b[ok]); return np.median(a[ok] - b[ok]), p

    def paired_pg(a, b):
        ok = np.isfinite(a) & np.isfinite(b)
        _, p = wilcoxon(a[ok], b[ok]); return np.mean(a[ok] - b[ok]), p

    base = "concat+fourier (origin)"
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

    # Which variant does VALIDATION pick? Selecting an architecture on test scores biases
    # the selected model's reported test number upward. The selection is only legitimate if
    # validation ranks the same winner -- print it so that can be checked rather than
    # assumed.
    win_val = max(val_iou, key=val_iou.get)
    win_test = max(variants, key=lambda n: np.nanmean(vals[n]["iou"]))
    print(f"\nSELECTION CHECK split={split_seed} seed={seed}: validation picks {win_val!r}; "
          f"test-best is {win_test!r}; agree={win_val == win_test}")

    print("\nread: Fourier mattering = 'raw position' loses to baseline (sig). Cross-\n"
          "attention earning its complexity = 'cross-attention' BEATS baseline (sig);\n"
          "if it ties or loses, the simple concat/MLP (Stage 1/2's validated choice)\n"
          "is the one to keep -- per Occam's razor / small-data discipline, don't\n"
          "adopt more complex machinery than the data supports.")


def _selfcheck():
    """Structural check only (Stage 3 is a sensitivity check on an already-validated
    causal claim, not a new claim needing a dedicated proof synthetic): confirm all
    three variants (raw pos, crossattn fusion) train and evaluate without error and
    produce non-degenerate attention on a small synthetic."""
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

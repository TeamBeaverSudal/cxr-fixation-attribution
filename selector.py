"""Stage 1 (agreed plan): Temporal-only (position-agnostic) vs +gaze (Fourier-encoded
(x,y) added to the attention computation), with position-shuffle as the MANDATORY
validation attached to +gaze (not a separate/optional group).

"position-agnostic" is defined precisely as: attention computation does not see
explicit (x,y). (Temporal-only's final softmax-splat still uses real fixation (x,y) for
localization — that hasn't changed — only whether the ATTENTION that produces the
weights gets to see position is what "+gaze" toggles.)

Fourier (not raw) coordinates are used from the start: this is the stopping gate,
and a plain MLP provably struggles to learn high-frequency functions of raw
low-dimensional coordinates (spectral bias). Using raw (x,y) here risks a FALSE
NEGATIVE (concluding "position doesn't help" when the real problem is "raw MLP
can't extract it"). Fourier-vs-raw is deferred to Stage 3 as a sensitivity check
on top of a result that already passed the gate, not the first/decisive test.

Position-shuffle control: within each instance, permute WHICH fixation gets WHICH
(x,y) (word/timing/label untouched). If the model doesn't degrade under this, it
isn't using genuine per-instance gaze position — full stop, regardless of whether
it beats Temporal-only on the unshuffled numbers.

Stopping rule:
  +gaze beats Temporal-only (sig) AND +gaze(real) beats +gaze(shuffled) (sig) -> PASS,
  proceed to Stage 2 (+text). Otherwise -> STOP here.

Usage (reuses align.pt from core.py's extraction; no new extraction needed):
  python selector.py --cache align.pt --epochs 40
  python selector.py                                   # synthetic self-check
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from core import (iou, pointing, b1_heat, align_feats, raster, inside_ellipses,
                         WORD_DIM)

FOURIER_L = 4
POS_DIM = 4 * FOURIER_L  # 2 coords x (sin+cos) x L


def fourier_feat(xy, L=FOURIER_L):
    """NeRF-style positional encoding: avoids the spectral-bias failure mode of
    feeding raw low-dim (x,y) straight into a small MLP. xy in [0,1]. -> (N, 4L)."""
    freqs = (2.0 ** np.arange(L)).astype(np.float32) * np.pi
    ang = xy[:, :, None].astype(np.float32) * freqs[None, None, :]   # (N,2,L)
    return np.concatenate([np.sin(ang), np.cos(ang)], axis=-1).reshape(len(xy), -1).astype(np.float32)


def pos_feat(xy, pos_mode):
    """Stage 3 axis A: 'fourier' (default, validated in Stage 1/2) vs 'raw' (x,y
    fed straight in, no encoding) -- isolates whether the Fourier encoding itself
    mattered, or any position channel would have done."""
    if pos_mode == "raw":
        return xy.astype(np.float32)
    return fourier_feat(xy)


def _pos_dim(use_position, pos_mode):
    if not use_position:
        return 0
    return 2 if pos_mode == "raw" else POS_DIM


def _temporal_dim():
    """Deferred so this module stays importable without core at import time."""
    from core import temporal_dim
    return temporal_dim()


def make_net(labels, use_position, use_text=False, fusion="concat", pos_mode="fourier"):
    """fusion='concat' (default, Stage 1/2's validated MLP-over-concatenated-
    features) or 'crossattn' (Stage 3 axis B: word+label QUERY does scaled
    dot-product attention over per-fixation KEYS built from temporal+label+
    position -- a genuine, minimal single-layer cross-attention, TransVG-style
    'let a simple homogeneous mechanism do the fusion' rather than hand-engineering)."""
    import torch.nn as nn
    pd = _pos_dim(use_position, pos_mode)
    wd = WORD_DIM if use_text else 0

    if fusion == "crossattn":
        dk = 16
        key_dim = _temporal_dim() + 8 + pd

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = nn.Embedding(len(labels), 8)
                self.key_proj = nn.Linear(key_dim, dk)
                self.query_proj = nn.Linear(8 + wd, dk)

            def attn(self, feats, lab, posf, wf=None):
                import torch
                e = self.emb(torch.tensor(lab))                    # (8,)
                kparts = [feats, e.expand(len(feats), -1)]
                if use_position:
                    kparts.append(torch.from_numpy(posf))
                K = self.key_proj(torch.cat(kparts, 1))             # (N, dk)
                qparts = [e] + ([torch.from_numpy(wf)] if use_text else [])
                q = self.query_proj(torch.cat(qparts, 0))           # (dk,)
                s = (K @ q) / (dk ** 0.5)                           # scaled dot-product
                return torch.softmax(s, 0)
        return Net()

    in_dim = _temporal_dim() + 8 + pd + wd

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(len(labels), 8)
            self.mlp = nn.Sequential(nn.Linear(in_dim, 32), nn.ReLU(),
                                     nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1))

        def attn(self, feats, lab, posf, wf=None):
            import torch
            e = self.emb(torch.tensor(lab)).expand(len(feats), -1)
            parts = [feats, e]
            if use_position:
                parts.append(torch.from_numpy(posf))   # already (N, pd), per-fixation
            if use_text:
                parts.append(torch.from_numpy(wf).expand(len(feats), -1))  # one vector, broadcast
            s = self.mlp(torch.cat(parts, 1)).squeeze(-1)
            return torch.softmax(s, 0)
    return Net()


def train_model(tr, labels, use_position, epochs, use_text=False, fusion="concat",
                pos_mode="fourier", seed=0):
    """tr items are 4-tuples (fix,ment,ells,lab) when use_text=False, or 5-tuples
    (fix,ment,ells,lab,wf) when use_text=True -- backward compatible with Stage 1."""
    import torch, os
    torch.set_num_threads(max(1, os.cpu_count() - 1))
    torch.manual_seed(seed)
    np.random.seed(seed)  # np.random.shuffle(tr) below draws from the global RNG; without
                          # this, example order (hence final weights) varies run-to-run.
                          # The drift is small for the headline numbers but large enough to
                          # flip the spatial-word-masking control's significance, so this
                          # must be seeded AND swept (see --seed) rather than left to luck.
    # Shuffle a COPY. np.random.shuffle is in-place, so shuffling the caller's list left
    # it permuted afterwards and the NEXT train_model call in the same process started
    # from that permutation instead of the original order. Re-seeding does not help: the
    # seed fixes the permutation, not the order it is applied to. The consequence was that
    # a model's weights depended on how many models had been trained before it -- the two
    # tables reporting the same configuration disagreed, and the four cells of the
    # architecture ablation each saw a different data order, which is a confound in the
    # comparison the ablation exists to make. Copying references only; the arrays inside
    # each tuple are shared, so this costs nothing.
    tr = list(tr)
    net = make_net(labels, use_position, use_text, fusion, pos_mode)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    pd = _pos_dim(use_position, pos_mode)
    empty_pos = np.zeros((0, pd), np.float32)
    for ep in range(epochs):
        net.train(); np.random.shuffle(tr); tot = 0
        for item in tr:
            if use_text:
                fix, ment, ells, lab, wf = item[:5]   # callers may append extras (e.g. an
                                                      # instance key for external joins)
            else:
                fix, ment, ells, lab = item[:4]; wf = None
            ins = torch.from_numpy(inside_ellipses(fix, ells).astype(np.float32))
            if ins.sum() == 0:
                continue
            posf = pos_feat(fix[:, :2], pos_mode) if use_position else empty_pos
            a = net.attn(torch.from_numpy(align_feats(fix, ment)), lab, posf, wf)
            massin = (a * ins).sum()
            ent = -(a * (a + 1e-9).log()).sum()
            loss = -torch.log(massin + 1e-6) - 0.01 * ent
            opt.zero_grad(); loss.backward(); opt.step(); tot += massin.item()
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"    epoch {ep} train mass-in {tot/max(len(tr),1):.4f}", flush=True)
    net.eval()
    return net


def predict_raw(net, fix, ment, lab, use_position, shuffle_pos=False, rng=None,
                use_text=False, wf=None, pos_mode="fourier", return_weights=False,
                shuffle_temporal=False):
    """Attention-weighted point grid BEFORE blur/normalize -- lets sigma be swept
    cheaply afterward (no repeated forward passes) via blur_norm()."""
    import torch
    f = fix
    posf = pos_feat(f[:, :2], pos_mode) if use_position else np.zeros((len(f), 0), np.float32)
    if shuffle_pos:
        # Permute the coordinate rows the scorer reads while the splat stays at the recorded
        # coordinates. Permuting the coordinates themselves and splatting at the permuted ones
        # leaves the weight computed for a position landing on that same position, so the map
        # remains a function of position and the control barely bites; this breaks the
        # correspondence the model is supposed to depend on.
        posf = posf[rng.permutation(len(f))]
    tf = align_feats(f, ment)
    if shuffle_temporal:
        # The complement of shuffle_pos: keep every coordinate where it was recorded and
        # permute the mention-offset and kinematic rows instead, so a fixation is scored on
        # another fixation's timing and dynamics while the map still lands on real fixations.
        tf = tf[rng.permutation(len(tf))]
    with torch.no_grad():
        a = net.attn(torch.from_numpy(tf), lab, posf, wf).numpy()
    # return_weights hands back the per-fixation attention itself rather than the splat, so a
    # fixation-level diagnostic reads exactly the selection the heatmap was built from.
    return a if return_weights else _raw_grid(f[:, 0], f[:, 1], a)


def _raw_grid(x, y, wt):
    from core import HEAT_RES
    m = np.zeros((HEAT_RES, HEAT_RES), np.float32)
    xi = np.clip((x * HEAT_RES).astype(int), 0, HEAT_RES - 1)
    yi = np.clip((y * HEAT_RES).astype(int), 0, HEAT_RES - 1)
    np.add.at(m, (yi, xi), wt)
    return m


def blur_norm(raw, sigma):
    from scipy.ndimage import gaussian_filter
    m = gaussian_filter(raw, sigma)
    mx = m.max()
    return m / mx if mx > 0 else m


def run(cache, epochs):
    import torch
    recs, labels = torch.load(cache, weights_only=False)
    li = {L: i for i, L in enumerate(labels)}
    subs = np.array([r["subject"] for r in recs]); uniq = np.array(sorted(set(subs)))
    rng = np.random.default_rng(0); rng.shuffle(uniq); n = len(uniq)
    val_s = set(uniq[:int(n * .15)]); test_s = set(uniq[int(n * .15):int(n * .45)])
    part = lambda r: "val" if r["subject"] in val_s else "test" if r["subject"] in test_s else "train"

    def insts(p):
        return [(r["fix"], d["mentions"], d["ellipses"], li[d["label"]])
                for r in recs if part(r) == p for d in r["labels"]]
    tr, va, te = insts("train"), insts("val"), insts("test")
    print(f"instances train/val/test = {len(tr)}/{len(va)}/{len(te)}, {len(labels)} labels", flush=True)

    print("[Stage 0] training Temporal-only (position-agnostic attention)...", flush=True)
    net_r1 = train_model(tr, labels, use_position=False, epochs=epochs)
    print("[Stage 1] training +gaze (Fourier position in attention)...", flush=True)
    net_pg = train_model(tr, labels, use_position=True, epochs=epochs)

    shuf_rng = np.random.default_rng(2)  # fixed once -> same shuffled raw grid at every sigma

    def raw_for(method, item):
        f, m, e, l = item
        if method == "result1":
            return predict_raw(net_r1, f, m, l, use_position=False)
        if method == "gaze":
            return predict_raw(net_pg, f, m, l, use_position=True)
        if method == "gaze_posshuf":
            return predict_raw(net_pg, f, m, l, use_position=True,
                               shuffle_pos=True, rng=shuf_rng)

    def cache_raw(items, method):
        return [(raw_for(method, it), raster(it[2])) for it in items]

    def metric_at(cached, t, sigma, metric):
        out = []
        for raw, gt in cached:
            hm = blur_norm(raw, sigma)
            out.append(iou(hm, gt, t) if metric == "iou" else pointing(hm, gt))
        return np.array(out)

    ts = np.linspace(0.05, 0.6, 23)
    sigmas = [0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]

    def tune_sigma_threshold(va_cached):
        """Joint grid search on val: for each sigma, best threshold by IoU; keep the
        (sigma, threshold) pair with the highest val IoU overall."""
        best = (1.5, 0.3, -1.0)
        for s in sigmas:
            scores = [np.nanmean(metric_at(va_cached, t, s, "iou")) for t in ts]
            j = int(np.argmax(scores))
            if scores[j] > best[2]:
                best = (s, ts[j], scores[j])
        return best[0], best[1]

    # B1 keeps its established default (sigma=1.5) -- it's not part of the
    # position-sparsity issue (its "attention" is a hard 0/1 gate, not learned),
    # and its number should stay comparable to linker_and_temporal.py's faithful 0.225.
    va_b1 = [(b1_heat(f, m), raster(e)) for f, m, e, l in va]
    te_b1 = [(b1_heat(f, m), raster(e)) for f, m, e, l in te]
    tb1 = ts[int(np.argmax([np.nanmean([iou(hm, gt, t) for hm, gt in va_b1]) for t in ts]))]
    b1v = np.array([iou(hm, gt, tb1) for hm, gt in te_b1])
    pb1 = np.array([pointing(hm, gt) for hm, gt in te_b1])

    va_r1, te_r1 = cache_raw(va, "result1"), cache_raw(te, "result1")
    va_gz, te_gz = cache_raw(va, "gaze"), cache_raw(te, "gaze")
    te_gs = cache_raw(te, "gaze_posshuf")   # control: same (sigma,thr) as "gaze", not re-tuned

    s_r1, t_r1 = tune_sigma_threshold(va_r1)
    s_gz, t_gz = tune_sigma_threshold(va_gz)
    print(f"\nval-tuned (sigma, threshold): Temporal-only={s_r1,t_r1}  +gaze={s_gz,t_gz}  B1_sigma=1.5,thr={tb1:.3f}")

    r1v = metric_at(te_r1, t_r1, s_r1, "iou"); pr1 = metric_at(te_r1, t_r1, s_r1, "pg")
    gzv = metric_at(te_gz, t_gz, s_gz, "iou"); pgz = metric_at(te_gz, t_gz, s_gz, "pg")
    # shuffle control MUST reuse +gaze's own (sigma,threshold) -- same eval settings,
    # only the position permutation changes -- that's the whole point of the control.
    gsv = metric_at(te_gs, t_gz, s_gz, "iou"); pgs = metric_at(te_gs, t_gz, s_gz, "pg")

    def paired_iou(a, b):     # continuous metric -> median delta is meaningful
        ok = np.isfinite(a) & np.isfinite(b)
        _, p = wilcoxon(a[ok], b[ok]); return np.median(a[ok] - b[ok]), p

    def paired_pg(a, b):      # binary metric -> median is dominated by ties; use mean
        ok = np.isfinite(a) & np.isfinite(b)
        _, p = wilcoxon(a[ok], b[ok]); return np.mean(a[ok] - b[ok]), p

    print(f"\nSTAGE 1 (n={len(te)}):              IoU      pointing-game")
    print(f"  B1 fixed-window          {np.nanmean(b1v):.4f}   {np.nanmean(pb1):.4f}")
    print(f"  Temporal-only (no position)   {np.nanmean(r1v):.4f}   {np.nanmean(pr1):.4f}")
    print(f"  +gaze (Fourier position) {np.nanmean(gzv):.4f}   {np.nanmean(pgz):.4f}")
    print(f"  +gaze POSITION-SHUFFLED  {np.nanmean(gsv):.4f}   {np.nanmean(pgs):.4f}  (control)")

    d1, p1 = paired_iou(gzv, r1v); dp1, pp1 = paired_pg(pgz, pr1)
    d2, p2 = paired_iou(gzv, gsv); dp2, pp2 = paired_pg(pgz, pgs)
    print(f"\npaired IoU      +gaze vs Temporal-only   : Δmed ={d1:+.4f} p={p1:.3g} {'*' if p1 < 0.05 else ''}")
    print(f"paired pointing +gaze vs Temporal-only   : Δmean={dp1:+.4f} p={pp1:.3g} {'*' if pp1 < 0.05 else ''}")
    print(f"paired IoU      +gaze vs posshuffled: Δmed ={d2:+.4f} p={p2:.3g} {'*' if p2 < 0.05 else ''}")
    print(f"paired pointing +gaze vs posshuffled: Δmean={dp2:+.4f} p={pp2:.3g} {'*' if pp2 < 0.05 else ''}")

    print("\n--- STOPPING-RULE VERDICT (per-metric, no cherry-picking) ---")
    print(f"IoU:      +gaze vs Temporal-only {'WINS' if d1 > 0 and p1 < 0.05 else 'LOSES/no-sig'} "
          f"(Δ={d1:+.4f}, p={p1:.3g}) | vs shuffle "
          f"{'survives' if d2 > 0 and p2 < 0.05 else 'does NOT survive'} (Δ={d2:+.4f}, p={p2:.3g})")
    print(f"pointing: +gaze vs Temporal-only {'WINS' if dp1 > 0 and pp1 < 0.05 else 'LOSES/no-sig'} "
          f"(Δ={dp1:+.4f}, p={pp1:.3g}) | vs shuffle "
          f"{'survives' if dp2 > 0 and pp2 < 0.05 else 'does NOT survive'} (Δ={dp2:+.4f}, p={pp2:.3g})")
    print("Report both; do not silently pick the metric that passes. Human call needed if split.")


def _selfcheck():
    """Synthetic: ALL temporal/kinematic features are UNINFORMATIVE (identical
    across fixations), so Temporal-only (position-agnostic) is structurally blind and
    should sit near chance. Only POSITION distinguishes lesion-clustered fixations
    from scattered ones, so +gaze should win -- and position-shuffle should destroy
    that win, since shuffling removes the only signal that mattered."""
    import torch
    rng = np.random.default_rng(0); recs = []
    for k in range(150):
        N = 30
        tc = np.linspace(0, 10, N)           # uninformative: identical for every instance
        dur = np.full(N, 0.2); vel = np.zeros(N)
        lx, ly = rng.uniform(.3, .7, 2)
        x = rng.uniform(0, 1, N); y = rng.uniform(0, 1, N)
        near = rng.random(N) < 0.4            # only clue: some fixations cluster on the lesion
        x[near] = lx + rng.normal(0, .02, near.sum())
        y[near] = ly + rng.normal(0, .02, near.sum())
        fix = np.stack([x, y, tc, dur, vel], 1).astype(np.float32)
        recs.append({"rid": f"r{k}", "subject": f"s{k}", "fix": fix,
                     "labels": [{"label": "L", "ellipses": [(lx - .05, ly - .05, lx + .05, ly + .05)],
                                 "mentions": [], "mtext": []}]})
    p = Path("/tmp/_stage1_selfcheck.pt"); torch.save((recs, ["L"]), p)
    run(p, epochs=15)
    print("\nself-check ran (position is the ONLY signal -> expect +gaze to beat Temporal-only "
          "and beat position-shuffle)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="align.pt")
    ap.add_argument("--epochs", type=int, default=40)
    a = ap.parse_args()
    if Path(a.cache).exists():
        run(a.cache, a.epochs)
    else:
        _selfcheck()

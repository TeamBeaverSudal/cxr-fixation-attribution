"""Reading-level clustered significance for the five headline comparisons.

The paired Wilcoxon tests in evaluate.py run over test INSTANCES, but an
instance is a (reading, finding) pair: one reading contributes one instance per
annotated finding and all of them share that reading's single fixation sequence. On the
primary split 1093 instances come from 489 readings (419 subjects) and 86% of instances
sit in a reading contributing more than one, so the instance-level test treats
correlated units as independent. The paper bounds the damage with a design effect; this
runs the test at the reading level instead, two ways:

  1. cluster bootstrap -- resample READINGS with replacement, take every instance of
     each drawn reading, recompute the mean paired difference, percentile CI;
  2. reading-aggregated Wilcoxon -- average the paired difference within a reading
     first, then a signed-rank test over the per-reading values.

Subject-level (the coarser grouping: 419 subjects, some contributing several readings)
is reported too, as a strictly more conservative row.

Scoring is NOT reimplemented here: evaluate.run() is called and now returns
its per-instance arrays, so these p-values are computed on exactly the arrays the
paper's tables were computed on.

  uv run --with "numpy<2" --with pandas,scipy,scikit-learn,torch python cluster_stats.py
"""
import argparse

import numpy as np
from scipy.stats import wilcoxon

import evaluate as sfv

# The nine runs the paper reports: five training seeds on split 0, splits 1-4 at seed 0,
# overlapping at (0,0).
RUNS = [(0, s) for s in range(5)] + [(p, 0) for p in (1, 2, 3, 4)]

# (name, better, worse) -- all five are stated as "first beats second" in the paper.
COMPARISONS = [("ours_vs_B1", "fn", "b1"),
               ("ours_vs_temporal", "fn", "r1"),
               ("temporal_vs_B1", "r1", "b1"),
               ("ours_vs_posshuf", "fn", "shuf"),
               ("ours_vs_wordmask", "fn", "mask")]


def cluster_stats(d, g, nboot, seed):
    """d: per-instance paired differences. g: dense cluster id per instance.

    Returns the observed instance mean difference, the percentile CI, the fraction of
    resamples on the wrong side of zero, and the p-value of a signed-rank test over the
    per-cluster mean differences.
    """
    ncl = int(g.max()) + 1
    sums = np.bincount(g, weights=d, minlength=ncl)
    cnts = np.bincount(g, minlength=ncl).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, ncl, size=(nboot, ncl))
    boot = sums[idx].sum(1) / cnts[idx].sum(1)
    obs = float(d.mean())
    lo, hi = np.percentile(boot, [2.5, 97.5])
    cross = float((boot <= 0).mean() if obs > 0 else (boot >= 0).mean())
    per = sums / cnts                     # every cluster has >=1 instance by construction
    # zeros are dropped by scipy's default zero_method, same as the instance-level tests
    p = float(wilcoxon(per).pvalue) if np.any(per != 0) else 1.0
    # Clustered vs i.i.d. standard error of the SAME mean difference: their squared ratio
    # is the design effect the clustering actually costs, which is what the paper's
    # instance-weighted-mean-cluster-size argument only bounds from above.
    se_clus = float(boot.std(ddof=1))
    se_iid = float(d.std(ddof=1) / np.sqrt(len(d)))
    return obs, float(lo), float(hi), cross, p, ncl, se_clus, se_iid


def dense(keys, ok, of):
    """Cluster ids for the surviving instances, renumbered 0..ncl-1."""
    lab = [of(k) for k, m in zip(keys, ok) if m]
    u = {v: i for i, v in enumerate(dict.fromkeys(lab))}
    return np.array([u[v] for v in lab])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="align.pt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--boot-seed", type=int, default=20260809, dest="boot_seed")
    a = ap.parse_args()

    import torch
    recs, _ = torch.load(a.cache, weights_only=False)
    subj = {r["rid"]: r["subject"] for r in recs}

    print(f"cluster bootstrap: B={a.boot}, seed={a.boot_seed}, percentile CI")
    for split_seed, seed in RUNS:
        print(f"\n{'='*30} split={split_seed} seed={seed} {'='*30}", flush=True)
        res = sfv.run(a.cache, a.epochs, seed, None, split_seed)
        keys = res["keys"]
        rid_of, sub_of = (lambda k: k[0]), (lambda k: subj[k[0]])
        print(f"GROUPING split={split_seed} seed={seed} instances={len(keys)} "
              f"readings={len(set(map(rid_of, keys)))} subjects={len(set(map(sub_of, keys)))}")
        for metric in ("iou", "pg"):
            arr = res[metric]
            for name, hi_, lo_ in COMPARISONS:
                x, y = arr[hi_], arr[lo_]
                ok = np.isfinite(x) & np.isfinite(y)
                d = x[ok] - y[ok]
                p_inst = float(wilcoxon(x[ok], y[ok]).pvalue)
                for unit, of in (("reading", rid_of), ("subject", sub_of)):
                    g = dense(keys, ok, of)
                    obs, lo, hi, cross, p_cl, ncl, sec, sei = cluster_stats(
                        d, g, a.boot, a.boot_seed)
                    print(f"CLUS split={split_seed} seed={seed} unit={unit} cmp={name} "
                          f"metric={metric} n_inst={int(ok.sum())} n_clus={ncl} "
                          f"mean_d={obs:+.4f} ci=[{lo:+.4f},{hi:+.4f}] cross={cross:.5f} "
                          f"p_clus={p_cl:.3g} p_inst={p_inst:.3g} "
                          f"se_clus={sec:.5f} se_iid={sei:.5f} deff={(sec/sei)**2:.3f}",
                          flush=True)


def _selfcheck():
    """One runnable check on the only non-trivial logic here: the cluster resampling.
    Build data where every cluster is internally identical, so cluster-level and
    instance-level inference must agree, and data where a strong instance-level effect
    is carried by a handful of clusters, where they must not."""
    n = 200
    # (a) singleton clusters, effect present -> CI excludes 0, no crossings
    d = np.full(n, 0.1) + np.arange(n) * 0.0
    g = np.arange(n)
    obs, lo, hi, cross, p, _, sec, sei = cluster_stats(d, g, 2000, 0)
    assert abs(obs - 0.1) < 1e-9 and lo > 0 and cross == 0.0, (obs, lo, cross)
    # (b) same 200 numbers, but all in ONE cluster: nothing is replicated, so the
    # bootstrap has exactly one distinct resample and the signed-rank test has n=1.
    obs2, lo2, hi2, _, p2, ncl, _, _ = cluster_stats(d, np.zeros(n, int), 2000, 0)
    assert ncl == 1 and abs(lo2 - hi2) < 1e-12 and abs(obs2 - obs) < 1e-12
    # (c) effect concentrated in 2 of 100 clusters: instance mean is positive but the
    # cluster CI must include 0, which is the whole point of clustering.
    d = np.zeros(1000); d[:20] = 5.0
    g = np.repeat(np.arange(100), 10)
    obs3, lo3, hi3, cross3, _, _, sec3, sei3 = cluster_stats(d, g, 4000, 0)
    assert obs3 > 0 and lo3 <= 0 < hi3 and cross3 > 0.01, (obs3, lo3, hi3, cross3)
    assert sec3 > 2 * sei3, (sec3, sei3)   # clustering must inflate the SE here
    # dense() renumbers only surviving instances
    ks = [("a", "L1"), ("a", "L2"), ("b", "L1")]
    assert list(dense(ks, np.array([True, False, True]), lambda k: k[0])) == [0, 1]
    print("cluster_stats self-check OK")


if __name__ == "__main__":
    _selfcheck()
    main()

"""Check every number in BEAVER_FACR_MEDAI26.pdf against a run of the released code.

One pass, one report. Each block prints PASS/FAIL per value against the published table,
so what reproduces and what does not is visible without cross-reading logs.

    python verify_paper.py --cache align.pt                 # everything affordable
    python verify_paper.py --cache align.pt --only cohort,table1

Blocks are independent; `--only` runs a subset. The expensive ones (table3, stats, table5)
train the selector and are skipped unless named or `--all` is given.
"""
import argparse
import sys
import time

import numpy as np
from scipy.ndimage import zoom

sys.path.insert(0, "release")

from core import EVAL_RES, HEAT_RES, TUNE_SIGMAS, LOOKBACK, iou, pointing, raster, tune_thresholds, word_feat
from prior_and_swap import label_prior, prior_heat, split
from structured_baselines import b1_weights, modulate, scan_heat, word_dirs, _wsplat

TS = tune_thresholds()
TOL = 5e-4          # published values carry four decimals

# ---------------------------------------------------------------- published values
PAPER = {
    "cohort": {"test": 987, "val": 547, "train": 1895, "exact": 948},
    "table1": [                                  # (row, pointing, iou)
        ("Complete-scanpath density",        0.4063, 0.2008),
        ("1.5-s temporal baseline",          0.6211, 0.2773),
        ("Anatomical prior",                 0.5035, 0.2736),
        ("Prior x scanpath support",         0.6717, 0.3056),
        ("  + directional terms",            0.7183, 0.3408),
        ("Prior x scanpath + temporal gate", 0.7528, 0.3201),
        ("1.5-s combined structured",        0.7923, 0.3439),
    ],
    "lookback3": ("3.0-s combined structured", 0.7893, 0.3549),
    "table2": [("Target record", 0.8333, 0.3604),
               ("Matched other-patient records", 0.5527, 0.2849)],
    "table3": [("Full-query selector",                 0.8245, 0.3584),
               ("Four-indicator selector",             0.8373, 0.3574),
               ("Finding + temporal/kinematic",        0.7295, 0.3135),
               ("Position features permuted",          0.5534, 0.2306),
               ("Temporal/kinematic features permuted", 0.6833, 0.2976),
               ("Spatial indicators masked",           0.7495, 0.3068)],
    "table4": {                                  # finding: (n, d_pointing, d_iou)
        "Atelectasis":                 (185, +0.0184, +0.0188),
        "Consolidation":               (160, -0.0288, -0.0032),
        "Enlarged cardiac silhouette": (147, +0.0517, +0.0796),
        "Groundglass opacity":         (100, -0.0600, +0.0004),   # paper prints "Ground-glass"
        "Lung nodule or mass":          (28, +0.1357, -0.0120),
        "Pleural abnormality":         (210, +0.1314, +0.0118),
        "Pulmonary edema":              (90, -0.0178, -0.0021),
    },
    "table4_macro": (+0.028, -0.011),            # unweighted, all 14 findings
    "stats": {"pg":  (0.0405,  0.0108, 0.0696, 0.040),
              "iou": (0.0036, -0.0039, 0.0108, 0.665)},
    "table5": {                                  # fraction: (learn_pg, str_pg, learn_iou, str_iou)
        "10%": (0.776, 0.772, 0.321, 0.335),
        "25%": (0.806, 0.786, 0.333, 0.341),
        "50%": (0.816, 0.792, 0.339, 0.342),
        # one seed-0 run, not a five-subsample mean
        "100%": (0.830, 0.792, 0.358, 0.344),
    },
}

_fails, _passes, _skips = [], [], []


def chk(block, name, got, want, tol=TOL):
    if got is None:
        _skips.append(f"{block}: {name}")
        print(f"  SKIP  {name:38s} (not computed)")
        return
    ok = abs(got - want) <= tol
    (_passes if ok else _fails).append(f"{block}: {name}")
    d = got - want
    print(f"  {'PASS' if ok else 'FAIL'}  {name:38s} {got:.4f}  paper {want:.4f}"
          f"{'' if ok else f'   delta {d:+.4f}'}")


# ---------------------------------------------------------------- shared data
def load(cache, split_seed):
    import torch
    recs, labels = torch.load(cache, weights_only=False)
    li = {l: i for i, l in enumerate(labels)}
    part = split(recs, split_seed)

    def insts(p):
        return [(r["fix"], d["mentions"], d["ellipses"], li[d["label"]],
                 word_feat(d.get("mtext", [])), d["label"], r["rid"])
                for r in recs if part(r) == p for d in r["labels"]]

    return recs, labels, insts("train"), insts("val"), insts("test")


def _stamp(a):
    """Cache name plus a hash of the scoring code, so a checkpoint written by an earlier
    version is never silently reused. Editing selector.py or evaluate.py changes the name."""
    import hashlib
    from pathlib import Path
    h = hashlib.sha256()
    for f in ("selector.py", "evaluate.py", "core.py"):
        for d in (Path("release") / f, Path(f)):
            if d.exists():
                h.update(d.read_bytes())
                break
    return f"{Path(a.cache).stem}_{h.hexdigest()[:8]}"


def resolved(items):
    return [it for it in items if len(it[1])]


# ---------------------------------------------------------------- blocks
def blk_cohort(ctx):
    print("\n== cohort ==")
    tr, va, te = ctx["tr"], ctx["va"], ctx["te"]
    p = PAPER["cohort"]
    for nm, items, want in (("test mention-resolved", te, p["test"]),
                            ("val mention-resolved", va, p["val"]),
                            ("train mention-resolved", tr, p["train"])):
        chk("cohort", nm, float(len(resolved(items))), float(want), tol=0.5)
        print(f"        ({len(items)} annotated -> {len(resolved(items))} resolved)")

    # Table II runs on "exact-match" instances. The paper does not define the term in the
    # methods; the only per-instance filter in the code is a single-ellipse exact hit.
    print(f"  NOTE  Table II's n={p['exact']} 'exact match' cohort is not derivable here; "
          "see the table2 block.")


def blk_table1(ctx):
    print("\n== Table I, rule-based rows (prior on mention-resolved train) ==")
    tr_m, va, te_m = resolved(ctx["tr"]), ctx["va"], resolved(ctx["te"])
    prior = label_prior((it[2], it[5]) for it in tr_m)
    dirs = word_dirs(tr_m)

    def masks(items, words):
        out = []
        for it in items:
            m = prior.get(it[5])
            out.append(modulate(m, it[4], dirs) if (m is not None and words) else m)
        return out

    VARIANTS = [
        (lambda i, it, mk, sg: _wsplat(it[0], it[0][:, 3], sg), False),
        (lambda i, it, mk, sg: _wsplat(it[0], b1_weights(it[0], it[1]), sg), False),
        (lambda i, it, mk, sg: prior_heat(mk[i], sg), False),
        (lambda i, it, mk, sg: scan_heat(it[0], mk[i], sg), False),
        (lambda i, it, mk, sg: scan_heat(it[0], mk[i], sg), True),
        (lambda i, it, mk, sg: _wsplat(it[0], b1_weights(it[0], it[1], mk[i]), sg), False),
        (lambda i, it, mk, sg: _wsplat(it[0], b1_weights(it[0], it[1], mk[i]), sg), True),
    ]
    for (name, wp, wi), (fn, words) in zip(PAPER["table1"], VARIANTS):
        mk_v = masks(va, words)
        # the two prior-only rows search sigma=0 as well, matching the published protocol
        sigmas = [0.0] + list(TUNE_SIGMAS) if "prior" in name.lower() and "scan" not in name.lower() \
            else TUNE_SIGMAS
        sg, th = _tune(fn, va, mk_v, sigmas)
        i_, p_ = _score(fn, te_m, masks(te_m, words), sg, th)
        print(f"  [sigma={sg} thr={th:.3f}]")
        chk("table1", name + " (pg)", p_, wp)
        chk("table1", name + " (IoU)", i_, wi)


def _tune(fn, items, mk, sigmas):
    best = (sigmas[0], TS[0], -1.0)
    for sg in sigmas:
        acc, cnt = np.zeros(len(TS)), np.zeros(len(TS))
        for i, it in enumerate(items):
            h = fn(i, it, mk, sg)
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


def _score(fn, items, mk, sg, th):
    io, pg = [], []
    for i, it in enumerate(items):
        h = fn(i, it, mk, sg)
        up = zoom(h, EVAL_RES / h.shape[0], order=0)
        g = raster(it[2])
        io.append(iou(up, g, th)); pg.append(pointing(up, g))
    return float(np.nanmean(io)), float(np.nanmean(pg))


def blk_lookback3(ctx):
    """The 3.0-s row. Reachable only if the cache carries sentence boundaries."""
    print("\n== 3.0-s combined structured baseline ==")
    name, wp, wi = PAPER["lookback3"]
    has_sents = all("sents" in r for r in ctx["recs"][:20])
    if not has_sents:
        clip = [g > s - LOOKBACK + 1e-6 for it in ctx["te"] for g, s, e, *_ in it[1]]
        print(f"  BLOCKED  the cache stores gate_start = max(sent_start - {LOOKBACK}, prev_start).")
        print(f"           Where the clip did not bind ({100 * (1 - np.mean(clip)):.1f}% of "
              f"{len(clip)} test mentions) prev_start is unrecoverable, so no window past "
              f"{LOOKBACK}s can be rebuilt.")
        print("           Fix: store sentence starts at extraction, then re-extract.")
        print("             core.py, in the per-record dict:   \"sents\": sents,")
        chk("lookback3", name + " (pg)", None, wp)
        chk("lookback3", name + " (IoU)", None, wi)
        return

    tr_m, va, te_m = resolved(ctx["tr"]), ctx["va"], resolved(ctx["te"])
    prior = label_prior((it[2], it[5]) for it in tr_m)
    dirs = word_dirs(tr_m)
    starts = {r["rid"]: [s for s, _e in r["sents"]] for r in ctx["recs"]}

    def regate(it, d):
        ss = starts.get(it[6])
        out = []
        for g, s, e, *rest in it[1]:
            prev = max([x for x in (ss or []) if x < s - 1e-6], default=s)
            out.append((max(min(s - d, g), prev), s, e, *rest))
        return out

    fn = lambda i, it, mk, sg, d=None: _wsplat(
        it[0], b1_weights(it[0], regate(it, d), mk[i]), sg)

    def masks(items):
        return [modulate(prior.get(it[5]), it[4], dirs) if prior.get(it[5]) is not None else None
                for it in items]

    best = None
    for d in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
        f = lambda i, it, mk, sg, d=d: fn(i, it, mk, sg, d)
        sg, th = _tune(f, va, masks(va), TUNE_SIGMAS)
        vi, _ = _score(f, va, masks(va), sg, th)
        ti, tp = _score(f, te_m, masks(te_m), sg, th)
        print(f"  d={d:<4} val_iou={vi:.4f}  test {tp:.4f}/{ti:.4f}")
        if best is None or vi > best[0]:
            best = (vi, d, tp, ti)
    print(f"  validation selects d={best[1]}")
    chk("lookback3", name + " (pg)", best[2], wp)
    chk("lookback3", name + " (IoU)", best[3], wi)

    # The paper's inferential comparison is against this window, not the prespecified 1.5-s
    # one: its reported IoU difference of 0.0036 is 0.3584 - 0.3549, and its pointing
    # difference of 0.0405 is 0.0375 + (0.7923 - 0.7893).
    f = lambda i, it, mk, sg, d=best[1]: fn(i, it, mk, sg, d)
    sg, th = _tune(f, va, masks(va), TUNE_SIGMAS)
    mk_t, arr = masks(te_m), {"pg": [], "iou": []}
    for i, it in enumerate(te_m):
        up = zoom(f(i, it, mk_t, sg), EVAL_RES / HEAT_RES, order=0)
        g = raster(it[2])
        arr["iou"].append(iou(up, g, th)); arr["pg"].append(pointing(up, g))
    ctx["base3"] = {m: np.asarray(v) for m, v in arr.items()}


def blk_table3(ctx, a):
    """Table I's learned row and all of Table III: five seeds through evaluate.run."""
    print("\n== Table I learned row + Table III (five seeds) ==")
    import pickle
    from pathlib import Path

    import evaluate
    # Each seed costs ~15 min and the shared node kills long jobs, so keep every finished seed
    # on disk. A rerun then picks up where it stopped instead of paying for the whole sweep.
    runs = []
    for s in range(a.seeds):
        ck = Path(f"runs_{_stamp(a)}_split{a.split_seed}_seed{s}.pkl")
        if ck.exists():
            runs.append(pickle.loads(ck.read_bytes()))
            print(f"  seed {s} loaded from {ck}", flush=True)
            continue
        t0 = time.time()
        r = evaluate.run(a.cache, a.epochs, s, None, a.split_seed)
        ck.write_bytes(pickle.dumps(r))
        runs.append(r)
        print(f"  seed {s} done in {time.time() - t0:.0f}s -> {ck}", flush=True)
    ctx["runs"] = runs

    # Table III is reported on the 987 mention-resolved instances, but evaluate.run returns
    # arrays over every annotated test instance. Restrict by key before averaging.
    # evaluate.py's sixth field is a key tuple while ours is the label, so match positionally:
    # both build the test list from the same records under the same split, in the same order.
    assert len(runs[0]["keys"]) == len(ctx["te"]), "test ordering differs from evaluate.run"
    sel = np.array([len(it[1]) > 0 for it in ctx["te"]])
    ctx["sel"] = sel
    print(f"  restricting to {int(sel.sum())} of {len(sel)} test instances")

    def mean(metric, key):
        return float(np.mean([np.nanmean(np.asarray(r[metric][key])[sel]) for r in runs]))

    KEYS = {"Full-query selector": "fn", "Finding + temporal/kinematic": "r1",
            "Position features permuted": "shuf", "Spatial indicators masked": "mask",
            "Four-indicator selector": "q4",
            "Temporal/kinematic features permuted": "tshuf"}
    for name, wp, wi in PAPER["table3"]:
        k = KEYS.get(name)
        avail = k in runs[0]["pg"] if k else False
        chk("table3", name + " (pg)", mean("pg", k) if avail else None, wp)
        chk("table3", name + " (IoU)", mean("iou", k) if avail else None, wi)


def blk_table4(ctx):
    """Per-finding differences, learned minus the 1.5-s combined baseline."""
    print("\n== Table IV, finding-level differences ==")
    runs = ctx.get("runs")
    if not runs:
        print("  SKIP  needs the table3 block (five-seed learned run)")
        for f in PAPER["table4"]:
            _skips.append(f"table4: {f}")
        return
    sel = ctx["sel"]
    te_m = resolved(ctx["te"])
    labs = np.array([it[5] for it in te_m])

    # the learned side: five-seed mean per instance, on the mention-resolved cohort
    learn = {m: np.mean([np.asarray(r[m]["fn"])[sel] for r in runs], axis=0)
             for m in ("pg", "iou")}

    # the 1.5-s combined structured baseline, per instance, same cohort and calibration path
    tr_m, va = resolved(ctx["tr"]), ctx["va"]
    prior = label_prior((it[2], it[5]) for it in tr_m)
    dirs = word_dirs(tr_m)

    def masks(items):
        return [modulate(prior[it[5]], it[4], dirs) if it[5] in prior else None for it in items]

    fn = lambda i, it, mk, sg: _wsplat(it[0], b1_weights(it[0], it[1], mk[i]), sg)
    sg, th = _tune(fn, va, masks(va), TUNE_SIGMAS)
    base = {"pg": [], "iou": []}
    mk_t = masks(te_m)
    for i, it in enumerate(te_m):
        up = zoom(fn(i, it, mk_t, sg), EVAL_RES / HEAT_RES, order=0)
        g = raster(it[2])
        base["iou"].append(iou(up, g, th)); base["pg"].append(pointing(up, g))
    base = {m: np.asarray(v) for m, v in base.items()}
    ctx["base"] = base                       # the stats block compares against this exact array

    diffs = {}
    for f in np.unique(labs):
        m = labs == f
        diffs[f] = (int(m.sum()),
                    float(np.nanmean(learn["pg"][m] - base["pg"][m])),
                    float(np.nanmean(learn["iou"][m] - base["iou"][m])))
    for f, (n, dp, di) in sorted(diffs.items(), key=lambda kv: -kv[1][0]):
        print(f"  {f:34s} n={n:4d}  d_pg={dp:+.4f}  d_iou={di:+.4f}")

    for f, (n, dp, di) in PAPER["table4"].items():
        got = diffs.get(f)
        chk("table4", f"{f} (n)", None if got is None else float(got[0]), float(n), tol=0.5)
        chk("table4", f"{f} (d pg)", None if got is None else got[1], dp)
        chk("table4", f"{f} (d IoU)", None if got is None else got[2], di)

    big = [v for v in diffs.values()]
    chk("table4", "macro d pg (14 findings)",
        float(np.mean([v[1] for v in big])), PAPER["table4_macro"][0], tol=5e-3)
    chk("table4", "macro d IoU (14 findings)",
        float(np.mean([v[2] for v in big])), PAPER["table4_macro"][1], tol=5e-3)
    print(f"        (macro over {len(big)} findings)")


def blk_stats(ctx, a):
    """The prespecified seed-0 inferential comparison: learned vs the 1.5-s combined baseline.

    cluster_stats.py cannot produce this. Its five comparisons pit the learned selector against
    the temporal baseline, the reduced-input model and the two perturbations -- never against
    the combined structured baseline the paper reports -- and it scores every annotated test
    instance rather than the 987. Both quantities are already in hand here, so the statistic is
    computed directly instead of retraining nine selectors to get the wrong contrast.
    """
    print("\n== Clustered statistics (seed 0, split 0) ==")
    from scipy.stats import wilcoxon

    from cluster_stats import cluster_stats as cboot

    runs = ctx.get("runs")
    # the validation-selected window is the paper's comparison baseline; fall back to the
    # prespecified one only to keep the block runnable without the lookback3 result
    base = ctx.get("base3") or ctx.get("base")
    print(f"  baseline: {'3.0-s (validation-selected)' if ctx.get('base3') else '1.5-s'}")
    if not runs or base is None:
        print("  SKIP  needs the table3 block and either lookback3 or table4")
        for m in ("pg", "iou"):
            _skips.append(f"stats: {m}")
        return

    sel = ctx["sel"]
    te_m = resolved(ctx["te"])
    subj = {r["rid"]: r["subject"] for r in ctx["recs"]}
    pat = np.array([subj[it[6]] for it in te_m])

    for metric, key in (("pg", "pg"), ("iou", "iou")):
        # seed 0 alone: the paper calls this comparison prespecified, and the five-seed mean
        # is a different estimator with a different variance.
        learn = np.asarray(runs[0][key]["fn"])[sel]
        d = learn - base[metric]
        ok = np.isfinite(d)
        uniq, gi = np.unique(pat[ok], return_inverse=True)
        obs, lo, hi, cross, p_cl, ncl, sec, sei = cboot(d[ok], gi, a.boot, 20260809)
        # the paper's sensitivity test is over patient-level mean differences, not instances
        per_patient = np.array([d[ok][gi == k].mean() for k in range(len(uniq))])
        p_w = float(wilcoxon(per_patient).pvalue)
        wd, wlo, whi, wp = PAPER["stats"][metric]
        print(f"  n_inst={int(ok.sum())} n_patients={ncl}")
        chk("stats", f"{metric} mean difference", float(obs), wd)
        chk("stats", f"{metric} CI low", float(lo), wlo, tol=1e-3)
        chk("stats", f"{metric} CI high", float(hi), whi, tol=1e-3)
        chk("stats", f"{metric} Wilcoxon p", p_w, wp, tol=5e-3)


def blk_table2(ctx, a):
    """Record substitution, from the seed-0 selector's exact-match donor groups."""
    print("\n== Table II, record substitution ==")
    runs = ctx.get("runs")
    sub = runs[0].get("subst") if runs else None
    if sub is None:
        print("  SKIP  needs the table3 block (seed-0 run)")
        for name, wp, wi in PAPER["table2"]:
            _skips.append(f"table2: {name}")
        return
    chk("table2", "exact-match instances", float(sub["n"]), float(PAPER["cohort"]["exact"]),
        tol=0.5)
    for name, wp, wi in PAPER["table2"]:
        pre = "target" if name == "Target record" else "donor"
        chk("table2", name + " (pg)", sub[f"{pre}_pg"], wp)
        chk("table2", name + " (IoU)", sub[f"{pre}_iou"], wi)


def blk_table5(ctx, a):
    """Training-set-size sensitivity: five nested patient subsample chains per fraction."""
    print("\n== Table V, training-set size ==")
    import pickle
    from pathlib import Path

    import evaluate
    sel = ctx.get("sel")
    if sel is None:
        sel = np.array([len(it[1]) > 0 for it in ctx["te"]])
    te_m = resolved(ctx["te"])
    va = ctx["va"]

    # structured side: the prior is retrained on the same patient subsample, so the two
    # columns answer the same question about how much annotated training data each needs.
    recs = ctx["recs"]
    part = split(recs, a.split_seed)
    sub_of = {r["rid"]: r["subject"] for r in recs}
    tr_m = resolved(ctx["tr"])
    # "eligible" is the patients that contribute a mention-resolved training instance, which
    # is what the fractions are taken of -- not every patient in the training split.
    tp = np.array(sorted({sub_of[it[6]] for it in tr_m}))
    chk("table5", "eligible training patients", float(len(tp)), 735.0, tol=0.5)

    def structured(keep):
        sub = [it for it in tr_m if sub_of.get(it[6]) in keep]
        if not sub:
            return None, None
        prior = label_prior((it[2], it[5]) for it in sub)
        dirs = word_dirs(sub)
        mk = lambda items: [modulate(prior[it[5]], it[4], dirs) if it[5] in prior else None
                            for it in items]
        fn = lambda i, it, m, sg: _wsplat(it[0], b1_weights(it[0], it[1], m[i]), sg)
        sg, th = _tune(fn, va, mk(va), TUNE_SIGMAS)
        i_, p_ = _score(fn, te_m, mk(te_m), sg, th)
        return p_, i_

    for frac, key in ((0.10, "10%"), (0.25, "25%"), (0.50, "50%"), (1.00, "100%")):
        lp, li_, sp, si = [], [], [], []
        for chain in range(1 if key == "100%" else 5):
            ck = Path(f"t5_{_stamp(a)}_f{int(frac * 100)}_c{chain}.pkl")
            if ck.exists():
                r, s = pickle.loads(ck.read_bytes())
            else:
                t0 = time.time()
                r = evaluate.run(a.cache, a.epochs, 0, None, a.split_seed,
                                 train_frac=frac, chain_seed=chain)
                order = tp.copy()
                np.random.default_rng(chain).shuffle(order)
                s = structured(set(order[:max(1, int(round(frac * len(order))))]))
                ck.write_bytes(pickle.dumps((r, s)))
                print(f"  {key} chain {chain} done in {time.time() - t0:.0f}s", flush=True)
            lp.append(np.nanmean(np.asarray(r["pg"]["fn"])[sel]))
            li_.append(np.nanmean(np.asarray(r["iou"]["fn"])[sel]))
            if s[0] is not None:
                sp.append(s[0]); si.append(s[1])
        w = PAPER["table5"][key]
        # The partial rows average five patient subsamples whose draw is not recorded, and the
        # table prints three decimals, so a different chain lands nearby rather than on the
        # value. The 100% row is a single deterministic run and is held to the usual precision.
        tol = 5e-4 if key == "100%" else 1e-2
        chk("table5", f"{key} learned pg", float(np.mean(lp)), w[0], tol=tol)
        chk("table5", f"{key} structured pg", float(np.mean(sp)) if sp else None, w[1], tol=tol)
        chk("table5", f"{key} learned IoU", float(np.mean(li_)), w[2], tol=tol)
        chk("table5", f"{key} structured IoU", float(np.mean(si)) if si else None, w[3], tol=tol)


def blk_misc(ctx, a):
    """Numbers stated in the prose rather than in a table."""
    print("\n== prose values ==")
    print(f"  test readings/patients: paper says 489 readings from 419 patients")
    recs = ctx["recs"]
    part = split(recs, a.split_seed)
    te_r = [r for r in recs if part(r) == "test"]
    chk("misc", "test readings", float(len(te_r)), 489.0, tol=0.5)
    chk("misc", "test patients", float(len({r["subject"] for r in te_r})), 419.0, tol=0.5)

    # Fig. 2 quotes four per-instance IoU values. They identify no instance, so the check is
    # existence: does any test instance reach them under the two methods the figure contrasts?
    runs = ctx.get("runs")
    if not runs:
        print("  SKIP  Fig. 2 IoUs need the table3 block")
        for v in (0.153, 0.274, 0.297, 0.205):
            _skips.append(f"misc: Fig.2 IoU {v}")
        return
    sel = ctx["sel"]
    learn = np.mean([np.asarray(r["iou"]["fn"])[sel] for r in runs], axis=0)
    for v in (0.274, 0.205):
        near = int(np.sum(np.abs(learn - v) < 5e-4))
        print(f"  Fig.2 learned IoU {v}: {near} test instance(s) within 0.0005")
    print("  NOTE  the structured-side values (0.153, 0.297) need the per-instance baseline "
          "array, which blk_table4 computes; wire them together if the figure must be checked.")
    print("  NOTE  K = 8 (from {1,2,4,8}) belongs to the record-substitution control, which is "
          "the code the table2 block reports missing.")


# table2 reads the seed-0 run, so it has to follow table3; --all walks this order.
BLOCKS = {"cohort": blk_cohort, "table1": blk_table1, "lookback3": blk_lookback3,
          "table3": blk_table3, "table2": blk_table2, "table4": blk_table4,
          "table5": blk_table5, "stats": blk_stats, "misc": blk_misc}
CHEAP = ("cohort", "table1", "lookback3")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="align.pt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--split-seed", type=int, default=0, dest="split_seed")
    ap.add_argument("--only", default="", help="comma-separated block names")
    ap.add_argument("--all", action="store_true", help="include the blocks that train")
    a = ap.parse_args()

    names = [n.strip() for n in a.only.split(",") if n.strip()] or \
        (list(BLOCKS) if a.all else list(CHEAP))
    bad = [n for n in names if n not in BLOCKS]
    if bad:
        sys.exit(f"unknown block(s): {', '.join(bad)}  (choose from {', '.join(BLOCKS)})")

    recs, labels, tr, va, te = load(a.cache, a.split_seed)
    ctx = {"recs": recs, "labels": labels, "tr": tr, "va": va, "te": te}

    for n in names:
        fn = BLOCKS[n]
        t0 = time.time()
        (fn(ctx, a) if fn.__code__.co_argcount == 2 else fn(ctx))
        print(f"  [{n}: {time.time() - t0:.0f}s]", flush=True)

    print(f"\n{'=' * 60}\n{len(_passes)} pass, {len(_fails)} fail, {len(_skips)} not computed")
    for f in _fails:
        print(f"  FAIL  {f}")
    for s in _skips:
        print(f"  ----  {s}")
    sys.exit(1 if _fails else 0)


if __name__ == "__main__":
    main()

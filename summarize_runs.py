"""Compact 9-run summary of structured_baselines.py's log.

The script prints one PSSUM line per (run, subset) and one PSCMP line per
(run, subset, comparison). This folds them into two tables: per-method means
with the across-run range, and every comparison against the shipped model.

    python summarize_runs.py [ps.log] [--sub all|ment|noment]
"""
import statistics as st
import sys

path = next((a for a in sys.argv[1:] if not a.startswith("-")), "ps.log")
sub = "all"
if "--sub" in sys.argv:
    sub = sys.argv[sys.argv.index("--sub") + 1]


def kv(line):
    d = dict(p.split("=", 1) for p in line.split() if "=" in p)
    d["_tag"] = line.split(None, 1)[0]
    return d


rows = [kv(l) for l in open(path) if l.startswith(("PSSUM ", "PSCMP "))
        and f" sub={sub} " in l]
S = [r for r in rows if "b1_iou" in r]
C = [r for r in rows if "d_iou" in r]
if not S:
    sys.exit(f"{path}: no PSSUM lines for sub={sub} -- run not finished?")


def band(v):
    return f"{st.mean(v):8.4f} [{min(v):.3f},{max(v):.3f}]"


print(f"\n=== sub={sub}, {len(S)} runs, n={S[0]['n']} ===")
print(f"{'method':14s} {'IoU':>19s} {'pointing':>19s}")
for m in [k[:-4] for k in S[0] if k.endswith("_iou")]:
    print(f"{m:14s} {band([float(r[m + '_iou']) for r in S])} "
          f"{band([float(r[m + '_pg']) for r in S])}")

print("\n=== vs FINAL (negative = the shipped model wins) ===")
for a in sorted({r["a"] for r in C if r["b"] == "FINAL"}):
    r = [x for x in C if x["a"] == a and x["b"] == "FINAL"]
    sig = lambda f: sum(float(x[f]) < .05 for x in r)          # noqa: E731
    print(f"{a:14s} IoU {band([float(x['d_iou']) for x in r])} sig {sig('p_iou')}/{len(r)}"
          f"   pg {band([float(x['d_pg']) for x in r])} sig {sig('p_pg')}/{len(r)}")

# --- fixation-mass diagnostic ---------------------------------------------------------
fx = [kv(l) for l in open(path) if l.startswith("FIXMASS ")]
if fx:
    print("\n=== fixation mass inside the ellipse (untuned; FINAL trains on a related "
          "objective -- behaviour, not ranking) ===")
    print(f"{'method':10s} {'mean':>19s} {'median':>19s}")
    for m in ("B1", "B1P", "B1PW", "FINAL"):
        r = [x for x in fx if x["method"] == m]
        if r:
            print(f"{m:10s} {band([float(x['mean']) for x in r])} "
                  f"{band([float(x['median']) for x in r])}")

# --- B1 / B1PW temporal-offset sweep -------------------------------------------------
# Only the per-delta lines are parsed. The BEST lines carry a trailing "(shipped 1.5 ->
# iou=...)" whose k=v pairs would overwrite the row's own iou/pg, so the winner is taken
# from the curve instead, on validation IoU, exactly as the script selects it.
sweep = [kv(l) for l in open(path) if l.startswith(("B1OFF ", "B1PWOFF "))]
if sweep:
    for tag, name in (("B1OFF", "B1"), ("B1PWOFF", "B1PW")):
        rows = [r for r in sweep if r["_tag"] == tag]
        if not rows:
            continue
        splits = sorted({r["split"] for r in rows})
        print(f"\n=== {name} offset sweep, {len(splits)} split(s) ===")
        print(f"{'delta':>6s} {'val IoU':>19s} {'test IoU':>19s} {'test pointing':>19s}")
        curve = []
        for d in sorted({float(r["delta"]) for r in rows}):
            g = [r for r in rows if float(r["delta"]) == d]
            v = [float(r["val_iou"]) for r in g]
            print(f"{d:6.2f} {band(v)} {band([float(r['iou']) for r in g])} "
                  f"{band([float(r['pg']) for r in g])}")
            curve.append((st.mean(v), d, st.mean([float(r["iou"]) for r in g]),
                          st.mean([float(r["pg"]) for r in g])))
        bv, bd, bi, bp = max(curve)
        sh = [c for c in curve if c[1] == 1.5]
        tail = (f"; shipped 1.5 -> {sh[0][2]:.4f}/{sh[0][3]:.4f}" if sh else "")
        print(f"  best on validation: delta={bd} -> {bi:.4f} IoU / {bp:.4f} pointing{tail}")
        if bd in (min(c[1] for c in curve), max(c[1] for c in curve)):
            print(f"  !! delta={bd} sits on a grid boundary")
else:
    print("\n(no offset-sweep lines in the log)")

if len(S) != 9:
    print(f"\n!! {len(S)} runs, expected 9 -- log is incomplete.")

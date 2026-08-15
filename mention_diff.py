"""Scope the enlarged-cardiac-silhouette pattern fix before rerunning anything.

The original pattern was `cardiomegaly|enlarged.*(cardiac|heart)|heart.*enlarg`, which misses
"the cardiac silhouette is enlarged" -- the commonest phrasing -- because `enlarged.*` demands
the wrong word order and `heart.*enlarg` demands the token "heart".

Fixing a matcher is not a free per-label re-score. The matched sentence supplies the spatial
words in the query, the offsets that drive every temporal feature, and B1's gate, so a changed
assignment changes training inputs. This script measures how much changes, before any rerun:

  1. the audit on the 200 hand-annotated readings, old pattern vs new, so newly recovered
     mentions can be checked against ground truth and new false positives counted;
  2. the assignment diff over every cached reading, split by train/val/test, counting not only
     unmatched -> matched but matched -> matched with a different sentence set, which is the
     case that silently changes an existing training input.

    python mention_diff.py /path/to/reflacx --cache align.pt [--gt manually_labeled_reports_3.csv]
"""
import argparse
import ast
import re
from pathlib import Path

import pandas as pd

from linker_and_temporal import KEYWORDS, NEG, sentences_with_times
from prior_and_swap import split as make_split

OLD_CARDIAC = r"cardiomegaly|enlarged.*(cardiac|heart)|heart.*enlarg"
CARDIAC = "Enlarged cardiac silhouette"


def matched(sents, pat):
    """The sentence texts our matcher assigns to a label under `pat`."""
    rx = re.compile(pat, re.I)
    return tuple(t for t, _s, _e in sents if rx.search(t) and not NEG.search(t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--cache", default="align.pt")
    ap.add_argument("--gt", default="manually_labeled_reports_3.csv")
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--examples", type=int, default=0,
                    help="print this many newly matched sentences per split")
    a = ap.parse_args()

    new_pat = KEYWORDS[CARDIAC]
    if new_pat == OLD_CARDIAC:
        raise SystemExit("KEYWORDS still holds the old pattern -- nothing to diff")

    tr = {p.parent.name: p for p in Path(a.root).rglob("timestamps_transcription.csv")}
    import torch
    recs, _ = torch.load(a.cache, weights_only=False)
    part = make_split(recs, a.split_seed)
    sents = {}

    def sents_of(rid):
        if rid not in sents and rid in tr:
            sents[rid] = sentences_with_times(pd.read_csv(tr[rid]))
        return sents.get(rid)

    # ---- 1. audit on the hand-annotated readings, old vs new -------------------------------
    gt = pd.read_csv(a.gt).set_index("IDs")
    print(f"=== audit on {len(gt)} hand-annotated readings, {CARDIAC} only ===")
    for name, pat in (("old", OLD_CARDIAC), ("new", new_pat)):
        tp = fp = fn = 0
        for rid in gt.index:
            s = sents_of(rid)
            if s is None:
                continue
            truth = bool(ast.literal_eval(str(gt.loc[rid, CARDIAC.lower() + "_location"]) or "[]"))
            hit = bool(matched(s, pat))
            tp += hit and truth
            fp += hit and not truth
            fn += truth and not hit
        p_ = tp / (tp + fp) if tp + fp else float("nan")
        r_ = tp / (tp + fn) if tp + fn else float("nan")
        print(f"  {name:3s}  TP {tp:3d}  FP {fp:3d}  FN {fn:3d}   "
              f"precision {p_:.3f}  recall {r_:.3f}")

    # newly recovered: did the fix pick up real mentions, or invent them?
    good = bad = 0
    for rid in gt.index:
        s = sents_of(rid)
        if s is None:
            continue
        if matched(s, OLD_CARDIAC) or not matched(s, new_pat):
            continue
        truth = bool(ast.literal_eval(str(gt.loc[rid, CARDIAC.lower() + "_location"]) or "[]"))
        good += truth; bad += not truth
    print(f"  newly matched by the fix: {good + bad}  "
          f"(annotated in ground truth: {good}, not annotated: {bad})")

    # ---- 2. assignment diff over the whole cache, by split ---------------------------------
    kinds = ("unmatched->matched", "matched->unmatched", "sentences changed")
    tally = {s: dict.fromkeys(kinds, 0) for s in ("train", "val", "test")}
    total = {s: 0 for s in tally}
    notx = {s: 0 for s in tally}      # instances whose transcript was not found: in `total`
    ex = {s: [] for s in tally}       # but never in `tally`, so the two must be read together
    for r in recs:
        s = sents_of(r["rid"])
        pt = part(r)
        for d in r["labels"]:
            if d["label"] != CARDIAC:
                continue
            total[pt] += 1
            if s is None:
                notx[pt] += 1
                continue
            o, n = matched(s, OLD_CARDIAC), matched(s, new_pat)
            if o == n:
                continue
            k = ("unmatched->matched" if not o else
                 "matched->unmatched" if not n else "sentences changed")
            tally[pt][k] += 1
            if k == "unmatched->matched":
                ex[pt].append((r["rid"], n[0]))

    print(f"\n=== assignment diff over the cache, {CARDIAC} instances ===")
    print(f"{'split':6s} {'instances':>10s} {'no transcript':>14s} "
          + " ".join(f"{k:>19s}" for k in kinds))
    for s in ("train", "val", "test"):
        print(f"{s:6s} {total[s]:10d} {notx[s]:14d} "
              + " ".join(f"{tally[s][k]:>19d}" for k in kinds))
    changed = sum(sum(v.values()) for v in tally.values())
    print(f"\n{changed} of {sum(total.values())} instances change assignment. "
          f"{'Training inputs move: rerun the full pipeline.' if tally['train']['unmatched->matched'] or tally['train']['sentences changed'] or tally['train']['matched->unmatched'] else 'No training instance changes.'}")
    print("Only this label's patterns differ, so no other label's assignment can move.")

    # Is the cache itself stale? Both columns above are recomputed from the raw transcripts, so
    # they are identical whether or not align.pt was re-extracted. This is the check that
    # actually reads what the cache stores.
    old_like = new_like = 0
    for r in recs:
        s = sents_of(r["rid"])
        if s is None:
            continue
        for d in r["labels"]:
            if d["label"] != CARDIAC:
                continue
            cached = len(d["mentions"])
            old_like += cached == len(matched(s, OLD_CARDIAC))
            new_like += cached == len(matched(s, new_pat))
    print(f"\ncache state: {new_like} of {old_like + new_like - min(old_like, new_like) or 1} "
          f"instances agree with the NEW pattern, {old_like} with the old "
          f"(they coincide where the fix changed nothing).")
    print("  -> re-extract is still needed" if new_like < old_like else
          "  -> cache carries the corrected mentions")

    for s in ("train", "test"):
        if a.examples and ex[s]:
            print(f"\n--- {s}: sentences newly matched by the fix "
                  f"(first {min(a.examples, len(ex[s]))} of {len(ex[s])}) ---")
            for rid, sent in ex[s][:a.examples]:
                print(f"  [{rid}] {sent[:120]}")


if __name__ == "__main__":
    main()

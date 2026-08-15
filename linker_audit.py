"""Audit the keyword+negation mention matcher against manual mention annotations.

A no-match rate says nothing about how often a matched sentence is the wrong sentence, and
the patterns overlap across labels (`opacit` appears in
consolidation, effusion and atelectasis sentences; `mass` matches "mass effect"), while the
negation guard fires on any word-boundary "no", which radiology prose embeds inside positive
statements ("Stable large right pleural effusion, no significant interval change").

The REFLACX authors released ground truth for exactly this. `manually_labeled_reports_3.csv`
in ricbl/eyetracking (examples_and_paper_numbers/) gives, for 200 random Phase 3 readings and
each of the 14 findings, the character spans of every mention of that finding in the report --
produced by running a modified CheXpert labeler and then correcting it by hand. Fourteen
labels, the same fourteen we use.

Unit of evaluation is the (reading, label) pair: did the matcher find a mention, and was there
one? 200 x 14 = 2800 pairs. Sentence-level agreement is deliberately NOT attempted: their
character offsets index the report text their `extract_report.py` produced, and our sentences
come from joining the word-timestamp transcript, so offsets need not line up. Presence is the
robust unit, and it separates the two failure modes that matter -- a false
positive is a pattern that fires on the wrong finding, a false negative is a mention the
patterns or the guard threw away.

The guard is scored both ways, so how much recall it costs and how much precision it buys
are both visible.

    curl -sLO https://raw.githubusercontent.com/ricbl/eyetracking/master/\
examples_and_paper_numbers/manually_labeled_reports_3.csv
    python linker_audit.py /path/to/reflacx --gt manually_labeled_reports_3.csv
"""
import argparse
import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd

from linker_and_temporal import KEYWORDS, NEG, sentences_with_times


def predicted(sents, label, guard):
    """Sentences our matcher assigns to `label`. guard=False disables the negation cue list."""
    pat = KEYWORDS.get(label)
    if not pat:
        return []
    rx = re.compile(pat, re.I)
    return [t for t, _s, _e in sents if rx.search(t) and not (guard and NEG.search(t))]


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f = 2 * p * r / (p + r) if p and r and np.isfinite(p + r) else float("nan")
    return p, r, f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="REFLACX root (searched recursively)")
    ap.add_argument("--gt", default="manually_labeled_reports_3.csv")
    ap.add_argument("--examples", type=int, default=6, help="FP/FN sentences to print per type")
    ap.add_argument("--cache", default=None,
                    help="align.pt; restricts the audit to (reading, label) pairs that carry "
                         "an ellipse, which is the set the pipeline actually evaluates")
    a = ap.parse_args()

    gt = pd.read_csv(a.gt)
    labels = [c[:-len("_location")] for c in gt.columns if c.endswith("_location")]
    ours = {k.lower(): k for k in KEYWORDS}
    missing = [l for l in labels if l.lower() not in ours]
    if missing:                       # fail loudly: a silent label mismatch would fake a score
        raise SystemExit(f"labels in the ground truth we do not match: {missing}")

    # reading id -> transcript. REFLACX stores one directory per reading, named by its id.
    tr = {p.parent.name: p for p in Path(a.root).rglob("timestamps_transcription.csv")}
    ids = [i for i in gt["IDs"] if i in tr]
    print(f"ground truth {len(gt)} readings, {len(ids)} found under {a.root}", flush=True)
    if not ids:
        # say which of the two failures this is, rather than making the caller guess
        if not tr:
            raise SystemExit(f"no timestamps_transcription.csv anywhere under {a.root} -- "
                             f"wrong root?")
        got = sorted(tr)[:5]
        raise SystemExit(
            f"found {len(tr)} transcripts, but none of their directory names match the "
            f"ground-truth ids.\n  ids look like: {list(gt['IDs'][:3])}\n"
            f"  directories look like: {got}\n"
            f"  example path: {tr[got[0]]}")

    sent_cache = {i: sentences_with_times(pd.read_csv(tr[i])) for i in ids}
    g = gt.set_index("IDs")

    # A (reading, label) pair only becomes an instance if that label was annotated in that
    # reading. A pattern firing on "mediastinum normal" is a false positive here but never
    # reaches the pipeline, because no ellipse means no instance. Restricting to the annotated
    # set gives the precision that describes the data the model is actually scored on.
    annotated = None
    if a.cache:
        import torch
        recs, _ = torch.load(a.cache, weights_only=False)
        annotated = {(r["rid"], d["label"]) for r in recs for d in r["labels"]
                     if len(d.get("ellipses", []))}
        print(f"cache: {len(annotated)} annotated (reading, label) pairs; "
              f"{len({r for r, _l in annotated} & set(ids))} of the {len(ids)} audited "
              f"readings appear in it", flush=True)

    for scope in (("all", None), ("annotated", annotated)) if annotated else (("all", None),):
      for guard in (True, False):
        sname, keep = scope
        per, fps, fns, guard_kills = {}, [], [], 0
        for lab in labels:
            k = ours[lab.lower()]
            tp = fp = fn = tn = 0
            for i in ids:
                if keep is not None and (i, k) not in keep:
                    continue
                truth = bool(ast.literal_eval(str(g.loc[i, lab + "_location"]) or "[]"))
                hits = predicted(sent_cache[i], k, guard)
                if hits and truth:
                    tp += 1
                elif hits:
                    fp += 1
                    fps.append((lab, hits[0]))
                elif truth:
                    fn += 1
                    fns.append((lab, ""))
                    if guard and predicted(sent_cache[i], k, False):
                        guard_kills += 1        # the pattern DID fire; the guard threw it away
                else:
                    tn += 1
            per[lab] = (tp, fp, fn, tn)

        T = [sum(v[j] for v in per.values()) for j in range(4)]
        p, r, f = prf(T[0], T[1], T[2])
        tag = "guard ON (shipped)" if guard else "guard OFF"
        npair = sum(sum(v) for v in per.values())
        print(f"\n{'=' * 72}\n{tag} -- {sname} pairs   n = {npair}")
        print(f"  TP {T[0]}  FP {T[1]}  FN {T[2]}  TN {T[3]}")
        print(f"  precision {p:.3f}   recall {r:.3f}   F1 {f:.3f}")
        if guard:
            print(f"  false negatives whose pattern fired but the guard rejected: "
                  f"{guard_kills}/{T[2]}")

        print(f"\n  {'label':32s} {'n+':>4s} {'TP':>4s} {'FP':>4s} {'FN':>4s} "
              f"{'prec':>6s} {'rec':>6s}")
        macro = []
        for lab in labels:
            tp, fp, fn, _ = per[lab]
            pp, rr, _ff = prf(tp, fp, fn)
            macro.append((pp, rr))
            print(f"  {lab[:32]:32s} {tp + fn:4d} {tp:4d} {fp:4d} {fn:4d} "
                  f"{pp:6.3f} {rr:6.3f}")
        mp = np.nanmean([x[0] for x in macro]); mr = np.nanmean([x[1] for x in macro])
        print(f"  {'MACRO (over labels)':32s} {'':4s} {'':4s} {'':4s} {'':4s} "
              f"{mp:6.3f} {mr:6.3f}")

        if a.examples and guard:
            print(f"\n  false positives -- pattern fired, no mention annotated:")
            for lab, s in fps[:a.examples]:
                print(f"    [{lab}] {s[:110]}")
    print("\nPresence at the (reading, label) level. Sentence-level agreement is not scored: "
          "their char offsets index a different report rendering than our transcript join.")


if __name__ == "__main__":
    main()

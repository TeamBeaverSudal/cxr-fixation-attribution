# Finding-Level Fixation Attribution from Radiologist Gaze and Dictation

Code for *A Structured-Cue Analysis of Finding-Level Fixation Attribution in Chest
Radiography*. Nothing here contains patient data, and no derived cache is redistributed.

## Reproduction status

`verify_paper.py` rebuilds every table and compares each value against the manuscript. Of 86
reported quantities, 66 match to four decimals and 17 of the remaining 20 differ by less than
0.008. Nothing in the paper is unaccounted for.

Exact, to four decimals:

- The mention-resolved cohorts (987 test, 547 validation, 1,895 train), the test split
  description (489 readings, 419 patients) and the 735 eligible training patients
- Every Table I rule-based row, and the 3.0-s row's pointing-game accuracy
- The learned selector, the reduced-input selector, and the masking control's pointing
- Table II's exact-match cohort (n = 948) and the target-record row
- Every Table IV instance count and every finding-level pointing difference, plus both macro
  differences
- Table V's full-training row
- The inferential comparison: 0.0405 (CI 0.0109-0.0695, p = 0.0400)

Known gaps:

| quantity | ours | paper |
|---|---|---|
| Other-patient row, pointing | 0.4905 | 0.5527 |
| Table V 10% learned, pointing | 0.7566 | 0.7760 |
| Enlarged cardiac silhouette, IoU difference | 0.0650 | 0.0796 |
| Every other IoU | within 0.002 | |

Two of these are not closable from what the paper records. The donor row averages the first
`min(K, m)` records of a SHA-256 ranking; the ranking's input is not stated, so a different
top-8 is selected. Table V's partial fractions average five patient subsample chains whose
draw is not recorded.

The IoU offsets are threshold selection, not method. Pointing-game accuracy depends only on
the bandwidth and reproduces exactly everywhere, which fixes the bandwidth; IoU additionally
depends on the binarization threshold, and validation IoU is near-tied across neighbouring
grid points. The offsets trace to one choice: the 3.0-s row is 0.0019 low, and the clustered
IoU statistics computed against it are 0.0019 high.

Two definitions were recovered rather than read off the code, and both are confirmed by an
exact count. The record-substitution groups match on the seven location indicators, which
`masking_control.SPATIAL_IDX` already names: matching on all ten indicators gives 910
and on the four directional ones 970. The inferential comparison is against the validation-selected
3.0-s baseline, not the prespecified 1.5-s one.

## Data

REFLACX Phase 3 and MIMIC-CXR are distributed by PhysioNet under a credentialed data use
agreement. Obtain them there; this repository assumes a REFLACX root containing one directory
per reading, each with `fixations.csv`, `anomaly_location_ellipses.csv` and
`timestamps_transcription.csv`.

## What produces what

Indexed by the artifact in the paper rather than by filename, since the filenames carry the
development history rather than the argument.

| paper artifact | produced by |
|---|---|
| **Mention-resolved cohort** (987 test / 547 val / 1,895 train) | `linker_and_temporal.py` — `KEYWORDS` holds the per-finding patterns, `NEG` the sentence-level negation cues. These two objects decide which instances carry a resolved mention, so they define the cohort every reported number sits on. |
| **Table I**, rows 1–2 (scanpath density, temporal baseline) | `core.py` (`b1_at`), `structured_baselines.py` |
| **Table I**, rows 3–8 (anatomical prior, directional terms, scanpath support, temporal gate, and their combinations) | `structured_baselines.py`, with the prior and half-plane orientation from `prior_and_swap.py` |
| **Table I**, learned selector | `selector.py` (model and training), `evaluate.py` (evaluation) |
| **Table II**, record substitution | `evaluate.py` — exact-match groups on the seven location indicators, donors ranked by a fixed SHA-256 over the reading id, first `min(8, m)` maps averaged |
| **Table III**, input-feature controls | `evaluate.py`; the spatial-indicator masking is `masking_control.py`, the reduced query and both permutations are in `evaluate.py`/`selector.py` |
| **Counterfactual swap** | `prior_and_swap.py` |
| **Architecture selection** (2×2, validation IoU) | `architecture_selection.py` |
| **Patient-cluster bootstrap and patient-level signed-rank** | `cluster_stats.py` |
| **Linker audit** | `linker_audit.py`; `mention_diff.py` scopes a pattern change before rerunning |
| **Table IV** finding-level differences, **Table V** training-set size | `verify_paper.py`; `evaluate.py` takes the training fraction and chain seed |
| **Every reported value, checked against the manuscript** | `verify_paper.py` |
| **Table I cohort check** | `repro_check.py` |
| **Schematic figure** | `figure_schematic.py` |
| Data loading, fixation features, splatting, metrics | `reflacx_io.py`, `core.py` |
| Run logs → reported tables | `summarize_runs.py` |

The qualitative-example figure is not included: it resolves image paths through the evaluation
code for a comparison the paper no longer reports, which would pull an unrelated model and its
dependencies into this package. Everything needed to reproduce the reported numbers is here.

## The training cohort for derived quantities

The task is defined on instances with a resolved positive mention, and every quantity derived
from training data follows that definition: the label-conditional anatomical prior and the
directional half-plane assignment are estimated from the **mention-resolved** training
instances (1,895 of 2,097), not from all of them.

This matters more than it looks. Rebuilding Table I with the prior estimated on all training
instances reproduces the two rows that use no prior exactly and misses every row that does --
the anatomical prior lands at 0.5076/0.2736 instead of 0.5035/0.2736, the combined structured
baseline at 0.7953/0.3441 instead of 0.7923/0.3439. `repro_check.py` runs both cohorts side by
side and prints the published values underneath, so the choice can be verified rather than
taken on trust.

## Reproducing

Extraction runs once and is the only step that needs the raw dataset:

```bash
python core.py /path/to/reflacx --cache align.pt
```

Then one command checks every reported value and prints a per-value comparison:

```bash
python verify_paper.py --cache align.pt --all
```

Individual blocks run with `--only cohort,table1,lookback3`. Each training seed and each
Table V subsample chain is written to a checkpoint stamped with the cache name and a hash of
the scoring code, so an interrupted run resumes and an edited model never reads a stale one.

Then the evaluation. Runs span five optimization seeds on the primary patient split and five
patient partitions at seed 0; the two are summarized separately rather than pooled.

```bash
for s in 0 1 2 3 4; do python evaluate.py --cache align.pt --epochs 40 --seed $s --split-seed 0; done
for p in 1 2 3 4;   do python evaluate.py --cache align.pt --epochs 40 --seed 0 --split-seed $p; done

for s in 0 1 2 3 4; do python structured_baselines.py --epochs 40 --seed $s --split-seed 0; done
for p in 1 2 3 4;   do python structured_baselines.py --epochs 40 --seed 0 --split-seed $p; done

python summarize_runs.py <logfile>
```

Architecture selection and the clustered statistics:

```bash
python architecture_selection.py  --cache align.pt --epochs 40 --seed 0 --split-seed 0
python cluster_stats.py --cache align.pt --epochs 40
```

## Auditing the linker

The linker decides the cohort, so it is worth checking rather than trusting. REFLACX ships
hand-corrected mention locations for 200 Phase 3 readings:

```bash
curl -sLO https://raw.githubusercontent.com/ricbl/eyetracking/master/examples_and_paper_numbers/manually_labeled_reports_3.csv
python linker_audit.py /path/to/reflacx --gt manually_labeled_reports_3.csv --cache align.pt
```

It scores presence at the (reading, finding) level, with and without the negation guard, and
reports which findings the misses concentrate on. `mention_diff.py` does the complementary
job: before changing a pattern, it checks the newly matched sentences against that ground
truth and counts how many training instances the change would move — including matched-to-
matched cases where only the selected sentence differs, which silently alter a training input.

## Self-checks

Several scripts carry `--selfcheck`, which runs on synthetic data and needs no dataset:

```bash
python structured_baselines.py --selfcheck
python linker_and_temporal.py
```

These assert the properties the analysis depends on — that the gated constructions reduce to
the temporal baseline under a uniform prior, that a directional word moves mass into the half
it names, that the temporal gate still excludes out-of-window fixations, and that the
attention scorer is additive in the way the position-permutation control assumes.

## Note on one pattern

The pattern for enlarged cardiac silhouette originally required *enlarged* to precede
*cardiac*, so it missed "the cardiac silhouette is enlarged" — the commonest phrasing. The fix
raised that finding's recall against the released annotations from 0.744 to 0.974 and moved
187 of 558 instances for that label, 101 of them in training. All reported numbers use the
corrected pattern. `mention_diff.py` exists because of this: a linker change is not a
per-finding re-score, since the matched sentence supplies the query's directional terms, every
temporal feature, and the baseline's gate.

# Finding-Level Fixation Attribution from Radiologist Gaze and Dictation

Code for *A Structured-Cue Analysis of Finding-Level Fixation Attribution in Chest
Radiography*. Nothing here contains patient data, and no derived cache is redistributed.

## Reproduction status

The Table I rule-based rows and the learned selector row reproduce to four decimals with the
commands below. The remaining tables and the clustered statistics are being checked; this note
will be updated when they are.

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
| **Table II**, record substitution | `structured_baselines.py` |
| **Table III**, input-feature controls | `evaluate.py`; the spatial-indicator masking is `masking_control.py` |
| **Counterfactual swap** | `prior_and_swap.py` |
| **Architecture selection** (2×2, validation IoU) | `architecture_selection.py` |
| **Patient-cluster bootstrap and patient-level signed-rank** | `cluster_stats.py` |
| **Linker audit** | `linker_audit.py`; `mention_diff.py` scopes a pattern change before rerunning |
| **Table I reproduction check** | `repro_check.py` |
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

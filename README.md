# Finding-Level Gaze Targets

Reference implementation for constructing a separate localization target for
each reported finding from a radiologist's complete chest-radiograph scanpath.

A complete reading may contain several findings but provides only one recorded
scanpath. This project assigns finding-conditioned weights to the observed
fixations and renders them as localization maps. It provides structured and
learned selectors under a shared rendering, calibration, and patient-disjoint
evaluation protocol. Image pixels are not selector inputs.

## Method scope

The project contains:

- positive-mention linking and temporal alignment for REFLACX;
- a structured selector combining training-derived anatomy, target-record
  scanpath support, mention timing, and directional terms;
- a lightweight learned fixation selector using finding identity, continuous
  temporal and kinematic features, position, and mention indicators;
- record-substitution, feature-control, training-size, and patient-partition
  analyses; and
- an exact aggregate-result registry with provenance and data-free validation.

The implementation operates on recorded fixations. It does not infer attention
from image pixels or claim moment-to-moment diagnostic intent.

## Reference study

The fixed test cohort contains 987 mention-linked finding instances from 398
patients. Learned metrics are averaged over optimizer seeds 0--4 for each
instance before paired inference.

| Method | Pointing accuracy | IoU |
|---|---:|---:|
| Validation-selected 3.0-s structured selector | 0.7893 | 0.3549 |
| Ten-indicator learned selector | 0.8245 | 0.3584 |

The learned-minus-structured difference is +0.0353 for pointing accuracy
(patient-cluster bootstrap 95% CI +0.0067 to +0.0634; patient-mean signed-rank
`p=0.0767`) and +0.0035 for IoU (95% CI -0.0034 to +0.0102;
`p=0.8417`). Exact values and estimator definitions are stored in
[`results/study-results.json`](results/study-results.json).

## Project structure

```text
configs/study.json                      frozen study configuration
src/finding_level_gaze_targets/
  data/                                 REFLACX file adapters
  linking/                              positive-mention alignment
  maps/                                 fixation features and rendering
  baselines/                            anatomical and structured selectors
  models/                               learned fixation selector
  experiments/                          primary and controlled analyses
  reporting/                            result-registry validation and summary
scripts/run_analysis.py                 unified analysis entry point
results/study-results.json              exact reference-study aggregate
tests/                                  data-free contract and aggregation tests
docs/RESULTS.md                         analysis-to-result mapping
docs/REPRODUCIBILITY.md                 end-to-end execution protocol
docs/PROVENANCE.md                      executed source and job identities
verify_results.py                       result and optional PDF validation
```

## Install and validate

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
python verify_results.py
finding-level-gaze-results --summary
pytest -q
```

To verify a local copy of the associated publication against the same frozen
results, including its source and bibliography:

```bash
python verify_results.py \
  --pdf /path/to/paper.pdf \
  --tex /path/to/main.tex \
  --bibliography /path/to/references.bib
```

These data-free checks validate the released aggregate and study invariants.
Recomputation from credentialed source data follows
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Data boundary

REFLACX 1.0.0 and MIMIC-CXR 2.0.0 are distributed by PhysioNet under
credentialed access. This repository contains no raw images, reports, patient
identifiers, derived caches, patient-level predictions, model checkpoints, or
qualitative radiographs.

## Associated publication

The reference study accompanies **“From Complete Scanpaths to Finding-Level
Gaze Targets in Chest Radiography: Structured Cues and Learned Reweighting,”**
accepted at IEEE MedAI 2026. Citation metadata are provided in `CITATION.cff`.

The current branch contains only the code paths, configuration, aggregate
results, and documentation used by this paper. Earlier public framing and
superseded analysis code remain recoverable through version history; manuscript
snapshots and restricted artifacts stay in private provenance storage rather
than the release tree.

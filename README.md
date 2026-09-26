# Finding-Level Gaze Targets

Official code and publication-safe artifacts for **“From Complete Scanpaths to
Finding-Level Gaze Targets in Chest Radiography: Structured Cues and Learned
Reweighting,”** accepted at IEEE MedAI 2026.

The project assigns finding-specific weights to the fixations in a completed
radiology scanpath and renders them as a localization map. It compares a
structured selector with learned reweighting under the same rendering,
calibration, and patient-disjoint evaluation protocol. Image pixels are not
selector inputs.

## Camera-ready result

The fixed test cohort contains 987 mention-linked finding instances from 398
patients. Learned metrics are averaged over optimizer seeds 0--4 for each
instance before paired inference.

| Method | Pointing accuracy | IoU |
|---|---:|---:|
| Validation-selected 3.0-s structured baseline | 0.7893 | 0.3549 |
| Ten-indicator learned selector | 0.8245 | 0.3584 |

The learned-minus-structured difference is +0.0353 for pointing accuracy
(patient-cluster bootstrap 95% CI +0.0067 to +0.0634; patient-mean signed-rank
`p=0.0767`) and +0.0035 for IoU (95% CI -0.0034 to +0.0102;
`p=0.8417`). The complete exact-valued registry is
[`results/medai2026-camera-ready.json`](results/medai2026-camera-ready.json).

## Project structure

```text
configs/medai2026.json                 frozen cohort and estimator configuration
src/finding_level_gaze_targets/
  data/                                REFLACX file adapters
  linking/                             positive-mention and temporal alignment
  maps/                                fixation features, rendering, and metrics
  baselines/                           anatomical and structured selectors
  models/                              learned fixation reweighting
  experiments/                         reported analyses and strict aggregation
  reporting/                           result-registry validation and summary
scripts/run_analysis.py                single analysis entry point
results/medai2026-camera-ready.json     final publication-safe aggregate
tests/                                 data-free contract and aggregation tests
docs/RESULTS.md                        paper-to-result mapping
docs/REPRODUCIBILITY.md                end-to-end execution protocol
docs/PROVENANCE.md                     executed source and job identities
verify_paper.py                        final registry and optional PDF check
```

## Install and verify

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
python verify_paper.py
finding-level-gaze-results --summary
pytest -q
```

With a local copy of the final PDF:

```bash
python verify_paper.py --pdf /path/to/medai_final.pdf
```

The data-free checks validate the frozen result contract. Recomputing model
outputs requires credentialed REFLACX/MIMIC-CXR access and follows
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Data boundary

REFLACX 1.0.0 and MIMIC-CXR 2.0.0 are distributed by PhysioNet under
credentialed access. This repository contains no raw images, reports, patient
identifiers, derived caches, patient-level predictions, model checkpoints, or
qualitative radiographs.

The stable camera-ready snapshot is tagged `medai2026-camera-ready`. The GitHub
repository slug remains `cxr-fixation-attribution` so the URL printed in the
paper remains valid; the project and Python package names follow the final
paper's finding-level gaze-target terminology.

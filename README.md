# From Complete Scanpaths to Finding-Level Gaze Targets in Chest Radiography

Code and publication-safe artifacts for **“From Complete Scanpaths to
Finding-Level Gaze Targets in Chest Radiography: Structured Cues and Learned
Reweighting,”** accepted at IEEE MedAI 2026.

The study constructs a finding-specific gaze target from a radiologist's
complete REFLACX scanpath. A structured selector and a learned selector assign
weights to the same recorded fixations; the shared renderer turns those weights
into localization maps. No patient data, derived cache, model checkpoint, or
case identifier is distributed here.

## Camera-ready result contract

The camera-ready analysis uses 987 mention-linked test instances from 398
patients. Learned results average optimizer seeds 0--4 per instance before
paired inference. The structured reference uses the 3.0-s lookback selected by
validation IoU.

| Method | Pointing accuracy | IoU |
|---|---:|---:|
| Validation-selected structured baseline | 0.7893 | 0.3549 |
| Ten-indicator learned selector, five-seed mean | 0.8245 | 0.3584 |

The learned-minus-structured difference is +0.0353 for pointing accuracy
(patient-cluster bootstrap 95% CI +0.0067 to +0.0634; patient-mean signed-rank
`p=0.0767`) and +0.0035 for IoU (95% CI -0.0034 to +0.0102;
`p=0.8417`). These summaries use different weighting: the bootstrap targets the
instance-weighted mean while the signed-rank sensitivity gives each patient
equal weight.

The complete exact-valued contract is
[`results/camera-ready-results.json`](results/camera-ready-results.json).
[`docs/RESULT_CONTRACT.md`](docs/RESULT_CONTRACT.md) maps every camera-ready
table and quantitative statement to that file. The old 419-patient,
single-seed-inference, and Table-V values belong to a superseded submitted
analysis and are not camera-ready results.

## Repository map

```text
core.py, selector.py              shared representation and learned selector
structured_baselines.py          anatomical, scanpath, directional, temporal baselines
evaluate.py                      feature controls and seed-specific evaluation
experiments/                     camera-ready five-seed analyses and strict aggregation
results/camera-ready-results.json publication-safe final aggregate contract
docs/RESULT_CONTRACT.md           paper-to-artifact mapping and rounding rules
docs/REPRODUCIBILITY.md           end-to-end commands and variation axes
docs/PROVENANCE.md                executed jobs, source hashes, and audit boundary
verify_paper.py                   offline contract and optional PDF consistency check
```

## Verify the released contract

This check is data-free and verifies the frozen aggregate against an independent
set of camera-ready expectations, including cohort sizes, Tables I--IV, paired
inference, record substitution, all five patient partitions, and the
training-fraction design:

```bash
python verify_paper.py
```

If a local copy of the camera-ready PDF is available, also check its title and
all printed quantitative tokens:

```bash
python verify_paper.py --pdf /path/to/medai_final.pdf
```

The optional PDF check uses `pdftotext`. Contract validation is not a substitute
for rerunning the models from credentialed source data; the commands and exact
aggregation order for that audit are in
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Data boundary

REFLACX Phase 3 and MIMIC-CXR are available through PhysioNet under
credentialed access. Extraction expects a REFLACX root with the released gaze,
fixation, ellipse, and timestamped-transcription files. Keep raw data, caches,
patient-level predictions, qualitative radiographs, and checkpoints outside
Git.

## Installation

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python structured_baselines.py --selfcheck
python linker_and_temporal.py
python verify_paper.py
```

The execution used NumPy 1.26.4 and Python/PyTorch tooling compatible with the
versions listed in `requirements.txt`. The historical environment was not
preserved as a byte-identical lock; source hashes and completed job identities
are therefore part of the provenance record.

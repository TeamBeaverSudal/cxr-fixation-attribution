# Reproducing the camera-ready analysis

## Data and cache

Obtain REFLACX 1.0.0 and MIMIC-CXR through PhysioNet under the applicable data
use agreement. Do not place raw files or the derived cache in this repository.

Build the historical cache with the released linker, then retain the raw
timestamped transcripts because the 0.5--3.0-s lookback sweep reconstructs
sentence boundaries that are not fully represented in the original 1.5-s
cache.

```bash
python core.py /path/to/reflacx --cache /private/path/align.pt
```

The 1.5-s reconstruction must match the cached mentions before any extended
window result is accepted.

## Primary comparison and patient partitions

The primary run trains seeds 0--4, calibrates every learned model on validation,
selects the structured lookback using validation IoU, averages learned outcomes
per test instance, and then performs patient-cluster inference.

```bash
python experiments/strongest_window_inference.py \
  --cache /private/path/align.pt \
  --raw-root /path/to/reflacx \
  --epochs 40 --seeds 0,1,2,3,4 --split-seed 0
```

Repeat the same complete five-seed command for every patient partition. Do not
replace partitions 1--4 with a seed-0-only run.

```bash
for p in 0 1 2 3 4; do
  python experiments/strongest_window_inference.py \
    --cache /private/path/align.pt \
    --raw-root /path/to/reflacx \
    --epochs 40 --seeds 0,1,2,3,4 --split-seed "$p" \
    --models full
done
```

Keep partition outputs separate. They overlap in patient membership and are a
directional sensitivity analysis, not five independent replications.

## Record substitution

Donor ordering is fixed independently of optimizer initialization. Every seed
must retain the same 948-instance / 389-patient eligible cohort.

```bash
python experiments/matched_other_patient_scanpath.py \
  --cache /private/path/align.pt --epochs 40 \
  --seeds 0,1,2,3,4 --split-seed 0 \
  --donor-ranking-seed 20260818
```

## Feature controls

Run the selector and all controls on the fixed primary cohort for seeds 0--4.
Evaluation-time perturbations retain the original output coordinates and reuse
the ten-indicator selector's calibration.

```bash
for s in 0 1 2 3 4; do
  python evaluate.py --cache /private/path/align.pt \
    --epochs 40 --seed "$s" --split-seed 0
done
```

`experiments/refined_within_record_controls.py` is the compact executed-analysis
entry point for the same control definitions.

## Training-size sensitivity

The partial fractions cross five nested patient-subsample chains with five
optimizer seeds: 25 learned runs at each requested partial-fraction set. Average
optimizer seeds within each chain, then average the five chain means. The
structured result is deterministic within a retained patient set and must be
emitted only once, not replicated five times.

```bash
for subset in 0 1 2 3 4; do
  for model in 0 1 2 3 4; do
    extra=""
    if [ "$model" -ne 0 ]; then extra="--skip-structured"; fi
    python experiments/annotation_fraction_sensitivity.py \
      --cache /private/path/align.pt \
      --raw-root /path/to/reflacx \
      --subset-seed "$subset" --model-seed "$model" \
      --fractions 0.10,0.25,0.50 $extra
  done
done
```

The 100% row uses the primary five-seed learned result and the deterministic
3.0-s structured result. Every partial condition retains the complete
validation cohort, so this analysis varies training annotation only.

## Aggregation and acceptance

Use `experiments/summarize_consistent_runs.py` on the completed run tree. It
fails closed if a required seed, patient partition, chain, or completion marker
is absent. Compare the identifier-free output with
`results/camera-ready-results.json`, then run:

```bash
python verify_paper.py
```

Optimizer seeds, patient partitions, and patient-subsample chains are distinct
variation axes. Never pool them as if they were independent observations.

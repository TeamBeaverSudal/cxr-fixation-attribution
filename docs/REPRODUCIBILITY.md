# Reproducing the reference study

## 1. Environment and data

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
```

Obtain REFLACX 1.0.0 and MIMIC-CXR 2.0.0 through PhysioNet. Keep raw files and
derived artifacts outside the repository.

Build the cache with the released linker:

```bash
python -m finding_level_gaze_targets.maps.core \
  /path/to/reflacx --cache /private/path/align.pt
```

The extended-window analyses also require the raw timestamped transcripts. The
rebuilt 1.5-s mention windows must match the cache before longer lookbacks are
evaluated.

## 2. Primary comparison

```bash
python scripts/run_analysis.py primary \
  --cache /private/path/align.pt \
  --raw-root /path/to/reflacx \
  --epochs 40 --seeds 0,1,2,3,4 --split-seed 0
```

This run trains the ten- and four-indicator selectors, calibrates each learned
seed on validation, selects the structured lookback using validation IoU, and
performs patient-cluster inference after per-instance seed averaging.

## 3. Patient partitions

Repeat the full five-seed pipeline for all five partitions:

```bash
for p in 0 1 2 3 4; do
  out="/private/runs/patient-partitions/split-$p"
  mkdir -p "$out"
  python scripts/run_analysis.py primary \
    --cache /private/path/align.pt \
    --raw-root /path/to/reflacx \
    --epochs 40 --seeds 0,1,2,3,4 --split-seed "$p" \
    --models full > "$out/run.log"
  touch "$out/COMPLETE"
done
```

Partitions overlap in patient membership and remain separate sensitivity
analyses.

## 4. Record and feature controls

```bash
python scripts/run_analysis.py record-substitution \
  --cache /private/path/align.pt --epochs 40 \
  --seeds 0,1,2,3,4 --split-seed 0 \
  --donor-ranking-seed 20260818

for s in 0 1 2 3 4; do
  python scripts/run_analysis.py feature-controls \
    --cache /private/path/align.pt --epochs 40 --seed "$s"
done
```

Donor ordering is independent of optimizer initialization. The feature-control
run trains both the ten-indicator selector and the finding-plus-temporal model.
Evaluation-time feature perturbations retain the original output coordinates
and reuse the ten-indicator selector's calibration.

## 5. Training-size sensitivity

```bash
for subset in 0 1 2 3 4; do
  for model in 0 1 2 3 4; do
    out="/private/runs/training-fraction/subset-$subset/seed-$model"
    mkdir -p "$out"
    extra=""
    if [ "$model" -ne 0 ]; then extra="--skip-structured"; fi
    python scripts/run_analysis.py training-fraction \
      --cache /private/path/align.pt \
      --raw-root /path/to/reflacx \
      --subset-seed "$subset" --model-seed "$model" \
      --fractions 0.10,0.25,0.50 $extra > "$out/run.log"
    touch "$out/COMPLETE"
  done
done
```

Structured results are emitted once per retained patient set. Learned seeds are
averaged within each chain before the five chain means are summarized. The full
validation cohort is retained at every training fraction.

## 6. Aggregate and verify

Store the primary log under `/private/runs/primary/`, the record-substitution
log under `/private/runs/record-substitution/`, and use the directory layouts
shown above. Each completed run directory contains a `COMPLETE` marker.

```bash
python scripts/run_analysis.py aggregate /private/runs \
  --output /private/runs/aggregate.json
python verify_results.py
pytest -q
```

Compare the identifier-free aggregate with
`results/study-results.json`. Optimizer seeds, patient partitions, and
patient-subsample chains are distinct variation axes and are not pooled as
independent observations.

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

## 2. Architecture selection

The reported scoring and coordinate representation are selected only from the
validation scores of the complete two-by-two grid:

```bash
python scripts/run_analysis.py architecture-selection \
  --cache /private/path/align.pt --epochs 40 --seed 0 --split-seed 0
```

The routine never evaluates the test partition. It verifies that the
validation winner matches the configuration frozen in `settings.py`.

## 3. Structured comparison and lookback sweep

```bash
python scripts/run_analysis.py structured-comparison \
  --cache /private/path/align.pt \
  --raw-root /path/to/reflacx
```

This reconstructs all deterministic rows of Table I and every Table II
lookback. It verifies the reconstructed 1.5-s window against the cache and
then compares the measured aggregates with `results/study-results.json`.

## 4. Primary comparison

```bash
python scripts/run_analysis.py primary \
  --cache /private/path/align.pt \
  --raw-root /path/to/reflacx \
  --epochs 40 --seeds 0,1,2,3,4 --split-seed 0
```

This run trains the ten- and four-indicator selectors, calibrates each learned
seed on validation, independently rechecks structured lookback selection, and
performs patient-cluster inference after per-instance seed averaging.

## 5. Patient partitions

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

## 6. Record and feature controls

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

## 7. Training-size sensitivity

```bash
for subset in 0 1 2 3 4; do
  for model in 0 1 2 3 4; do
    out="/private/runs/training-fraction/subset-$subset/seed-$model"
    mkdir -p "$out"
    fractions="0.10,0.25,0.50"
    if [ "$subset" -eq 0 ]; then fractions="$fractions,1.00"; fi
    extra=""
    if [ "$model" -ne 0 ]; then extra="--skip-structured"; fi
    python scripts/run_analysis.py training-fraction \
      --cache /private/path/align.pt \
      --raw-root /path/to/reflacx \
      --subset-seed "$subset" --model-seed "$model" \
      --fractions "$fractions" $extra > "$out/run.log"
    touch "$out/COMPLETE"
  done
done
```

Structured results are emitted once per retained patient set. Learned seeds are
averaged within each chain before the five chain means are summarized. The full
validation cohort is retained at every training fraction. The 100% fraction is
identical across subset chains, so it is emitted only for subset chain 0.

## 8. Mention-linker audit

The released manual Phase-3 reference can be checked at the reading--finding
level without exporting report text or identifiers:

```bash
python scripts/run_analysis.py linker-audit /path/to/reflacx \
  --gt /private/path/manually_labeled_reports_3.csv \
  --cache /private/path/align.pt --examples 0
```

This is a coverage and pair-level presence audit. It does not treat transcript
sentence alignment as independently adjudicated ground truth.

## 9. Aggregate and verify

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

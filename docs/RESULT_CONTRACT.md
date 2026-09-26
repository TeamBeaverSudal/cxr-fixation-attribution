# Camera-ready result contract

This document maps the MedAI 2026 camera-ready paper to
`results/camera-ready-results.json`. The JSON stores execution-level precision;
the paper prints three decimals. A printed value is valid when conventional
rounding of the registered value gives the paper value.

## Cohort and estimator

- The fixed primary split contains 1,895 training, 547 validation, and 987 test
  mention-linked instances. The test cohort spans 398 patients.
- The learned selector is run with optimizer seeds 0--4. Each seed is calibrated
  separately on validation data. Seed-specific metrics are averaged for each
  test instance before any paired difference or patient resampling.
- The primary mean is instance weighted. Its percentile interval resamples 398
  patients and retains all instances for each sampled patient. The signed-rank
  sensitivity instead operates on one mean per patient.
- Patient partitions are overlapping sensitivity analyses. Every partition
  repeats all five optimizer seeds; partitions are never pooled as independent
  observations.

## Paper mapping

| Paper element | JSON field | Registered result |
|---|---|---|
| Cohort paragraph | `cohort` | 987 test instances, 398 patients |
| Table I | `table_1` | nine structured/learned rows |
| Table II | `table_2` | five independently calibrated lookbacks; 3.0 s selected by validation IoU |
| Primary comparison | `primary_inference.learned_minus_structured_3_0s` | pointing +0.0353 [0.0067, 0.0634], `p=0.0766815`; IoU +0.0035 [-0.0034, 0.0102], `p=0.841748` |
| Record substitution | `record_substitution` | 948 instances / 389 patients; pointing loss 0.2816; IoU loss 0.0769 |
| Table III | `table_3` | six five-seed feature conditions |
| Four-vs-ten indicator comparison | `primary_inference.four_indicator_minus_ten_indicator` | pointing +0.0128 [0.0006, 0.0250]; IoU -0.0010 [-0.0042, 0.0023] |
| Table IV | `table_4` | five seeds within each of five patient-subsample chains at partial fractions |
| Patient-partition paragraph | `patient_partitions` | pointing differences +0.0167 to +0.0356; IoU +0.0002 to +0.0122 |
| Figure 2 caption | `figure_2` | qualitative IoUs only; not an inferential sample |

## Superseded values

The following values are internally coherent results from an earlier submitted
analysis, but they do not describe the camera-ready paper and must not be used
as its contract:

- 419 test patients;
- seed-0 paired pointing difference 0.0405 with CI 0.0109--0.0695 and
  `p=0.0400`;
- patient partitions trained only at optimizer seed 0;
- the old Table V numbering and its single-seed/full-data training-size row.

Git history preserves that analysis. The current branch exposes only the
camera-ready contract in active documentation and verification.

## Audit boundary

`verify_paper.py` checks this registry against independent expectations and can
also compare the printed camera-ready tokens with a supplied PDF. It does not
claim an independent rerun of credentialed REFLACX data. Model reruns require
the data and commands in `docs/REPRODUCIBILITY.md`; executed job and source
identities are in `docs/PROVENANCE.md`.

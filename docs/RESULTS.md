# Reference study results

`results/study-results.json` is the exact quantitative registry for the
reference REFLACX study. It preserves execution-level precision; the associated
publication rounds displayed values to three decimals.

## Estimator

- Primary split: 1,895 training, 547 validation, and 987 test mention-linked
  instances; the test cohort spans 398 patients.
- Learned models: optimizer seeds 0--4, independently calibrated on validation.
- Paired inference: average seed-specific metrics within each test instance,
  then resample patients while retaining all of their instances.
- Sensitivity test: Wilcoxon signed-rank test on one mean difference per patient.
- Partition analysis: repeat the complete five-seed pipeline for each of five
  patient partitions and keep the partitions separate.
- Training-size analysis: average five optimizer seeds within each of five
  patient-subsample chains, then summarize the chain means.

## Registry mapping

| Analysis | Runner | Registry field | Result |
|---|---|---|---|
| Cohort | cache construction and all runners | `cohort` | 987 test instances, 398 patients |
| Deterministic structured selectors | `structured-comparison` | first eight rows of `table_1` | common-renderer single-source and combined comparisons |
| Temporal lookback sweep | `structured-comparison` | `table_2` | five independently calibrated lookbacks; validation selects 3.0 s |
| Learned selector and paired inference | `primary` | learned row of `table_1`; `primary_inference` | pointing +0.0353 [0.0067, 0.0634], `p=0.0766815`; IoU +0.0035 [-0.0034, 0.0102], `p=0.841748` |
| Record substitution | `record-substitution` | `record_substitution` | 948 instances / 389 patients; pointing reduction 0.2816; IoU reduction 0.0769 |
| Feature controls | `feature-controls` and `primary` | `table_3` | six five-seed selector and perturbation conditions, including four-vs-ten indicators |
| Training-size sensitivity | `training-fraction` | `table_4` | 10%, 25%, 50%, and 100% training fractions |
| Patient partitions | `primary` with split seeds 0--4 | `patient_partitions` | pointing differences +0.0167 to +0.0356; IoU +0.0002 to +0.0122 |
| Qualitative examples | restricted renderer, aggregate only | `figure_2` | example-case IoUs; not an inferential sample |

`verify_results.py` checks these fields against independently encoded expected
values and can also verify the associated publication's exact PDF, TeX, and
bibliography hashes. This offline validation does not replace a model rerun
from credentialed source data. `configs/study.json` maps every released runner
name to the corresponding analysis role.

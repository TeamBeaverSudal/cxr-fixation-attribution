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

| Analysis | Registry field | Result |
|---|---|---|
| Cohort | `cohort` | 987 test instances, 398 patients |
| Structured and learned selectors | `table_1` | nine common-renderer comparisons |
| Temporal lookback sweep | `table_2` | five calibrated lookbacks; validation selects 3.0 s |
| Primary paired inference | `primary_inference.learned_minus_structured_3_0s` | pointing +0.0353 [0.0067, 0.0634], `p=0.0766815`; IoU +0.0035 [-0.0034, 0.0102], `p=0.841748` |
| Record substitution | `record_substitution` | 948 instances / 389 patients; pointing reduction 0.2816; IoU reduction 0.0769 |
| Feature controls | `table_3` | six five-seed selector and perturbation conditions |
| Four-vs-ten indicators | `primary_inference.four_indicator_minus_ten_indicator` | pointing +0.0128 [0.0006, 0.0250]; IoU -0.0010 [-0.0042, 0.0023] |
| Training-size sensitivity | `table_4` | 10%, 25%, 50%, and 100% training fractions |
| Patient partitions | `patient_partitions` | pointing differences +0.0167 to +0.0356; IoU +0.0002 to +0.0122 |
| Qualitative examples | `figure_2` | example-case IoUs; not an inferential sample |

`verify_results.py` checks these fields against independently encoded expected
values and can also verify the associated publication's exact PDF hash. This
offline validation does not replace a model rerun from credentialed source data.

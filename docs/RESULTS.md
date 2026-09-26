# Camera-ready results

`results/medai2026-camera-ready.json` is the quantitative contract for the
MedAI 2026 paper. It stores execution-level precision; the paper displays three
decimals.

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

## Paper mapping

| Paper element | Registry field | Result |
|---|---|---|
| Cohort | `cohort` | 987 test instances, 398 patients |
| Table I | `table_1` | nine structured and learned rows |
| Table II | `table_2` | five independently calibrated lookbacks; validation selects 3.0 s |
| Primary comparison | `primary_inference.learned_minus_structured_3_0s` | pointing +0.0353 [0.0067, 0.0634], `p=0.0766815`; IoU +0.0035 [-0.0034, 0.0102], `p=0.841748` |
| Record substitution | `record_substitution` | 948 instances / 389 patients; pointing reduction 0.2816; IoU reduction 0.0769 |
| Table III | `table_3` | six five-seed selector and feature-control conditions |
| Four-vs-ten indicator comparison | `primary_inference.four_indicator_minus_ten_indicator` | pointing +0.0128 [0.0006, 0.0250]; IoU -0.0010 [-0.0042, 0.0023] |
| Table IV | `table_4` | 10%, 25%, 50%, and 100% training fractions |
| Patient partitions | `patient_partitions` | pointing differences +0.0167 to +0.0356; IoU +0.0002 to +0.0122 |
| Figure 2 | `figure_2` | qualitative-case IoUs; not an inferential sample |

`verify_paper.py` checks these fields against an independent set of expected
values and can verify the exact final-PDF hash. It does not replace a model
rerun from credentialed source data.

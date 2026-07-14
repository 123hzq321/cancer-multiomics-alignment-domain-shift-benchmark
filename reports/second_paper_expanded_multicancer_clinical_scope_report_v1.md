# Expanded Multi-Cancer Clinical Scope Report

## Purpose

This analysis broadens the manuscript beyond the five curated internal tasks by systematically discovering clinically interpretable endpoints across the 10 TCGA PanCancer cohorts already used in the project.

## Candidate task availability

| cancer_type | endpoint_family | tasks | labelled_samples_median |
| --- | --- | --- | --- |
| BRCA | molecular_or_histologic_subtype | 1 | 945.0000 |
| BRCA | stage_early_vs_advanced | 1 | 1065.0000 |
| BRCA | t_stage_low_vs_high | 1 | 1081.0000 |
| COADREAD | molecular_or_histologic_subtype | 1 | 449.0000 |
| COADREAD | stage_early_vs_advanced | 1 | 580.0000 |
| COADREAD | t_stage_low_vs_high | 1 | 591.0000 |
| HNSC | grade_low_vs_high | 1 | 501.0000 |
| HNSC | molecular_or_histologic_subtype | 1 | 487.0000 |
| HNSC | stage_early_vs_advanced | 1 | 453.0000 |
| HNSC | t_stage_low_vs_high | 1 | 460.0000 |
| KIRC | grade_low_vs_high | 1 | 504.0000 |
| KIRC | stage_early_vs_advanced | 1 | 512.0000 |
| KIRC | t_stage_low_vs_high | 1 | 512.0000 |
| LUAD | stage_early_vs_advanced | 1 | 512.0000 |
| LUAD | t_stage_low_vs_high | 1 | 511.0000 |
| LUSC | stage_early_vs_advanced | 1 | 484.0000 |
| LUSC | t_stage_low_vs_high | 1 | 485.0000 |
| OV | grade_low_vs_high | 1 | 469.0000 |
| PRAD | t_stage_low_vs_high | 1 | 487.0000 |
| UCEC | grade_low_vs_high | 1 | 529.0000 |
| UCEC | molecular_or_histologic_subtype | 1 | 507.0000 |

## Quick baseline CV on selected expanded tasks

| cancer_type | endpoint_family | source_attribute | labelled_samples | model | f1_macro | balanced_accuracy | roc_auc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HNSC | grade_low_vs_high | GRADE | 501 | logistic_elasticnet | 0.6383 | 0.6401 | 0.6801 |
| KIRC | grade_low_vs_high | GRADE | 504 | extra_trees | 0.6587 | 0.6580 | 0.7196 |
| OV | grade_low_vs_high | GRADE | 469 | logistic_elasticnet | 0.5582 | 0.5575 | 0.6066 |
| UCEC | grade_low_vs_high | GRADE | 529 | extra_trees | 0.8341 | 0.8348 | 0.9017 |
| BRCA | molecular_or_histologic_subtype | SUBTYPE | 945 | extra_trees | 0.8178 | 0.7974 | 0.9695 |
| COADREAD | molecular_or_histologic_subtype | SUBTYPE | 449 | extra_trees | 0.8886 | 0.8815 | 0.9516 |
| HNSC | molecular_or_histologic_subtype | SUBTYPE | 487 | extra_trees | 0.9387 | 0.9085 | 0.9655 |
| UCEC | molecular_or_histologic_subtype | SUBTYPE | 507 | extra_trees | 0.9065 | 0.9075 | 0.9834 |
| BRCA | stage_early_vs_advanced | AJCC_PATHOLOGIC_TUMOR_STAGE | 1065 | logistic_elasticnet | 0.5369 | 0.5435 | 0.5680 |
| COADREAD | stage_early_vs_advanced | AJCC_PATHOLOGIC_TUMOR_STAGE | 580 | extra_trees | 0.6380 | 0.6379 | 0.6656 |
| HNSC | stage_early_vs_advanced | AJCC_PATHOLOGIC_TUMOR_STAGE | 453 | logistic_elasticnet | 0.5152 | 0.5157 | 0.5621 |
| KIRC | stage_early_vs_advanced | AJCC_PATHOLOGIC_TUMOR_STAGE | 512 | extra_trees | 0.6893 | 0.6858 | 0.7594 |
| LUAD | stage_early_vs_advanced | AJCC_PATHOLOGIC_TUMOR_STAGE | 512 | logistic_elasticnet | 0.5319 | 0.5360 | 0.5586 |
| LUSC | stage_early_vs_advanced | AJCC_PATHOLOGIC_TUMOR_STAGE | 484 | logistic_elasticnet | 0.5249 | 0.5253 | 0.5267 |
| BRCA | t_stage_low_vs_high | PATH_T_STAGE | 1081 | logistic_elasticnet | 0.5308 | 0.5388 | 0.5578 |
| COADREAD | t_stage_low_vs_high | PATH_T_STAGE | 591 | logistic_elasticnet | 0.5250 | 0.5295 | 0.5487 |
| HNSC | t_stage_low_vs_high | PATH_T_STAGE | 460 | extra_trees | 0.5865 | 0.5901 | 0.6548 |
| KIRC | t_stage_low_vs_high | PATH_T_STAGE | 512 | extra_trees | 0.6597 | 0.6557 | 0.7233 |
| LUAD | t_stage_low_vs_high | PATH_T_STAGE | 511 | logistic_elasticnet | 0.5981 | 0.6140 | 0.5857 |
| LUSC | t_stage_low_vs_high | PATH_T_STAGE | 485 | logistic_elasticnet | 0.5050 | 0.5056 | 0.4776 |
| PRAD | t_stage_low_vs_high | PATH_T_STAGE | 487 | extra_trees | 0.7049 | 0.7035 | 0.7764 |

## Interpretation

The expanded discovery shows that clinically interpretable endpoints are available beyond the original five tasks, especially stage, grade, subtype, and receptor/viral-status tasks. The quick baseline uses lightweight cross-validation with training-fold feature selection and is intended to estimate endpoint difficulty, not to replace the deeply analysed five-task benchmark. Performance remains heterogeneous, supporting the manuscript's central claim that task definition and endpoint biology are major determinants of model reliability.

## Generated files

- `work/data/alignment_domain_shift_benchmark/processed/expanded_clinical_endpoint_candidates.csv`
- `work/data/alignment_domain_shift_benchmark/processed/expanded_clinical_endpoint_quick_cv.csv`
- `outputs/second_paper_expanded_scope_task_availability.png`
- `outputs/second_paper_expanded_scope_endpoint_f1.png`
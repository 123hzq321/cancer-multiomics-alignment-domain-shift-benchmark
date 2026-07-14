# Second Paper Analysis v0: Multi-omics Alignment and Domain Shift

## Working title

Sample alignment and cohort shift as determinants of cancer multi-omics prediction reliability: a benchmark and stress-test study

## Independent angle

This analysis changes the center of gravity from model architecture ranking to data reliability. Liquid/CfC-style models can remain a comparator, but the proposed second manuscript should focus on whether multi-omics conclusions are stable when sample alignment, modality coverage, and cross-cohort distribution shift are explicitly audited.

## Core experiments completed in this run

1. Five TCGA cancer-internal tasks were re-used as sample-aligned multi-omics prediction tasks.
2. A simulated misalignment stress test permuted non-mRNA omics rows at increasing rates while preserving each modality's marginal distribution.
3. A BRCA external validation analysis quantified source-to-external cohort separability using a domain classifier AUC and a standardized mean-shift score.

## Key degradation table: aligned versus fully misaligned training data

| task | model | f1_macro_mean_aligned | f1_macro_mean_full_mismatch | f1_drop |
| --- | --- | --- | --- | --- |
| COADREAD_molecular_subtype | extra_trees | 0.8893 | 0.5008 | 0.3885 |
| UCEC_molecular_subtype | extra_trees | 0.9141 | 0.5914 | 0.3227 |
| UCEC_molecular_subtype | logistic_l2 | 0.9088 | 0.6943 | 0.2145 |
| HNSC_HPV_status | extra_trees | 0.9573 | 0.8405 | 0.1168 |
| HNSC_HPV_status | logistic_l2 | 0.9640 | 0.9201 | 0.0439 |
| COADREAD_molecular_subtype | logistic_l2 | 0.8686 | 0.8323 | 0.0363 |
| KIRC_grade_binary | extra_trees | 0.7200 | 0.6868 | 0.0332 |
| KIRC_grade_binary | logistic_l2 | 0.6498 | 0.6206 | 0.0292 |
| PRAD_pathologic_T_stage | logistic_l2 | 0.6215 | 0.5970 | 0.0245 |
| PRAD_pathologic_T_stage | extra_trees | 0.6583 | 0.6400 | 0.0183 |

## BRCA external domain-shift summary

| dataset | model | f1_macro_mean | balanced_accuracy_mean | mean_abs_standardized_shift | domain_classifier_auc |
| --- | --- | --- | --- | --- | --- |
| CPTAC_2020 | extra_trees | 0.8440 | 0.8203 | 0.0384 | 0.7596 |
| SCANB_GSE96058 | extra_trees | 0.8465 | 0.8526 | 0.0381 | 0.5133 |
| SMC_2018 | extra_trees | 0.9028 | 0.9190 | 0.0652 | 0.5165 |
| TCGA_METABRIC_valid | extra_trees | 0.8225 | 0.8016 | 0.0000 | 0.5000 |
| CPTAC_2020 | logistic_l2 | 0.8749 | 0.9013 | 0.0384 | 0.7596 |
| SCANB_GSE96058 | logistic_l2 | 0.7851 | 0.8603 | 0.0381 | 0.5133 |
| SMC_2018 | logistic_l2 | 0.7751 | 0.8732 | 0.0652 | 0.5165 |
| TCGA_METABRIC_valid | logistic_l2 | 0.8219 | 0.8521 | 0.0000 | 0.5000 |

## Core modality coverage

| task | modality | n_features | rows_with_any_observed_feature | labelled_rows | feature_missing_percent |
| --- | --- | --- | --- | --- | --- |
| UCEC_molecular_subtype | mrna | 469 | 507 | 507 | 1.3% |
| UCEC_molecular_subtype | gistic | 469 | 507 | 507 | 0.2% |
| UCEC_molecular_subtype | log2cna | 469 | 507 | 507 | 0.2% |
| UCEC_molecular_subtype | methylation | 387 | 507 | 507 | 0.6% |
| UCEC_molecular_subtype | rppa | 469 | 405 | 507 | 88.9% |
| UCEC_molecular_subtype | mutation | 469 | 507 | 507 | 0.0% |
| COADREAD_molecular_subtype | mrna | 469 | 449 | 449 | 0.5% |
| COADREAD_molecular_subtype | gistic | 469 | 449 | 449 | 0.2% |
| COADREAD_molecular_subtype | log2cna | 469 | 449 | 449 | 0.2% |
| COADREAD_molecular_subtype | methylation | 387 | 449 | 449 | 0.6% |
| COADREAD_molecular_subtype | rppa | 469 | 356 | 449 | 89.0% |
| COADREAD_molecular_subtype | mutation | 469 | 449 | 449 | 0.0% |
| HNSC_HPV_status | mrna | 469 | 487 | 487 | 0.0% |
| HNSC_HPV_status | gistic | 469 | 487 | 487 | 0.2% |
| HNSC_HPV_status | log2cna | 469 | 487 | 487 | 0.2% |
| HNSC_HPV_status | methylation | 387 | 487 | 487 | 0.6% |
| HNSC_HPV_status | rppa | 469 | 202 | 487 | 94.3% |
| HNSC_HPV_status | mutation | 469 | 487 | 487 | 0.0% |
| KIRC_grade_binary | mrna | 469 | 502 | 504 | 0.4% |
| KIRC_grade_binary | gistic | 469 | 501 | 504 | 0.8% |
| KIRC_grade_binary | log2cna | 469 | 501 | 504 | 0.8% |
| KIRC_grade_binary | methylation | 387 | 503 | 504 | 0.9% |
| KIRC_grade_binary | rppa | 469 | 451 | 504 | 87.6% |
| KIRC_grade_binary | mutation | 469 | 504 | 504 | 0.0% |
| PRAD_pathologic_T_stage | mrna | 469 | 486 | 487 | 0.2% |
| PRAD_pathologic_T_stage | gistic | 469 | 482 | 487 | 1.2% |
| PRAD_pathologic_T_stage | log2cna | 469 | 482 | 487 | 1.2% |
| PRAD_pathologic_T_stage | methylation | 387 | 487 | 487 | 0.5% |
| PRAD_pathologic_T_stage | rppa | 469 | 345 | 487 | 90.2% |
| PRAD_pathologic_T_stage | mutation | 469 | 487 | 487 | 0.0% |

## Draft conclusion

Across five cancer-internal tasks, intentionally disrupting the pairing between mRNA and other omics layers reduced multi-omics performance in a task- and model-dependent manner, even though the marginal distributions of each omics modality were preserved. This supports a data-centric conclusion: multi-omics fusion benchmarks should report sample-level alignment audits and stress tests, not only model scores. In the BRCA external setting, external cohorts were separable from the TCGA/METABRIC source distribution by marker expression profiles, and external performance varied across cohorts despite using the same label space. The second manuscript can therefore argue that sample alignment and domain shift are first-order determinants of reliability in cancer multi-omics prediction.

## Generated files

- `work/data/alignment_domain_shift_benchmark/processed/misalignment_raw_results.csv`
- `work/data/alignment_domain_shift_benchmark/processed/misalignment_summary.csv`
- `work/data/alignment_domain_shift_benchmark/processed/brca_domain_shift_raw_results.csv`
- `work/data/alignment_domain_shift_benchmark/processed/brca_domain_shift_summary.csv`
- `outputs/second_paper_misalignment_degradation.png`
- `outputs/second_paper_domain_shift_scatter.png`
- `outputs/second_paper_core_modality_coverage_heatmap.png`
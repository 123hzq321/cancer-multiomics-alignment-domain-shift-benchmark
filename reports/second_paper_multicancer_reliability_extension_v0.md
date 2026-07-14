# Multi-Cancer Reliability Extension v0

## Purpose

This extension strengthens the second manuscript by adding a 10-cancer TCGA PanCancer audit. It separates three facts that can otherwise be conflated: cancer-type classification is easy, within-cancer clinical or molecular endpoints are harder, and modality coverage varies strongly by assay.

## PanCancer modality coverage summary

| modality | cancer_types | observed_fraction_mean | observed_fraction_min | observed_fraction_max |
| --- | --- | --- | --- | --- |
| mutation | 10 | 1.0000 | 1.0000 | 1.0000 |
| gistic | 10 | 0.9791 | 0.9009 | 0.9979 |
| log2cna | 10 | 0.9791 | 0.9009 | 0.9979 |
| methylation | 10 | 0.9615 | 0.7048 | 0.9945 |
| mrna | 10 | 0.8619 | 0.2615 | 0.9982 |
| rppa | 10 | 0.0938 | 0.0532 | 0.1232 |

## Cancer identity separability

| cancer_type | modality | positive_samples | negative_samples | one_vs_rest_auc_mean | one_vs_rest_auc_sd |
| --- | --- | --- | --- | --- | --- |
| COADREAD | mrna | 594 | 5363 | 0.9997 | 0.0004 |
| PRAD | mrna | 494 | 5463 | 0.9995 | 0.0011 |
| KIRC | mrna | 512 | 5445 | 0.9988 | 0.0013 |
| HNSC | mrna | 523 | 5434 | 0.9986 | 0.0009 |
| BRCA | mrna | 1084 | 4873 | 0.9977 | 0.0017 |
| UCEC | mrna | 529 | 5428 | 0.9974 | 0.0017 |
| LUAD | mrna | 566 | 5391 | 0.9812 | 0.0046 |
| LUSC | mrna | 487 | 5470 | 0.9771 | 0.0115 |
| GBM | mrna | 585 | 5372 | 0.9733 | 0.0053 |
| OV | mrna | 583 | 5374 | 0.9721 | 0.0066 |

## Between-cancer versus within-cancer difficulty

| analysis | task | best_model | macro_f1_mean | macro_f1_sd | task_family |
| --- | --- | --- | --- | --- | --- |
| PanCancer cancer-type classification | 10 cancer types | tcn_modality_sequence__expression_last | 0.9904 | 0.0006 | between-cancer |
| Within-cancer endpoint prediction | HNSC_HPV_status | extra_trees | 0.9505 | 0.0459 | within-cancer |
| Within-cancer endpoint prediction | UCEC_molecular_subtype | logistic_elasticnet | 0.9134 | 0.0502 | within-cancer |
| Within-cancer endpoint prediction | COADREAD_molecular_subtype | extra_trees | 0.9060 | 0.0495 | within-cancer |
| Within-cancer endpoint prediction | PRAD_pathologic_T_stage | small_liquid_cfc_modality_sequence | 0.7071 | 0.0651 | within-cancer |
| Within-cancer endpoint prediction | KIRC_grade_binary | hist_gradient_boosting | 0.6996 | 0.0158 | within-cancer |

## Interpretation

The PanCancer cancer-type task reaches very high macro F1 in the existing benchmark, and one-vs-rest mRNA classifiers separate every cancer type with high AUC. This supports the argument that cancer identity is a dominant signal in multi-omics data. In contrast, clinically meaningful within-cancer endpoints show lower and more variable macro F1. Therefore, high pan-cancer classification performance should not be interpreted as evidence that multi-omics models are robust for clinically relevant within-cancer prediction. The coverage audit also shows that RPPA remains the sparsest modality at the PanCancer level, reinforcing the need for modality coverage checks before early fusion.

## Generated files

- `work/data/alignment_domain_shift_benchmark/processed/pancancer_modality_coverage_by_cancer.csv`
- `work/data/alignment_domain_shift_benchmark/processed/pancancer_one_vs_rest_mrna_separability.csv`
- `work/data/alignment_domain_shift_benchmark/processed/pancancer_vs_internal_task_difficulty.csv`
- `outputs/second_paper_pancancer_modality_coverage_by_cancer.png`
- `outputs/second_paper_pancancer_cancer_separability_auc.png`
- `outputs/second_paper_pancancer_vs_internal_task_f1.png`
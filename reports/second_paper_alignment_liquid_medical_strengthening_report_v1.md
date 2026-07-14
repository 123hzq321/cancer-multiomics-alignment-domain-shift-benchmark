# Alignment, Liquid Mismatch, and Medical Interpretation Strengthening Report

## Real sample-alignment audit

| task | labelled_rows | unique_sample_ids | duplicate_sample_rows | unique_patient_ids | patients_with_multiple_labelled_samples | complete_mismatch_core_fraction | rppa_sample_observed_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| UCEC molecular subtype | 507 | 507 | 0 | 507 | 0 | 1.0000 | 0.7988 |
| COADREAD molecular subtype | 449 | 449 | 0 | 449 | 0 | 1.0000 | 0.7929 |
| HNSC HPV status | 487 | 487 | 0 | 487 | 0 | 1.0000 | 0.4148 |
| KIRC grade | 504 | 504 | 0 | 504 | 0 | 0.9881 | 0.8948 |
| PRAD pathologic T stage | 487 | 487 | 0 | 487 | 0 | 0.9877 | 0.7084 |

## Liquid-family train-mismatch degradation

| task | model | f1_macro_mean_aligned | f1_macro_mean_full_mismatch | f1_drop |
| --- | --- | --- | --- | --- |
| COADREAD molecular subtype | small-Liquid/CfC | 0.8041 | 0.5639 | 0.2402 |
| UCEC molecular subtype | Liquid/CfC | 0.8982 | 0.6587 | 0.2396 |
| UCEC molecular subtype | small-Liquid/CfC | 0.8487 | 0.6257 | 0.2230 |
| COADREAD molecular subtype | Liquid/CfC | 0.8382 | 0.7010 | 0.1372 |
| HNSC HPV status | small-Liquid/CfC | 0.9236 | 0.7996 | 0.1240 |
| HNSC HPV status | Liquid/CfC | 0.9445 | 0.8259 | 0.1187 |
| PRAD pathologic T stage | small-Liquid/CfC | 0.6667 | 0.5714 | 0.0953 |
| KIRC grade | small-Liquid/CfC | 0.6601 | 0.5959 | 0.0642 |
| PRAD pathologic T stage | Liquid/CfC | 0.6690 | 0.6247 | 0.0443 |
| KIRC grade | Liquid/CfC | 0.6529 | 0.6317 | 0.0212 |

## Medical interpretation table

| task | medical_endpoint | dominant_biology | why_multiomics_relevant | model_interpretation |
| --- | --- | --- | --- | --- |
| UCEC molecular subtype | Endometrial cancer molecular subtype: CN-high, CN-low, MSI, POLE | Copy-number burden, mismatch repair deficiency, polymerase-epsilon proofreading mutations | Subtype labels are explicitly multi-mechanistic; CNA, mutation, methylation, and expression can provide complementary signals. | Liquid/CfC was near-best, consistent with potential benefit from cross-modality integration. |
| COADREAD molecular subtype | Colorectal cancer molecular subtype: CIN, MSI, genome-stable | Chromosomal instability, microsatellite instability, and epigenetic/immune-associated programs | The endpoint combines genome instability and expression/methylation programs; modality mismatch strongly degraded performance. | small-Liquid/CfC was near-best, but ExtraTrees remained the top mean-F1 model. |
| HNSC HPV status | Head and neck squamous cancer HPV-positive versus HPV-negative status | Viral oncogenesis with strong transcriptional and cell-cycle signatures | The signal may already be high in expression space, so complex cross-modality modeling is not always necessary. | Elastic-net logistic regression was strongest; Liquid/CfC was weaker, suggesting a simpler high-signal boundary. |
| KIRC grade | Clear-cell renal cancer low versus high histologic grade | Tumor aggressiveness, hypoxia/angiogenesis, chromatin remodeling, and VHL-related biology | Grade is partly histopathologic and may not be fully captured by bulk molecular features alone. | Random Forest was strongest; pathology imaging may be a more natural complementary modality. |
| PRAD pathologic T stage | Prostate cancer pathologic T2 versus T3/T4 stage | Local invasion and anatomic extension rather than a purely molecular phenotype | Molecular signal is expected to be modest; imaging/pathology or clinical variables may improve endpoint relevance. | Liquid/CfC was near-best but all models were close, consistent with limited molecular separability. |

## Generated files

- `work/data/alignment_domain_shift_benchmark/processed/sample_alignment_audit.csv`
- `work/data/alignment_domain_shift_benchmark/processed/liquid_mismatch_raw_results.csv`
- `work/data/alignment_domain_shift_benchmark/processed/liquid_mismatch_summary.csv`
- `work/data/alignment_domain_shift_benchmark/processed/liquid_mismatch_degradation.csv`
- `work/data/alignment_domain_shift_benchmark/processed/medical_interpretation_table.csv`
- `outputs/second_paper_liquid_mismatch_degradation.png`
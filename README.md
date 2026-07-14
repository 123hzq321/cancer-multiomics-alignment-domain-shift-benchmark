# Cancer Multi-Omics Alignment and Domain-Shift Benchmark

This repository supports the manuscript:

**Sample Alignment and Cohort Shift as Determinants of Cancer Multi-Omics Prediction Reliability: A Benchmark and Stress-Test Study**

The repository is intentionally data-centric. Liquid/CfC is included as one neural fusion comparator, not as the main claim of the study. The central benchmark asks how sample alignment, modality coverage, sample-mismatch stress tests, model-family baselines, and BRCA external cohort shift affect the reliability of cancer multi-omics prediction.

## Study Structure

| Tier | Data | Purpose |
| --- | --- | --- |
| Primary benchmark | Five TCGA cancer-internal tasks | Sample alignment and modality coverage audit |
| Stress test | Same five tasks | Simulated multi-omics sample mismatch |
| Model-family benchmark | Same five tasks | Seven-model 10-fold comparison with effect-size-oriented diagnostics |
| Exploratory scope screen | 10 TCGA PanCancer cohorts | Candidate endpoint landscape and apparent difficulty |
| External case study | BRCA source/external cohorts | Source-to-external domain-shift example |

## Repository Layout

- `scripts/`: analysis and model-training scripts.
- `processed_results/`: CSV outputs used to generate manuscript tables and figures.
- `figures/`: generated figures used in the manuscript.
- `manuscript/`: LaTeX manuscript source.
- `reports/`: markdown analysis summaries.
- `requirements.txt`: Python package requirements used in the analysis environment.
- `REPRODUCING.md`: practical reproduction notes.

## Data Sources

All source datasets are public. TCGA PanCancer Atlas, TCGA-BRCA, METABRIC, CPTAC 2020, and SMC 2018 data were accessed through cBioPortal where available. GSE96058/SCAN-B data were accessed through GEO and associated public processed files. Public data and metadata were accessed and processed through 14 July 2026.

Large source molecular matrices are not redistributed here. The repository includes processed result tables needed to reproduce the manuscript tables and figures.

## Main Scripts

- `scripts/analyze_alignment_domain_shift_benchmark.py`
- `scripts/strengthen_alignment_liquid_medical_evidence.py`
- `scripts/reviewer_cv_noise_significance_analysis.py`
- `scripts/analyze_multicancer_reliability_extension.py`
- `scripts/expand_multicancer_clinical_scope.py`

## Citation

If this repository is used before journal publication, please cite the manuscript title and repository URL. A Zenodo DOI should be added after public archiving.

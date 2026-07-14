# Reproducing the Manuscript Artifacts

## Environment

The analysis environment used for the current manuscript build was:

- Python 3.10.11
- numpy 2.2.6
- pandas 2.2.3
- scikit-learn 1.7.2
- scipy 1.15.3
- matplotlib 3.10.8
- torch 2.10.0+cu128
- pyreadr 0.5.6

Install the Python dependencies with:

```bash
python -m pip install -r requirements.txt
```

## Manuscript Tables and Figures

The `processed_results/` and `figures/` directories contain the derived tables and figures used by the manuscript. The manuscript can be compiled from `manuscript/` with:

```bash
cd manuscript
pdflatex main.tex
pdflatex main.tex
```

## Re-running Analyses

The scripts assume that public source data have been downloaded or regenerated into the expected `work/data/` structure used during the project. The processed results included in this repository are sufficient for reviewer inspection and manuscript figure/table verification.

Core workflow scripts:

```bash
python scripts/analyze_alignment_domain_shift_benchmark.py
python scripts/strengthen_alignment_liquid_medical_evidence.py
python scripts/reviewer_cv_noise_significance_analysis.py --folds 10
python scripts/analyze_multicancer_reliability_extension.py
python scripts/expand_multicancer_clinical_scope.py --folds 3 --max-features 300
```

Because the source datasets are public but externally hosted, exact end-to-end re-execution may depend on current cBioPortal/GEO availability and API behavior.

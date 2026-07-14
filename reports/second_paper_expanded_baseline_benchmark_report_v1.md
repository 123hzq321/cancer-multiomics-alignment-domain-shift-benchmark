# Expanded Baseline Benchmark for the Alignment/Domain-Shift Manuscript

## Scope

This report strengthens the second manuscript by adding a seven-model, 10-fold cross-validation baseline benchmark on the five TCGA cancer-internal tasks. The model set includes Logistic ElasticNet, Random Forest, ExtraTrees, HistGradientBoosting, MLP, Liquid/CfC, and small-Liquid/CfC.

## Best model per task

| Task | Best model | Folds | Macro F1 | F1 CI95 low | F1 CI95 high | ROC-AUC |
| --- | --- | --- | --- | --- | --- | --- |
| COADREAD subtype | ExtraTrees | 10 | 0.8676 | 0.8247 | 0.9106 | 0.9554 |
| HNSC HPV | Logistic | 10 | 0.9469 | 0.9235 | 0.9702 | 0.9611 |
| KIRC grade | RandomForest | 10 | 0.6784 | 0.6521 | 0.7047 | 0.7335 |
| PRAD T stage | RandomForest | 10 | 0.6839 | 0.6298 | 0.7380 | 0.7734 |
| UCEC subtype | ExtraTrees | 10 | 0.8944 | 0.8524 | 0.9364 | 0.9831 |

## Top-vs-runner-up paired significance

| Task | Top model | Runner-up | Mean F1 diff | Diff CI95 low | Diff CI95 high | Raw p | Holm p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| COADREAD subtype | ExtraTrees | small-Liquid | 0.0048 | -0.0431 | 0.0514 | 1.0000 | 1.0000 |
| HNSC HPV | Logistic | HistGB | 0.0159 | -0.0141 | 0.0470 | 0.3750 | 1.0000 |
| KIRC grade | RandomForest | ExtraTrees | 0.0122 | -0.0057 | 0.0310 | 0.3125 | 1.0000 |
| PRAD T stage | RandomForest | HistGB | 0.0021 | -0.0451 | 0.0389 | 0.4316 | 1.0000 |
| UCEC subtype | ExtraTrees | Logistic | 0.0053 | -0.0391 | 0.0471 | 0.6953 | 1.0000 |

## Significant top-vs-all comparisons after within-task Holm correction

| Task | Top model | Comparator | Mean F1 diff | Diff CI95 low | Diff CI95 high | Holm p |
| --- | --- | --- | --- | --- | --- | --- |
| HNSC HPV | Logistic | MLP | 0.0939 | 0.0444 | 0.1382 | 0.0469 |

## Noise elbow for the best model per task

| Task | Model | Clean F1 | F1 at noise 0.20 | F1 at noise 0.50 | First 5% drop sigma | Elbow sigma |
| --- | --- | --- | --- | --- | --- | --- |
| COADREAD subtype | ExtraTrees | 0.8676 | 0.8405 | 0.6821 | 0.3000 | 0.3000 |
| HNSC HPV | Logistic | 0.9469 | 0.9433 | 0.9345 |  | 0.1000 |
| KIRC grade | RandomForest | 0.6784 | 0.5823 | 0.4214 | 0.1000 | 0.0500 |
| PRAD T stage | RandomForest | 0.6839 | 0.3812 | 0.3812 | 0.0500 | 0.2000 |
| UCEC subtype | ExtraTrees | 0.8944 | 0.8978 | 0.8697 |  | 0.3000 |

## Interpretation

The expanded benchmark changes the manuscript from a two-baseline stress test into a proper model-family benchmark. The best models differ by task: ExtraTrees leads UCEC and COADREAD, Logistic ElasticNet leads HNSC HPV status, and Random Forest leads KIRC grade and PRAD pathologic T stage. Top-vs-runner-up differences remain small in all five tasks and do not survive Holm correction. The only significant top-vs-all comparison after within-task correction is HNSC Logistic ElasticNet versus MLP, which supports the more cautious conclusion that task difficulty and data reliability dominate over universal architecture superiority.

## Generated files

- `work/data/alignment_domain_shift_benchmark/processed/expanded_tenfold_baseline_summary.csv`
- `work/data/alignment_domain_shift_benchmark/processed/expanded_tenfold_top_runner_significance.csv`
- `work/data/alignment_domain_shift_benchmark/processed/expanded_tenfold_top_vs_all_significance.csv`
- `work/data/alignment_domain_shift_benchmark/processed/expanded_tenfold_noise_elbow_summary.csv`
- `outputs/second_paper_expanded_baseline_cv_heatmap.png`
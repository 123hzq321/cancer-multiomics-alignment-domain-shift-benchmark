from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(".")
PANCANCER_TABLE = ROOT / "work/data/tcga_pancancer_cbioportal/processed/pancancer_multimodal_impact468_table.csv"
PANCANCER_MODEL_SUMMARY = ROOT / "work/data/tcga_pancancer_cbioportal/processed/pancancer_cancer_type_model_summary.csv"
INTERNAL_BEST = ROOT / "work/data/multicancer_internal_benchmark/processed/multicancer_internal_best_by_task.csv"
OUT_DIR = ROOT / "work/data/alignment_domain_shift_benchmark/processed"
REPORT_DIR = ROOT / "outputs"

MODALITIES = ["mrna", "gistic", "log2cna", "methylation", "rppa", "mutation"]


def modality_columns(frame: pd.DataFrame, modality: str) -> list[str]:
    return sorted([col for col in frame.columns if col.startswith(f"{modality}__")])


def markdown_table(frame: pd.DataFrame, floatfmt: str = ".4f") -> str:
    if frame.empty:
        return ""
    clean = frame.copy()
    for col in clean.columns:
        if pd.api.types.is_float_dtype(clean[col]):
            clean[col] = clean[col].map(lambda x: "" if pd.isna(x) else format(float(x), floatfmt))
    cols = list(clean.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in clean.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    return "\n".join(lines)


def coverage_by_cancer(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cancer, group in frame.groupby("cancer_type"):
        for modality in MODALITIES:
            cols = modality_columns(frame, modality)
            x = group[cols].apply(pd.to_numeric, errors="coerce")
            rows.append(
                {
                    "cancer_type": cancer,
                    "modality": modality,
                    "samples": len(group),
                    "n_features": len(cols),
                    "rows_with_any_observed_feature": int(x.notna().any(axis=1).sum()),
                    "feature_observed_fraction": float(x.notna().to_numpy().mean()),
                    "feature_missing_fraction": float(x.isna().to_numpy().mean()),
                }
            )
    return pd.DataFrame(rows)


def one_vs_rest_separability(frame: pd.DataFrame, modality: str = "mrna") -> pd.DataFrame:
    cols = modality_columns(frame, modality)
    x = frame[cols].apply(pd.to_numeric, errors="coerce")
    y_cancer = frame["cancer_type"].astype(str).to_numpy()
    rows: list[dict[str, object]] = []
    for cancer in sorted(frame["cancer_type"].unique()):
        y = (y_cancer == cancer).astype(int)
        min_class = int(np.bincount(y).min())
        folds = max(2, min(5, min_class))
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        solver="lbfgs",
                        random_state=42,
                    ),
                ),
            ]
        )
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        scores = cross_val_score(model, x, y, cv=cv, scoring="roc_auc", n_jobs=None)
        rows.append(
            {
                "cancer_type": cancer,
                "modality": modality,
                "positive_samples": int(y.sum()),
                "negative_samples": int((1 - y).sum()),
                "one_vs_rest_auc_mean": float(scores.mean()),
                "one_vs_rest_auc_sd": float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def task_difficulty_summary() -> pd.DataFrame:
    pan = pd.read_csv(PANCANCER_MODEL_SUMMARY)
    internal = pd.read_csv(INTERNAL_BEST)
    rows: list[dict[str, object]] = []
    best_pan = pan.sort_values("f1_macro_mean", ascending=False).iloc[0]
    rows.append(
        {
            "analysis": "PanCancer cancer-type classification",
            "task": "10 cancer types",
            "best_model": best_pan["model"],
            "macro_f1_mean": float(best_pan["f1_macro_mean"]),
            "macro_f1_sd": float(best_pan.get("f1_macro_sd", 0.0)),
            "task_family": "between-cancer",
        }
    )
    for _, row in internal.iterrows():
        rows.append(
            {
                "analysis": "Within-cancer endpoint prediction",
                "task": row["task"],
                "best_model": row["model"],
                "macro_f1_mean": float(row["f1_macro_mean"]),
                "macro_f1_sd": float(row.get("f1_macro_sd", 0.0)),
                "task_family": "within-cancer",
            }
        )
    return pd.DataFrame(rows).sort_values(["task_family", "macro_f1_mean"], ascending=[True, False])


def make_figures(coverage: pd.DataFrame, separability: pd.DataFrame, difficulty: pd.DataFrame) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("default")

    pivot = coverage.pivot(index="cancer_type", columns="modality", values="feature_observed_fraction").loc[:, MODALITIES]
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    im = ax.imshow(pivot.to_numpy(), vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(np.arange(pivot.shape[1]), labels=pivot.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(pivot.shape[0]), labels=pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            ax.text(j, i, f"{value * 100:.0f}%", ha="center", va="center", color="white" if value < 0.6 else "black", fontsize=7)
    ax.set_title("PanCancer modality coverage by cancer type")
    fig.colorbar(im, ax=ax, label="Feature observed fraction")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "second_paper_pancancer_modality_coverage_by_cancer.png", dpi=220)
    plt.close(fig)

    sep = separability.sort_values("one_vs_rest_auc_mean")
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.barh(sep["cancer_type"], sep["one_vs_rest_auc_mean"], color="#4C78A8")
    ax.set_xlim(0.5, 1.01)
    ax.set_xlabel("One-vs-rest AUC using mRNA features")
    ax.set_title("Cancer identity separability in PanCancer expression space")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "second_paper_pancancer_cancer_separability_auc.png", dpi=220)
    plt.close(fig)

    diff = difficulty.copy()
    colors = diff["task_family"].map({"between-cancer": "#F58518", "within-cancer": "#4C78A8"}).fillna("#4C78A8")
    labels = diff["task"].str.replace("_", "\n")
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    ax.bar(range(len(diff)), diff["macro_f1_mean"], yerr=diff["macro_f1_sd"].fillna(0), color=colors, capsize=3)
    ax.set_xticks(range(len(diff)), labels=labels, rotation=0, fontsize=8)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Best-model macro F1")
    ax.set_title("Between-cancer classification is easier than within-cancer endpoints")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "second_paper_pancancer_vs_internal_task_f1.png", dpi=220)
    plt.close(fig)


def write_report(coverage: pd.DataFrame, separability: pd.DataFrame, difficulty: pd.DataFrame) -> None:
    coverage_summary = (
        coverage.groupby("modality")
        .agg(
            cancer_types=("cancer_type", "nunique"),
            observed_fraction_mean=("feature_observed_fraction", "mean"),
            observed_fraction_min=("feature_observed_fraction", "min"),
            observed_fraction_max=("feature_observed_fraction", "max"),
        )
        .reset_index()
        .sort_values("observed_fraction_mean", ascending=False)
    )
    lines = [
        "# Multi-Cancer Reliability Extension v0",
        "",
        "## Purpose",
        "",
        "This extension strengthens the second manuscript by adding a 10-cancer TCGA PanCancer audit. It separates three facts that can otherwise be conflated: cancer-type classification is easy, within-cancer clinical or molecular endpoints are harder, and modality coverage varies strongly by assay.",
        "",
        "## PanCancer modality coverage summary",
        "",
        markdown_table(coverage_summary),
        "",
        "## Cancer identity separability",
        "",
        markdown_table(separability.sort_values("one_vs_rest_auc_mean", ascending=False)),
        "",
        "## Between-cancer versus within-cancer difficulty",
        "",
        markdown_table(difficulty),
        "",
        "## Interpretation",
        "",
        "The PanCancer cancer-type task reaches very high macro F1 in the existing benchmark, and one-vs-rest mRNA classifiers separate every cancer type with high AUC. This supports the argument that cancer identity is a dominant signal in multi-omics data. In contrast, clinically meaningful within-cancer endpoints show lower and more variable macro F1. Therefore, high pan-cancer classification performance should not be interpreted as evidence that multi-omics models are robust for clinically relevant within-cancer prediction. The coverage audit also shows that RPPA remains the sparsest modality at the PanCancer level, reinforcing the need for modality coverage checks before early fusion.",
        "",
        "## Generated files",
        "",
        "- `work/data/alignment_domain_shift_benchmark/processed/pancancer_modality_coverage_by_cancer.csv`",
        "- `work/data/alignment_domain_shift_benchmark/processed/pancancer_one_vs_rest_mrna_separability.csv`",
        "- `work/data/alignment_domain_shift_benchmark/processed/pancancer_vs_internal_task_difficulty.csv`",
        "- `outputs/second_paper_pancancer_modality_coverage_by_cancer.png`",
        "- `outputs/second_paper_pancancer_cancer_separability_auc.png`",
        "- `outputs/second_paper_pancancer_vs_internal_task_f1.png`",
    ]
    (REPORT_DIR / "second_paper_multicancer_reliability_extension_v0.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(PANCANCER_TABLE, low_memory=False)
    coverage = coverage_by_cancer(frame)
    separability = one_vs_rest_separability(frame, "mrna")
    difficulty = task_difficulty_summary()

    coverage.to_csv(OUT_DIR / "pancancer_modality_coverage_by_cancer.csv", index=False)
    separability.to_csv(OUT_DIR / "pancancer_one_vs_rest_mrna_separability.csv", index=False)
    difficulty.to_csv(OUT_DIR / "pancancer_vs_internal_task_difficulty.csv", index=False)
    make_figures(coverage, separability, difficulty)
    write_report(coverage, separability, difficulty)
    print(REPORT_DIR / "second_paper_multicancer_reliability_extension_v0.md")
    print(OUT_DIR / "pancancer_modality_coverage_by_cancer.csv")
    print(OUT_DIR / "pancancer_one_vs_rest_mrna_separability.csv")


if __name__ == "__main__":
    main()

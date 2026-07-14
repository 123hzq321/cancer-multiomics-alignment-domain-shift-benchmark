from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


ROOT = Path(".")
OUT_DIR = ROOT / "work/data/alignment_domain_shift_benchmark/processed"
REPORT_DIR = ROOT / "outputs"

TASK_FILES = {
    "UCEC_molecular_subtype": ROOT / "work/data/multicancer_internal_benchmark/processed/UCEC_molecular_subtype_table.csv",
    "COADREAD_molecular_subtype": ROOT / "work/data/multicancer_internal_benchmark/processed/COADREAD_molecular_subtype_table.csv",
    "HNSC_HPV_status": ROOT / "work/data/multicancer_internal_benchmark/processed/HNSC_HPV_status_table.csv",
    "KIRC_grade_binary": ROOT / "work/data/multicancer_internal_benchmark/processed/KIRC_grade_binary_table.csv",
    "PRAD_pathologic_T_stage": ROOT / "work/data/multicancer_internal_benchmark/processed/PRAD_pathologic_T_stage_table.csv",
}

BRCA_TRAIN = ROOT / "work/data/joined_metabric_external_validation/processed/tcga_metabric_joined_training_table.csv"
BRCA_EXTERNAL = {
    "CPTAC_2020": ROOT / "work/data/joined_metabric_external_validation/processed/cptac_2020_marker_external_table.csv",
    "SMC_2018": ROOT / "work/data/joined_metabric_external_validation/processed/smc_2018_marker_external_table.csv",
    "SCANB_GSE96058": ROOT / "work/data/joined_metabric_external_validation/processed/scanb_gse96058_marker_mrna_external_table.csv",
}

ALL_MODALITIES = ["mrna", "gistic", "log2cna", "methylation", "rppa", "mutation"]
CORE_MODALITIES = ["mrna", "gistic", "methylation", "mutation"]
MISMATCH_MODALITIES = ["gistic", "methylation", "mutation"]


def parse_list(text: str, cast=str) -> list:
    return [cast(item.strip()) for item in text.split(",") if item.strip()]


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


def modality_columns(frame: pd.DataFrame, modalities: list[str]) -> dict[str, list[str]]:
    cols = {m: sorted([col for col in frame.columns if col.startswith(f"{m}__")]) for m in modalities}
    missing = [m for m, values in cols.items() if not values]
    if missing:
        raise ValueError(f"Missing modality columns: {missing}")
    return cols


def make_model(model_name: str, seed: int, n_classes: int) -> Pipeline:
    if model_name == "logistic_l2":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1500,
                        class_weight="balanced",
                        solver="lbfgs",
                        random_state=seed,
                    ),
                ),
            ]
        )
    if model_name == "extra_trees":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=180,
                        max_features="sqrt",
                        min_samples_leaf=2,
                        class_weight="balanced",
                        n_jobs=-1,
                        random_state=seed,
                    ),
                ),
            ]
        )
    raise KeyError(model_name)


def compute_metrics(y_true: np.ndarray, proba: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    y_pred = proba.argmax(axis=1)
    metrics = {
        "n": float(len(y_true)),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    try:
        if len(labels) == 2:
            metrics["roc_auc"] = roc_auc_score(y_true, proba[:, 1])
        else:
            metrics["roc_auc"] = roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=labels)
    except ValueError:
        metrics["roc_auc"] = np.nan
    return metrics


def safe_predict_proba(model: Pipeline, x: pd.DataFrame, labels: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(x)
    # Some folds can miss a class in pathological cases; expand probabilities defensively.
    class_values = model.named_steps["model"].classes_
    if len(class_values) == len(labels) and np.array_equal(class_values, labels):
        return proba
    expanded = np.zeros((len(x), len(labels)), dtype=float)
    for idx, cls in enumerate(class_values):
        target_idx = int(np.where(labels == cls)[0][0])
        expanded[:, target_idx] = proba[:, idx]
    return expanded


def misalign_modalities(
    frame: pd.DataFrame,
    cols_by_modality: dict[str, list[str]],
    modalities_to_shuffle: list[str],
    rate: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if rate <= 0:
        return frame.copy()
    out = frame.reset_index(drop=True).copy()
    n = len(out)
    k = int(round(rate * n))
    if k <= 1:
        return out
    affected = rng.choice(n, size=k, replace=False)
    for modality in modalities_to_shuffle:
        cols = cols_by_modality[modality]
        permuted = affected.copy()
        rng.shuffle(permuted)
        out.loc[affected, cols] = out.loc[permuted, cols].to_numpy()
    return out


def run_misalignment_benchmark(seeds: list[int], rates: list[float], models: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    for task, path in TASK_FILES.items():
        print(f"Misalignment task: {task}", flush=True)
        frame = pd.read_csv(path, low_memory=False)
        frame = frame[frame["task_label"].notna()].reset_index(drop=True)
        cols_by_modality = modality_columns(frame, CORE_MODALITIES)
        coverage_cols_by_modality = modality_columns(frame, ALL_MODALITIES)
        for modality, cols in coverage_cols_by_modality.items():
            x = frame[cols].apply(pd.to_numeric, errors="coerce")
            coverage_rows.append(
                {
                    "task": task,
                    "modality": modality,
                    "n_features": len(cols),
                    "rows_with_any_observed_feature": int(x.notna().any(axis=1).sum()),
                    "labelled_rows": len(frame),
                    "feature_missing_fraction": float(x.isna().to_numpy().mean()),
                }
            )
        le = LabelEncoder()
        y_all = le.fit_transform(frame["task_label"].astype(str))
        labels = np.arange(len(le.classes_))
        all_cols = [col for modality in CORE_MODALITIES for col in cols_by_modality[modality]]
        mrna_cols = cols_by_modality["mrna"]
        for seed in seeds:
            train_idx, test_idx = train_test_split(
                np.arange(len(frame)),
                train_size=0.70,
                random_state=seed,
                stratify=y_all,
            )
            train = frame.iloc[train_idx].reset_index(drop=True)
            test = frame.iloc[test_idx].reset_index(drop=True)
            y_train = le.transform(train["task_label"].astype(str))
            y_test = le.transform(test["task_label"].astype(str))
            for model_name in models:
                # mRNA-only benchmark; this defines the single-modality anchor.
                model = make_model(model_name, seed, len(labels))
                model.fit(train[mrna_cols], y_train)
                metrics = compute_metrics(y_test, safe_predict_proba(model, test[mrna_cols], labels), labels)
                rows.append(
                    {
                        "task": task,
                        "seed": seed,
                        "model": model_name,
                        "scenario": "mrna_only",
                        "mismatch_rate": 0.0,
                        "feature_set": "mrna",
                        **metrics,
                    }
                )
                for rate in rates:
                    rng = np.random.default_rng(seed * 1000 + int(rate * 1000) + 17)
                    for scenario in ["train_misaligned", "test_misaligned"]:
                        if scenario == "train_misaligned":
                            train_use = misalign_modalities(train, cols_by_modality, MISMATCH_MODALITIES, rate, rng)
                            test_use = test
                        else:
                            train_use = train
                            test_use = misalign_modalities(test, cols_by_modality, MISMATCH_MODALITIES, rate, rng)
                        model = make_model(model_name, seed, len(labels))
                        model.fit(train_use[all_cols], y_train)
                        metrics = compute_metrics(y_test, safe_predict_proba(model, test_use[all_cols], labels), labels)
                        rows.append(
                            {
                                "task": task,
                                "seed": seed,
                                "model": model_name,
                                "scenario": scenario,
                                "mismatch_rate": rate,
                                "feature_set": "multiomics_core",
                                **metrics,
                            }
                        )
    return pd.DataFrame(rows), pd.DataFrame(coverage_rows)


def brca_mrna_columns(*frames: pd.DataFrame) -> list[str]:
    common: set[str] | None = None
    for frame in frames:
        cols = {col for col in frame.columns if col.startswith("mrna_z__")}
        common = cols if common is None else common & cols
    if not common:
        raise ValueError("No common mRNA marker columns.")
    return sorted(common)


def fit_domain_shift(
    source: pd.DataFrame,
    external: pd.DataFrame,
    feature_cols: list[str],
    seed: int,
) -> dict[str, float]:
    src = source[feature_cols].apply(pd.to_numeric, errors="coerce")
    ext = external[feature_cols].apply(pd.to_numeric, errors="coerce")
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    src_z = scaler.fit_transform(imputer.fit_transform(src))
    ext_z = scaler.transform(imputer.transform(ext))
    shift = float(np.nanmean(np.abs(ext_z.mean(axis=0) - src_z.mean(axis=0))))
    x = np.vstack([src_z, ext_z])
    y = np.array([0] * len(src_z) + [1] * len(ext_z))
    min_class = int(np.bincount(y).min())
    folds = max(2, min(5, min_class))
    domain_model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    raw_auc = cross_val_score(domain_model, x, y, cv=StratifiedKFold(folds, shuffle=True, random_state=seed), scoring="roc_auc").mean()
    separability_auc = max(float(raw_auc), float(1.0 - raw_auc))
    return {
        "mean_abs_standardized_shift": shift,
        "raw_domain_classifier_auc": float(raw_auc),
        "domain_classifier_auc": separability_auc,
    }


def run_brca_domain_shift(seeds: list[int], models: list[str]) -> pd.DataFrame:
    source = pd.read_csv(BRCA_TRAIN, low_memory=False)
    external_frames = {name: pd.read_csv(path, low_memory=False) for name, path in BRCA_EXTERNAL.items()}
    feature_cols = brca_mrna_columns(source, *external_frames.values())
    source = source[source["subtype_label"].notna()].reset_index(drop=True)
    if "joint_split" in source.columns:
        train_source = source[source["joint_split"].eq("train")].reset_index(drop=True)
        valid_source = source[source["joint_split"].eq("valid")].reset_index(drop=True)
    else:
        train_source, valid_source = train_test_split(
            source,
            train_size=0.85,
            random_state=42,
            stratify=source["subtype_label"],
        )
        train_source = train_source.reset_index(drop=True)
        valid_source = valid_source.reset_index(drop=True)
    le = LabelEncoder()
    y_train = le.fit_transform(train_source["subtype_label"].astype(str))
    labels = np.arange(len(le.classes_))
    rows: list[dict[str, object]] = []
    shift_cache = {
        name: fit_domain_shift(train_source, frame[frame["subtype_label"].isin(le.classes_)].copy(), feature_cols, 42)
        for name, frame in external_frames.items()
    }
    for seed in seeds:
        y_valid = le.transform(valid_source["subtype_label"].astype(str))
        for model_name in models:
            model = make_model(model_name, seed, len(labels))
            model.fit(train_source[feature_cols], y_train)
            valid_metrics = compute_metrics(y_valid, safe_predict_proba(model, valid_source[feature_cols], labels), labels)
            rows.append(
                {
                    "dataset": "TCGA_METABRIC_valid",
                    "seed": seed,
                    "model": model_name,
                    "n_features": len(feature_cols),
                    "mean_abs_standardized_shift": 0.0,
                    "domain_classifier_auc": 0.5,
                    **valid_metrics,
                }
            )
            for name, frame in external_frames.items():
                ext = frame[frame["subtype_label"].isin(le.classes_)].copy().reset_index(drop=True)
                y_ext = le.transform(ext["subtype_label"].astype(str))
                metrics = compute_metrics(y_ext, safe_predict_proba(model, ext[feature_cols], labels), labels)
                rows.append(
                    {
                        "dataset": name,
                        "seed": seed,
                        "model": model_name,
                        "n_features": len(feature_cols),
                        **shift_cache[name],
                        **metrics,
                    }
                )
    return pd.DataFrame(rows)


def summarize_misalignment(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.groupby(["task", "model", "scenario", "feature_set", "mismatch_rate"])
        .agg(
            runs=("seed", "nunique"),
            n_mean=("n", "mean"),
            f1_macro_mean=("f1_macro", "mean"),
            f1_macro_sd=("f1_macro", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            roc_auc_mean=("roc_auc", "mean"),
        )
        .reset_index()
        .sort_values(["task", "model", "scenario", "mismatch_rate"])
    )


def summarize_domain_shift(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.groupby(["dataset", "model"])
        .agg(
            runs=("seed", "nunique"),
            n_mean=("n", "mean"),
            f1_macro_mean=("f1_macro", "mean"),
            f1_macro_sd=("f1_macro", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            roc_auc_mean=("roc_auc", "mean"),
            mean_abs_standardized_shift=("mean_abs_standardized_shift", "mean"),
            domain_classifier_auc=("domain_classifier_auc", "mean"),
            raw_domain_classifier_auc=("raw_domain_classifier_auc", "mean"),
        )
        .reset_index()
        .sort_values(["model", "dataset"])
    )


def make_figures(mis_summary: pd.DataFrame, domain_summary: pd.DataFrame, coverage: pd.DataFrame) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("default")
    # Misalignment degradation, averaged over tasks.
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    plot_df = (
        mis_summary[mis_summary["feature_set"].eq("multiomics_core")]
        .groupby(["model", "scenario", "mismatch_rate"], as_index=False)["f1_macro_mean"]
        .mean()
    )
    for (model, scenario), group in plot_df.groupby(["model", "scenario"]):
        label = f"{model}, {scenario.replace('_', ' ')}"
        ax.plot(group["mismatch_rate"], group["f1_macro_mean"], marker="o", linewidth=2, label=label)
    mrna_anchor = (
        mis_summary[mis_summary["feature_set"].eq("mrna")]
        .groupby("model", as_index=False)["f1_macro_mean"]
        .mean()
    )
    for _, row in mrna_anchor.iterrows():
        ax.axhline(row["f1_macro_mean"], linestyle="--", linewidth=1.2, alpha=0.55, label=f"{row['model']} mRNA-only")
    ax.set_xlabel("Fraction of non-mRNA omics rows permuted")
    ax.set_ylabel("Macro F1 averaged over five TCGA tasks")
    ax.set_title("Performance degradation under simulated multi-omics sample mismatch")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "second_paper_misalignment_degradation.png", dpi=220)
    plt.close(fig)

    # Domain shift scatter.
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    scatter_df = domain_summary[~domain_summary["dataset"].eq("TCGA_METABRIC_valid")].copy()
    markers = {"logistic_l2": "o", "extra_trees": "s"}
    for model, group in scatter_df.groupby("model"):
        ax.scatter(
            group["domain_classifier_auc"],
            group["f1_macro_mean"],
            s=70,
            marker=markers.get(model, "o"),
            label=model,
        )
        for _, row in group.iterrows():
            ax.annotate(row["dataset"], (row["domain_classifier_auc"], row["f1_macro_mean"]), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("Domain classifier AUC: source vs external")
    ax.set_ylabel("External macro F1")
    ax.set_title("External performance in relation to cohort separability")
    ax.set_xlim(0.45, 1.02)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "second_paper_domain_shift_scatter.png", dpi=220)
    plt.close(fig)

    # Coverage heatmap.
    cov = coverage.copy()
    cov["observed_feature_fraction"] = 1.0 - cov["feature_missing_fraction"]
    pivot = cov.pivot(index="task", columns="modality", values="observed_feature_fraction").loc[:, ALL_MODALITIES]
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    im = ax.imshow(pivot.to_numpy(), vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(np.arange(pivot.shape[1]), labels=pivot.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(pivot.shape[0]), labels=pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.iloc[i, j] * 100:.1f}%", ha="center", va="center", color="white" if pivot.iloc[i, j] < 0.6 else "black", fontsize=7)
    ax.set_title("Feature-level observed fraction for available multi-omics modalities")
    fig.colorbar(im, ax=ax, label="Observed fraction")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "second_paper_core_modality_coverage_heatmap.png", dpi=220)
    plt.close(fig)


def write_report(mis_summary: pd.DataFrame, domain_summary: pd.DataFrame, coverage: pd.DataFrame) -> None:
    aligned = mis_summary[
        (mis_summary["feature_set"].eq("multiomics_core"))
        & (mis_summary["scenario"].eq("train_misaligned"))
        & (mis_summary["mismatch_rate"].eq(0.0))
    ]
    full_mismatch = mis_summary[
        (mis_summary["feature_set"].eq("multiomics_core"))
        & (mis_summary["scenario"].eq("train_misaligned"))
        & (mis_summary["mismatch_rate"].eq(1.0))
    ]
    degradation = aligned.merge(
        full_mismatch,
        on=["task", "model", "scenario", "feature_set"],
        suffixes=("_aligned", "_full_mismatch"),
    )
    degradation["f1_drop"] = degradation["f1_macro_mean_aligned"] - degradation["f1_macro_mean_full_mismatch"]
    degradation_table = degradation[
        ["task", "model", "f1_macro_mean_aligned", "f1_macro_mean_full_mismatch", "f1_drop"]
    ].sort_values("f1_drop", ascending=False)

    domain_core = domain_summary[
        ["dataset", "model", "f1_macro_mean", "balanced_accuracy_mean", "mean_abs_standardized_shift", "domain_classifier_auc"]
    ].sort_values(["model", "dataset"])

    lines = [
        "# Second Paper Analysis v0: Multi-omics Alignment and Domain Shift",
        "",
        "## Working title",
        "",
        "Sample alignment and cohort shift as determinants of cancer multi-omics prediction reliability: a benchmark and stress-test study",
        "",
        "## Independent angle",
        "",
        "This analysis changes the center of gravity from model architecture ranking to data reliability. Liquid/CfC-style models can remain a comparator, but the proposed second manuscript should focus on whether multi-omics conclusions are stable when sample alignment, modality coverage, and cross-cohort distribution shift are explicitly audited.",
        "",
        "## Core experiments completed in this run",
        "",
        "1. Five TCGA cancer-internal tasks were re-used as sample-aligned multi-omics prediction tasks.",
        "2. A simulated misalignment stress test permuted non-mRNA omics rows at increasing rates while preserving each modality's marginal distribution.",
        "3. A BRCA external validation analysis quantified source-to-external cohort separability using a domain classifier AUC and a standardized mean-shift score.",
        "",
        "## Key degradation table: aligned versus fully misaligned training data",
        "",
        markdown_table(degradation_table, ".4f"),
        "",
        "## BRCA external domain-shift summary",
        "",
        markdown_table(domain_core, ".4f"),
        "",
        "## Core modality coverage",
        "",
        markdown_table(
            coverage.assign(feature_missing_percent=lambda x: (100 * x["feature_missing_fraction"]).map(lambda v: f"{v:.1f}%"))[
                ["task", "modality", "n_features", "rows_with_any_observed_feature", "labelled_rows", "feature_missing_percent"]
            ],
            ".4f",
        ),
        "",
        "## Draft conclusion",
        "",
        "Across five cancer-internal tasks, intentionally disrupting the pairing between mRNA and other omics layers reduced multi-omics performance in a task- and model-dependent manner, even though the marginal distributions of each omics modality were preserved. This supports a data-centric conclusion: multi-omics fusion benchmarks should report sample-level alignment audits and stress tests, not only model scores. In the BRCA external setting, external cohorts were separable from the TCGA/METABRIC source distribution by marker expression profiles, and external performance varied across cohorts despite using the same label space. The second manuscript can therefore argue that sample alignment and domain shift are first-order determinants of reliability in cancer multi-omics prediction.",
        "",
        "## Generated files",
        "",
        "- `work/data/alignment_domain_shift_benchmark/processed/misalignment_raw_results.csv`",
        "- `work/data/alignment_domain_shift_benchmark/processed/misalignment_summary.csv`",
        "- `work/data/alignment_domain_shift_benchmark/processed/brca_domain_shift_raw_results.csv`",
        "- `work/data/alignment_domain_shift_benchmark/processed/brca_domain_shift_summary.csv`",
        "- `outputs/second_paper_misalignment_degradation.png`",
        "- `outputs/second_paper_domain_shift_scatter.png`",
        "- `outputs/second_paper_core_modality_coverage_heatmap.png`",
    ]
    (REPORT_DIR / "second_paper_alignment_domain_shift_report_v0.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--rates", default="0,0.1,0.25,0.5,0.75,1.0")
    parser.add_argument("--models", default="logistic_l2,extra_trees")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    seeds = parse_list(args.seeds, int)
    rates = parse_list(args.rates, float)
    models = parse_list(args.models, str)

    misalignment, coverage = run_misalignment_benchmark(seeds, rates, models)
    domain_shift = run_brca_domain_shift(seeds, models)
    mis_summary = summarize_misalignment(misalignment)
    domain_summary = summarize_domain_shift(domain_shift)

    misalignment.to_csv(OUT_DIR / "misalignment_raw_results.csv", index=False)
    mis_summary.to_csv(OUT_DIR / "misalignment_summary.csv", index=False)
    coverage.to_csv(OUT_DIR / "core_modality_coverage.csv", index=False)
    domain_shift.to_csv(OUT_DIR / "brca_domain_shift_raw_results.csv", index=False)
    domain_summary.to_csv(OUT_DIR / "brca_domain_shift_summary.csv", index=False)

    make_figures(mis_summary, domain_summary, coverage)
    write_report(mis_summary, domain_summary, coverage)
    print(REPORT_DIR / "second_paper_alignment_domain_shift_report_v0.md")
    print(OUT_DIR / "misalignment_summary.csv")
    print(OUT_DIR / "brca_domain_shift_summary.csv")


if __name__ == "__main__":
    main()

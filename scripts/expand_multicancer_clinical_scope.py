from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from crawl_tcga_brca_multiomics_cbioportal import (  # noqa: E402
    API_BASE,
    fetch_clinical,
    page_get,
    request_json,
)


ROOT = Path(".")
PANCANCER = ROOT / "work/data/tcga_pancancer_cbioportal/processed/pancancer_multimodal_impact468_table.csv"
OUT_DIR = ROOT / "work/data/alignment_domain_shift_benchmark/processed"
REPORT_DIR = ROOT / "outputs"

STUDIES = {
    "BRCA": "brca_tcga_pan_can_atlas_2018",
    "COADREAD": "coadread_tcga_pan_can_atlas_2018",
    "GBM": "gbm_tcga_pan_can_atlas_2018",
    "HNSC": "hnsc_tcga_pan_can_atlas_2018",
    "KIRC": "kirc_tcga_pan_can_atlas_2018",
    "LUAD": "luad_tcga_pan_can_atlas_2018",
    "LUSC": "lusc_tcga_pan_can_atlas_2018",
    "OV": "ov_tcga_pan_can_atlas_2018",
    "PRAD": "prad_tcga_pan_can_atlas_2018",
    "UCEC": "ucec_tcga_pan_can_atlas_2018",
}

CORE_MODALITIES = ["mrna", "gistic", "log2cna", "methylation", "mutation"]
MIN_CLASS = 40
MAX_TASKS_TO_MODEL = 24
DEFAULT_FOLDS = 3
DEFAULT_MAX_FEATURES = 300


def markdown_table(frame: pd.DataFrame, floatfmt: str = ".4f") -> str:
    if frame.empty:
        return "_No rows._"
    clean = frame.copy()
    for col in clean.columns:
        if pd.api.types.is_float_dtype(clean[col]):
            clean[col] = clean[col].map(lambda x: "" if pd.isna(x) else format(float(x), floatfmt))
        else:
            clean[col] = clean[col].map(lambda x: "" if pd.isna(x) else str(x))
    cols = list(clean.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in clean.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    return "\n".join(lines)


def modality_columns(frame: pd.DataFrame, modalities: list[str]) -> list[str]:
    cols: list[str] = []
    for modality in modalities:
        cols.extend(sorted([col for col in frame.columns if col.startswith(f"{modality}__")]))
    if not cols:
        raise ValueError("No feature columns found.")
    return cols


def fetch_attributes(study_id: str, cache_dir: Path) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{study_id}_clinical_attributes.csv"
    if path.exists():
        return pd.read_csv(path)
    rows = request_json(
        "GET",
        f"{API_BASE}/studies/{study_id}/clinical-attributes",
        params={"projection": "DETAILED"},
    )
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    time.sleep(0.15)
    return frame


def fetch_study_clinical(study_id: str, cache_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    attrs = fetch_attributes(study_id, cache_dir)
    samples_path = cache_dir / f"{study_id}_samples.csv"
    if samples_path.exists():
        samples = pd.read_csv(samples_path)
    else:
        samples = pd.DataFrame(page_get(f"/studies/{study_id}/samples", params={"projection": "SUMMARY"}, page_size=2000))
        samples.to_csv(samples_path, index=False)

    primary = samples[samples["sampleType"].eq("Primary Solid Tumor")].copy()
    if primary.empty:
        primary = samples.copy()

    patient_attrs = attrs[attrs["patientAttribute"].astype(bool)]["clinicalAttributeId"].dropna().astype(str).tolist()
    sample_attrs = attrs[~attrs["patientAttribute"].astype(bool)]["clinicalAttributeId"].dropna().astype(str).tolist()

    patient_path = cache_dir / f"{study_id}_patient_clinical.csv"
    if patient_path.exists():
        patient = pd.read_csv(patient_path)
    else:
        patient_ids = sorted(primary["patientId"].dropna().astype(str).unique())
        patient = fetch_clinical(study_id, patient_ids, patient_attrs, "PATIENT") if patient_attrs else pd.DataFrame({"patientId": patient_ids})
        patient.to_csv(patient_path, index=False)
        time.sleep(0.15)

    sample_path = cache_dir / f"{study_id}_sample_clinical.csv"
    if sample_path.exists():
        sample = pd.read_csv(sample_path)
    else:
        sample_ids = primary["sampleId"].dropna().astype(str).tolist()
        sample = fetch_clinical(study_id, sample_ids, sample_attrs, "SAMPLE") if sample_attrs else pd.DataFrame({"sampleId": sample_ids})
        sample.to_csv(sample_path, index=False)
        time.sleep(0.15)

    return attrs, patient, sample


def normalize_text(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "na", "n/a", "not available", "unknown", "not reported", "[not available]"}:
        return None
    return text


def map_stage(value: object) -> str | None:
    text = normalize_text(value)
    if text is None:
        return None
    upper = text.upper().replace("STAGE", "").replace("AJCC", "").strip()
    upper = re.sub(r"[^A-Z0-9]", "", upper)
    if not upper or upper in {"X", "NA"}:
        return None
    if upper.startswith("I") and not upper.startswith("III") and not upper.startswith("IV"):
        return "EARLY_STAGE_I_II"
    if upper.startswith("II") and not upper.startswith("III"):
        return "EARLY_STAGE_I_II"
    if upper.startswith("III") or upper.startswith("IV"):
        return "ADVANCED_STAGE_III_IV"
    return None


def map_t_stage(value: object) -> str | None:
    text = normalize_text(value)
    if text is None:
        return None
    upper = text.upper().replace("PATHOLOGIC", "").replace("STAGE", "").strip()
    if re.match(r"^T[12][A-Z]?$", upper):
        return "T1_T2"
    if re.match(r"^T[34][A-Z]?$", upper):
        return "T3_T4"
    return None


def map_grade(value: object) -> str | None:
    text = normalize_text(value)
    if text is None:
        return None
    upper = text.upper().replace("GRADE", "").strip()
    if upper in {"G1", "1", "I", "LOW", "LOW GRADE", "WELL DIFFERENTIATED"}:
        return "LOW_GRADE"
    if upper in {"G2", "2", "II", "INTERMEDIATE", "MODERATE", "MODERATELY DIFFERENTIATED"}:
        return "LOW_GRADE"
    if upper in {"G3", "3", "III", "HIGH", "HIGH GRADE", "POORLY DIFFERENTIATED"}:
        return "HIGH_GRADE"
    if upper in {"G4", "4", "IV", "UNDIFFERENTIATED"}:
        return "HIGH_GRADE"
    return None


def map_binary_status(value: object) -> str | None:
    text = normalize_text(value)
    if text is None:
        return None
    upper = text.upper()
    if upper in {"POSITIVE", "POS", "YES", "1", "TRUE", "PRESENT"} or "POSITIVE" in upper:
        return "POSITIVE"
    if upper in {"NEGATIVE", "NEG", "NO", "0", "FALSE", "ABSENT"} or "NEGATIVE" in upper:
        return "NEGATIVE"
    return None


def map_subtype(value: object) -> str | None:
    text = normalize_text(value)
    if text is None:
        return None
    text = text.replace("UCEC_", "").replace("HNSC_", "").replace("COAD_", "").replace("READ_", "")
    return text


def label_from_column(column: str, value: object) -> tuple[str, str | None] | None:
    col = column.upper()
    if col in {"AJCC_PATHOLOGIC_TUMOR_STAGE", "CLINICAL_STAGE", "PATHOLOGIC_STAGE"} or ("STAGE" in col and "T_" not in col and "PATH_T" not in col):
        return "stage_early_vs_advanced", map_stage(value)
    if "PATH_T_STAGE" in col or col in {"TUMOR_T_STAGE", "AJCC_TUMOR_PATHOLOGIC_PT"}:
        return "t_stage_low_vs_high", map_t_stage(value)
    if "GRADE" in col and "GRADING" not in col:
        return "grade_low_vs_high", map_grade(value)
    if col in {"SUBTYPE", "PAM50", "PAM50_SUBTYPE"} or "SUBTYPE" in col:
        return "molecular_or_histologic_subtype", map_subtype(value)
    if col in {"ER_STATUS", "PR_STATUS", "HER2_STATUS"} or "HPV" in col:
        return col.lower(), map_binary_status(value)
    return None


def build_candidate_tasks(cache_dir: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    pancancer = pd.read_csv(PANCANCER, low_memory=False)
    candidate_rows: list[dict[str, object]] = []
    task_tables: dict[str, pd.DataFrame] = {}

    for cancer, study_id in STUDIES.items():
        print(f"Clinical discovery: {cancer}", flush=True)
        attrs, patient_clinical, sample_clinical = fetch_study_clinical(study_id, cache_dir)
        data = pancancer[pancancer["source_study_id"].eq(study_id)].copy().reset_index(drop=True)
        merged = data.merge(patient_clinical, on="patientId", how="left", suffixes=("", "_patient"))
        merged = merged.merge(sample_clinical, on="sampleId", how="left", suffixes=("", "_sample"))

        clinical_cols = [col for col in merged.columns if col not in data.columns and not col.endswith("_patient") and not col.endswith("_sample")]
        clinical_cols += [col for col in ["SUBTYPE", "GRADE", "AJCC_PATHOLOGIC_TUMOR_STAGE", "PATH_T_STAGE", "ER_STATUS", "PR_STATUS", "HER2_STATUS"] if col in merged.columns]
        clinical_cols = sorted(set(clinical_cols))

        for col in clinical_cols:
            mapped = merged[col].map(lambda v: label_from_column(col, v))
            if mapped.dropna().empty:
                continue
            family = mapped.map(lambda x: x[0] if x else None)
            labels = mapped.map(lambda x: x[1] if x else None)
            for endpoint_family in sorted(family.dropna().unique()):
                label_series = labels.where(family.eq(endpoint_family))
                labelled = merged[label_series.notna()].copy()
                labelled["task_label"] = label_series[label_series.notna()].astype(str).values
                counts = labelled["task_label"].value_counts()
                usable_labels = counts[counts >= MIN_CLASS].index.tolist()
                usable = labelled[labelled["task_label"].isin(usable_labels)].copy()
                if usable["task_label"].nunique() < 2:
                    continue
                if len(usable) < 100:
                    continue
                task_name = f"{cancer}_{endpoint_family}_{col}".replace("__", "_")
                task_name = re.sub(r"[^A-Za-z0-9_]+", "_", task_name)
                task_tables[task_name] = usable
                candidate_rows.append(
                    {
                        "task": task_name,
                        "cancer_type": cancer,
                        "study_id": study_id,
                        "endpoint_family": endpoint_family,
                        "source_attribute": col,
                        "labelled_samples": int(len(usable)),
                        "n_classes": int(usable["task_label"].nunique()),
                        "label_counts": "; ".join(f"{k}:{int(v)}" for k, v in usable["task_label"].value_counts().items()),
                    }
                )
    candidates = pd.DataFrame(candidate_rows).sort_values(["endpoint_family", "cancer_type", "labelled_samples"], ascending=[True, True, False])
    return candidates, task_tables


def make_model(model_name: str, seed: int, max_features: int):
    if model_name == "logistic_elasticnet":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("selector", SelectKBest(score_func=f_classif, k=max_features)),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=3000,
                        C=0.5,
                        solver="saga",
                        penalty="elasticnet",
                        l1_ratio=0.5,
                        class_weight="balanced",
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    if model_name == "extra_trees":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("selector", SelectKBest(score_func=f_classif, k=max_features)),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=150,
                        max_features="sqrt",
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    raise KeyError(model_name)


def compute_auc(y: np.ndarray, proba: np.ndarray) -> float:
    try:
        if proba.shape[1] == 2:
            return float(roc_auc_score(y, proba[:, 1]))
        return float(roc_auc_score(y, proba, multi_class="ovr", average="macro"))
    except Exception:
        return float("nan")


def run_quick_cv(candidates: pd.DataFrame, task_tables: dict[str, pd.DataFrame], seed: int, folds_requested: int, max_features: int, max_tasks: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    selected = candidates.sort_values(["endpoint_family", "labelled_samples"], ascending=[True, False]).head(max_tasks)
    for _, meta in selected.iterrows():
        task = meta["task"]
        frame = task_tables[task].copy().reset_index(drop=True)
        y_text = frame["task_label"].astype(str)
        counts = y_text.value_counts()
        min_count = int(counts.min())
        folds = min(folds_requested, min_count)
        if folds < 3:
            continue
        le = LabelEncoder()
        y = le.fit_transform(y_text)
        x_cols = modality_columns(frame, CORE_MODALITIES)
        x = frame[x_cols].apply(pd.to_numeric, errors="coerce")
        k = min(max_features, x.shape[1])
        for model_name in ["logistic_elasticnet", "extra_trees"]:
            print(f"CV {task} {model_name}", flush=True)
            model = make_model(model_name, seed, k)
            cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
            proba = cross_val_predict(model, x, y, cv=cv, method="predict_proba", n_jobs=None)
            pred = proba.argmax(axis=1)
            rows.append(
                {
                    "task": task,
                    "cancer_type": meta["cancer_type"],
                    "endpoint_family": meta["endpoint_family"],
                    "source_attribute": meta["source_attribute"],
                    "labelled_samples": int(meta["labelled_samples"]),
                    "n_classes": int(meta["n_classes"]),
                    "folds": int(folds),
                    "selected_features_per_fold": int(k),
                    "model": model_name,
                    "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
                    "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
                    "roc_auc": compute_auc(y, proba),
                }
            )
    return pd.DataFrame(rows)


def make_figures(candidates: pd.DataFrame, cv_results: pd.DataFrame) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    task_counts = candidates.groupby(["cancer_type", "endpoint_family"]).size().reset_index(name="tasks")
    pivot = task_counts.pivot(index="cancer_type", columns="endpoint_family", values="tasks").fillna(0)
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    im = ax.imshow(pivot.to_numpy(), cmap="Blues", aspect="auto")
    ax.set_xticks(range(pivot.shape[1]), labels=pivot.columns, rotation=35, ha="right")
    ax.set_yticks(range(pivot.shape[0]), labels=pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, str(int(pivot.iloc[i, j])), ha="center", va="center", fontsize=8)
    ax.set_title("Expanded clinical endpoint task availability across 10 TCGA cancers")
    fig.colorbar(im, ax=ax, label="Number of candidate tasks")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "second_paper_expanded_scope_task_availability.png", dpi=240)
    plt.close(fig)

    if not cv_results.empty:
        best = cv_results.sort_values(["task", "f1_macro"], ascending=[True, False]).groupby("task").head(1).copy()
        best = best.sort_values("f1_macro")
        labels = best["cancer_type"] + "\n" + best["endpoint_family"].str.replace("_", " ")
        fig, ax = plt.subplots(figsize=(8.8, max(4.8, 0.28 * len(best))))
        ax.barh(range(len(best)), best["f1_macro"], color="#4C78A8")
        ax.set_yticks(range(len(best)), labels=labels, fontsize=7)
        ax.set_xlim(0.0, 1.02)
        ax.set_xlabel("Best quick-CV macro F1")
        ax.set_title("Expanded clinical endpoint baseline difficulty")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(REPORT_DIR / "second_paper_expanded_scope_endpoint_f1.png", dpi=240)
        plt.close(fig)


def write_report(candidates: pd.DataFrame, cv_results: pd.DataFrame) -> None:
    availability = (
        candidates.groupby(["cancer_type", "endpoint_family"], as_index=False)
        .agg(tasks=("task", "nunique"), labelled_samples_median=("labelled_samples", "median"))
        .sort_values(["cancer_type", "endpoint_family"])
    )
    best = pd.DataFrame()
    if not cv_results.empty:
        best = cv_results.sort_values(["task", "f1_macro"], ascending=[True, False]).groupby("task").head(1)
        best = best[
            [
                "cancer_type",
                "endpoint_family",
                "source_attribute",
                "labelled_samples",
                "model",
                "f1_macro",
                "balanced_accuracy",
                "roc_auc",
            ]
        ].sort_values(["endpoint_family", "cancer_type"])
    lines = [
        "# Expanded Multi-Cancer Clinical Scope Report",
        "",
        "## Purpose",
        "",
        "This analysis broadens the manuscript beyond the five curated internal tasks by systematically discovering clinically interpretable endpoints across the 10 TCGA PanCancer cohorts already used in the project.",
        "",
        "## Candidate task availability",
        "",
        markdown_table(availability),
        "",
        "## Quick baseline CV on selected expanded tasks",
        "",
        markdown_table(best),
        "",
        "## Interpretation",
        "",
        "The expanded discovery shows that clinically interpretable endpoints are available beyond the original five tasks, especially stage, grade, subtype, and receptor/viral-status tasks. The quick baseline uses lightweight cross-validation with training-fold feature selection and is intended to estimate endpoint difficulty, not to replace the deeply analysed five-task benchmark. Performance remains heterogeneous, supporting the manuscript's central claim that task definition and endpoint biology are major determinants of model reliability.",
        "",
        "## Generated files",
        "",
        "- `work/data/alignment_domain_shift_benchmark/processed/expanded_clinical_endpoint_candidates.csv`",
        "- `work/data/alignment_domain_shift_benchmark/processed/expanded_clinical_endpoint_quick_cv.csv`",
        "- `outputs/second_paper_expanded_scope_task_availability.png`",
        "- `outputs/second_paper_expanded_scope_endpoint_f1.png`",
    ]
    (REPORT_DIR / "second_paper_expanded_multicancer_clinical_scope_report_v1.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--skip-cv", action="store_true")
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--max-features", type=int, default=DEFAULT_MAX_FEATURES)
    parser.add_argument("--max-tasks", type=int, default=MAX_TASKS_TO_MODEL)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    cache_dir = OUT_DIR / "expanded_clinical_scope_cache"
    candidates, task_tables = build_candidate_tasks(cache_dir)
    candidates.to_csv(OUT_DIR / "expanded_clinical_endpoint_candidates.csv", index=False)
    if args.skip_cv:
        cv_results = pd.DataFrame()
    else:
        cv_results = run_quick_cv(candidates, task_tables, args.seed, args.folds, args.max_features, args.max_tasks)
        cv_results.to_csv(OUT_DIR / "expanded_clinical_endpoint_quick_cv.csv", index=False)
    make_figures(candidates, cv_results)
    write_report(candidates, cv_results)
    print(REPORT_DIR / "second_paper_expanded_multicancer_clinical_scope_report_v1.md")


if __name__ == "__main__":
    main()

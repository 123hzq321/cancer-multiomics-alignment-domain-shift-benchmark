from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_tcga_brca_multiomics_baselines_vs_liquid import (  # noqa: E402
    MODALITIES,
    ModalityLiquidCfC,
    MultiOmicsDataset,
    MultiOmicsPreprocessor,
    compute_metrics,
    predict_torch,
    set_seed,
    train_torch_model,
)


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

TASK_LABELS = {
    "UCEC_molecular_subtype": "UCEC molecular subtype",
    "COADREAD_molecular_subtype": "COADREAD molecular subtype",
    "HNSC_HPV_status": "HNSC HPV status",
    "KIRC_grade_binary": "KIRC grade",
    "PRAD_pathologic_T_stage": "PRAD pathologic T stage",
}

ALL_MODALITIES = ["mrna", "gistic", "log2cna", "methylation", "rppa", "mutation"]
MISMATCH_MODALITIES = ["gistic", "methylation", "mutation"]
LIQUID_CORE_MODALITIES = ["mrna", "gistic", "methylation", "mutation"]


class SmallModalityLiquidCfC(ModalityLiquidCfC):
    def __init__(self, input_dims: list[int], num_classes: int):
        super().__init__(input_dims, num_classes, embed_dim=32, hidden_dim=48)


def parse_list(text: str, cast=str) -> list:
    return [cast(item.strip()) for item in text.split(",") if item.strip()]


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


def modality_columns(frame: pd.DataFrame, modalities: list[str]) -> dict[str, list[str]]:
    cols = {m: sorted([col for col in frame.columns if col.startswith(f"{m}__")]) for m in modalities}
    missing = [m for m, values in cols.items() if not values]
    if missing:
        raise ValueError(f"Missing modality columns: {missing}")
    return cols


def rows_with_any_observed(frame: pd.DataFrame, cols: list[str]) -> pd.Series:
    return frame[cols].apply(pd.to_numeric, errors="coerce").notna().any(axis=1)


def sample_alignment_audit() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for task, path in TASK_FILES.items():
        frame = pd.read_csv(path, low_memory=False)
        frame = frame[frame["task_label"].notna()].reset_index(drop=True)
        cols_by_modality = modality_columns(frame, ALL_MODALITIES)
        observed = {modality: rows_with_any_observed(frame, cols) for modality, cols in cols_by_modality.items()}
        core_observed = pd.concat([observed[m] for m in LIQUID_CORE_MODALITIES], axis=1).all(axis=1)
        expanded_core_observed = pd.concat([observed[m] for m in ["mrna", "gistic", "log2cna", "methylation", "mutation"]], axis=1).all(axis=1)
        patient_counts = frame["patientId"].astype(str).value_counts()
        rows.append(
            {
                "task": task,
                "labelled_rows": int(len(frame)),
                "unique_sample_ids": int(frame["sampleId"].astype(str).nunique()),
                "duplicate_sample_rows": int(frame["sampleId"].astype(str).duplicated().sum()),
                "unique_patient_ids": int(frame["patientId"].astype(str).nunique()),
                "patients_with_multiple_labelled_samples": int((patient_counts > 1).sum()),
                "primary_solid_tumor_rows": int(frame["sampleType"].astype(str).eq("Primary Solid Tumor").sum()),
                "rows_with_complete_mismatch_core": int(core_observed.sum()),
                "complete_mismatch_core_fraction": float(core_observed.mean()),
                "rows_with_complete_expanded_core": int(expanded_core_observed.sum()),
                "complete_expanded_core_fraction": float(expanded_core_observed.mean()),
                "rows_with_rppa_observed": int(observed["rppa"].sum()),
                "rppa_sample_observed_fraction": float(observed["rppa"].mean()),
            }
        )
    return pd.DataFrame(rows)


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


def build_processed(
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    modality_cols: dict[str, list[str]],
    label_encoder: LabelEncoder,
) -> dict[str, dict[str, object]]:
    pre = MultiOmicsPreprocessor(modality_cols).fit(train_frame)
    processed: dict[str, dict[str, object]] = {}
    for split_name, frame in [("train", train_frame), ("valid", valid_frame), ("test", test_frame)]:
        frame = frame.copy()
        modalities = pre.transform_modalities(frame)
        processed[split_name] = {
            "frame": frame,
            "modalities": modalities,
            "fusion": np.concatenate(modalities, axis=1).astype(np.float32),
            "y": label_encoder.transform(frame["task_label"].astype(str)).astype(np.int64),
            "sampleId": frame["sampleId"].to_numpy(),
        }
    return processed


def make_loaders(processed: dict[str, dict[str, object]], batch_size: int) -> dict[str, DataLoader]:
    loaders = {}
    for split_name, payload in processed.items():
        dataset = MultiOmicsDataset(payload["fusion"], payload["modalities"], payload["y"])
        loaders[split_name] = DataLoader(dataset, batch_size=batch_size, shuffle=(split_name == "train"))
    return loaders


def make_liquid_model(model_name: str, input_dims: list[int], num_classes: int) -> nn.Module:
    if model_name == "liquid_cfc_modality_sequence":
        return ModalityLiquidCfC(input_dims, num_classes, embed_dim=64, hidden_dim=96)
    if model_name == "small_liquid_cfc_modality_sequence":
        return SmallModalityLiquidCfC(input_dims, num_classes)
    raise KeyError(model_name)


def evaluate_torch_model(model: nn.Module, processed: dict[str, dict[str, object]], device: torch.device, batch_size: int) -> dict[str, float]:
    dataset = MultiOmicsDataset(processed["test"]["fusion"], processed["test"]["modalities"], processed["test"]["y"])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    y_true, proba = predict_torch(model, loader, device)
    return compute_metrics(y_true, proba)


def run_liquid_mismatch(
    seeds: list[int],
    rates: list[float],
    model_names: list[str],
    *,
    epochs: int,
    patience: int,
    batch_size: int,
    device: torch.device,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for task, path in TASK_FILES.items():
        print(f"Liquid mismatch task: {task}", flush=True)
        frame = pd.read_csv(path, low_memory=False)
        frame = frame[frame["task_label"].notna()].reset_index(drop=True)
        modality_cols = modality_columns(frame, LIQUID_CORE_MODALITIES)
        le = LabelEncoder()
        y_all = le.fit_transform(frame["task_label"].astype(str))
        num_classes = len(le.classes_)

        for seed in seeds:
            train_valid_idx, test_idx = train_test_split(
                np.arange(len(frame)),
                train_size=0.70,
                random_state=seed,
                stratify=y_all,
            )
            train_valid = frame.iloc[train_valid_idx].reset_index(drop=True)
            test_base = frame.iloc[test_idx].reset_index(drop=True)
            y_train_valid = le.transform(train_valid["task_label"].astype(str))
            inner_train_idx, valid_idx = train_test_split(
                np.arange(len(train_valid)),
                train_size=0.80,
                random_state=seed + 991,
                stratify=y_train_valid,
            )
            train_base = train_valid.iloc[inner_train_idx].reset_index(drop=True)
            valid_base = train_valid.iloc[valid_idx].reset_index(drop=True)

            for rate in rates:
                for scenario in ["train_misaligned", "test_misaligned"]:
                    rng = np.random.default_rng(seed * 1000 + int(rate * 1000) + (11 if scenario == "train_misaligned" else 29))
                    if scenario == "train_misaligned":
                        train_use = misalign_modalities(train_base, modality_cols, MISMATCH_MODALITIES, rate, rng)
                        valid_use = valid_base
                        test_use = test_base
                    else:
                        train_use = train_base
                        valid_use = valid_base
                        test_use = misalign_modalities(test_base, modality_cols, MISMATCH_MODALITIES, rate, rng)
                    processed = build_processed(train_use, valid_use, test_use, modality_cols, le)
                    input_dims = [arr.shape[1] for arr in processed["train"]["modalities"]]
                    loaders = make_loaders(processed, batch_size=batch_size)
                    for model_name in model_names:
                        print(f"  seed={seed} rate={rate} scenario={scenario} model={model_name}", flush=True)
                        stable_offset = sum(ord(ch) for ch in f"{task}|{model_name}|{scenario}") % 1000
                        set_seed(seed + int(rate * 1000) + stable_offset)
                        model = make_liquid_model(model_name, input_dims, num_classes)
                        result = train_torch_model(
                            model,
                            loaders,
                            processed["train"]["y"],
                            num_classes,
                            device=device,
                            lr=1e-3,
                            weight_decay=1e-4,
                            max_epochs=epochs,
                            patience=patience,
                        )
                        metrics = evaluate_torch_model(model, processed, device, batch_size)
                        rows.append(
                            {
                                "task": task,
                                "seed": seed,
                                "model": model_name,
                                "scenario": scenario,
                                "mismatch_rate": rate,
                                "feature_set": "multiomics_core",
                                "best_epoch": result["best_epoch"],
                                "best_validation_score": result["best_validation_score"],
                                **metrics,
                            }
                        )
    return pd.DataFrame(rows)


def summarize_liquid_mismatch(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        raw.groupby(["task", "model", "scenario", "feature_set", "mismatch_rate"], as_index=False)
        .agg(
            runs=("seed", "nunique"),
            n_mean=("n", "mean"),
            f1_macro_mean=("f1_macro", "mean"),
            f1_macro_sd=("f1_macro", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            roc_auc_ovr_macro_mean=("roc_auc_ovr_macro", "mean"),
        )
        .sort_values(["task", "model", "scenario", "mismatch_rate"])
    )
    aligned = summary[
        summary["scenario"].eq("train_misaligned")
        & summary["feature_set"].eq("multiomics_core")
        & summary["mismatch_rate"].eq(0.0)
    ]
    full = summary[
        summary["scenario"].eq("train_misaligned")
        & summary["feature_set"].eq("multiomics_core")
        & summary["mismatch_rate"].eq(1.0)
    ]
    degradation = aligned.merge(
        full,
        on=["task", "model", "scenario", "feature_set"],
        suffixes=("_aligned", "_full_mismatch"),
    )
    degradation["f1_drop"] = degradation["f1_macro_mean_aligned"] - degradation["f1_macro_mean_full_mismatch"]
    degradation = degradation[
        ["task", "model", "f1_macro_mean_aligned", "f1_macro_mean_full_mismatch", "f1_drop"]
    ].sort_values("f1_drop", ascending=False)
    return summary, degradation


def medical_interpretation_table() -> pd.DataFrame:
    rows = [
        {
            "task": "UCEC_molecular_subtype",
            "medical_endpoint": "Endometrial cancer molecular subtype: CN-high, CN-low, MSI, POLE",
            "dominant_biology": "Copy-number burden, mismatch repair deficiency, polymerase-epsilon proofreading mutations",
            "why_multiomics_relevant": "Subtype labels are explicitly multi-mechanistic; CNA, mutation, methylation, and expression can provide complementary signals.",
            "model_interpretation": "Liquid/CfC was near-best, consistent with potential benefit from cross-modality integration.",
        },
        {
            "task": "COADREAD_molecular_subtype",
            "medical_endpoint": "Colorectal cancer molecular subtype: CIN, MSI, genome-stable",
            "dominant_biology": "Chromosomal instability, microsatellite instability, and epigenetic/immune-associated programs",
            "why_multiomics_relevant": "The endpoint combines genome instability and expression/methylation programs; modality mismatch strongly degraded performance.",
            "model_interpretation": "small-Liquid/CfC was near-best, but ExtraTrees remained the top mean-F1 model.",
        },
        {
            "task": "HNSC_HPV_status",
            "medical_endpoint": "Head and neck squamous cancer HPV-positive versus HPV-negative status",
            "dominant_biology": "Viral oncogenesis with strong transcriptional and cell-cycle signatures",
            "why_multiomics_relevant": "The signal may already be high in expression space, so complex cross-modality modeling is not always necessary.",
            "model_interpretation": "Elastic-net logistic regression was strongest; Liquid/CfC was weaker, suggesting a simpler high-signal boundary.",
        },
        {
            "task": "KIRC_grade_binary",
            "medical_endpoint": "Clear-cell renal cancer low versus high histologic grade",
            "dominant_biology": "Tumor aggressiveness, hypoxia/angiogenesis, chromatin remodeling, and VHL-related biology",
            "why_multiomics_relevant": "Grade is partly histopathologic and may not be fully captured by bulk molecular features alone.",
            "model_interpretation": "Random Forest was strongest; pathology imaging may be a more natural complementary modality.",
        },
        {
            "task": "PRAD_pathologic_T_stage",
            "medical_endpoint": "Prostate cancer pathologic T2 versus T3/T4 stage",
            "dominant_biology": "Local invasion and anatomic extension rather than a purely molecular phenotype",
            "why_multiomics_relevant": "Molecular signal is expected to be modest; imaging/pathology or clinical variables may improve endpoint relevance.",
            "model_interpretation": "Liquid/CfC was near-best but all models were close, consistent with limited molecular separability.",
        },
    ]
    return pd.DataFrame(rows)


def make_liquid_mismatch_figure(summary: pd.DataFrame) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    plot_df = (
        summary[summary["scenario"].eq("train_misaligned")]
        .groupby(["model", "mismatch_rate"], as_index=False)["f1_macro_mean"]
        .mean()
    )
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    labels = {
        "liquid_cfc_modality_sequence": "Liquid/CfC",
        "small_liquid_cfc_modality_sequence": "small-Liquid/CfC",
    }
    for model, group in plot_df.groupby("model"):
        ax.plot(group["mismatch_rate"], group["f1_macro_mean"], marker="o", linewidth=2, label=labels.get(model, model))
    ax.set_xlabel("Fraction of non-mRNA omics rows permuted in training data")
    ax.set_ylabel("Macro F1 averaged over five TCGA tasks")
    ax.set_title("Liquid-family sensitivity to simulated multi-omics sample mismatch")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "second_paper_liquid_mismatch_degradation.png", dpi=240)
    plt.close(fig)


def write_report(audit: pd.DataFrame, liquid_summary: pd.DataFrame, liquid_degradation: pd.DataFrame, medical: pd.DataFrame) -> None:
    audit_display = audit.copy()
    audit_display["task"] = audit_display["task"].map(TASK_LABELS)
    audit_display = audit_display[
        [
            "task",
            "labelled_rows",
            "unique_sample_ids",
            "duplicate_sample_rows",
            "unique_patient_ids",
            "patients_with_multiple_labelled_samples",
            "complete_mismatch_core_fraction",
            "rppa_sample_observed_fraction",
        ]
    ]
    degradation_display = liquid_degradation.copy()
    degradation_display["task"] = degradation_display["task"].map(TASK_LABELS)
    degradation_display["model"] = degradation_display["model"].map(
        {
            "liquid_cfc_modality_sequence": "Liquid/CfC",
            "small_liquid_cfc_modality_sequence": "small-Liquid/CfC",
        }
    )
    medical_display = medical.copy()
    medical_display["task"] = medical_display["task"].map(TASK_LABELS)

    lines = [
        "# Alignment, Liquid Mismatch, and Medical Interpretation Strengthening Report",
        "",
        "## Real sample-alignment audit",
        "",
        markdown_table(audit_display),
        "",
        "## Liquid-family train-mismatch degradation",
        "",
        markdown_table(degradation_display),
        "",
        "## Medical interpretation table",
        "",
        markdown_table(medical_display),
        "",
        "## Generated files",
        "",
        "- `work/data/alignment_domain_shift_benchmark/processed/sample_alignment_audit.csv`",
        "- `work/data/alignment_domain_shift_benchmark/processed/liquid_mismatch_raw_results.csv`",
        "- `work/data/alignment_domain_shift_benchmark/processed/liquid_mismatch_summary.csv`",
        "- `work/data/alignment_domain_shift_benchmark/processed/liquid_mismatch_degradation.csv`",
        "- `work/data/alignment_domain_shift_benchmark/processed/medical_interpretation_table.csv`",
        "- `outputs/second_paper_liquid_mismatch_degradation.png`",
    ]
    (REPORT_DIR / "second_paper_alignment_liquid_medical_strengthening_report_v1.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--rates", default="0,0.1,0.25,0.5,0.75,1.0")
    parser.add_argument("--models", default="liquid_cfc_modality_sequence,small_liquid_cfc_modality_sequence")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-liquid", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    audit = sample_alignment_audit()
    audit.to_csv(OUT_DIR / "sample_alignment_audit.csv", index=False)
    medical = medical_interpretation_table()
    medical.to_csv(OUT_DIR / "medical_interpretation_table.csv", index=False)

    raw_path = OUT_DIR / "liquid_mismatch_raw_results.csv"
    if args.skip_liquid and raw_path.exists():
        raw = pd.read_csv(raw_path)
    else:
        raw = run_liquid_mismatch(
            parse_list(args.seeds, int),
            parse_list(args.rates, float),
            parse_list(args.models, str),
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            device=torch.device(args.device),
        )
        raw.to_csv(raw_path, index=False)

    summary, degradation = summarize_liquid_mismatch(raw)
    summary.to_csv(OUT_DIR / "liquid_mismatch_summary.csv", index=False)
    degradation.to_csv(OUT_DIR / "liquid_mismatch_degradation.csv", index=False)
    make_liquid_mismatch_figure(summary)
    write_report(audit, summary, degradation, medical)
    print(REPORT_DIR / "second_paper_alignment_liquid_medical_strengthening_report_v1.md")


if __name__ == "__main__":
    main()

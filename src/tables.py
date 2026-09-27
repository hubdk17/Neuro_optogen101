"""
tables.py
=========
Generation and formatting of publication-ready scientific tables (Tables 1 - 7).

Table 1: Dataset and session characteristics
Table 2: Unit feature distributions by reference class
Table 3: Rule-based threshold sensitivity (27 parameter combinations)
Table 4: ML performance under random unit split (data leakage baseline)
Table 5: ML performance under session-held-out validation (primary result)
Table 6: Feature ablation study (Models A through F)
Table 7: Specimen-held-out validation
"""

from typing import Dict, Any, List, Optional
import logging
from pathlib import Path
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def generate_table1_dataset_characteristics(
    inventory_df: pd.DataFrame,
    output_path: str = "results/tables/Table_1_dataset_characteristics.csv"
) -> pd.DataFrame:
    """
    Table 1: Cohort-level summary of dataset and recording sessions.
    Breakdown by transgenic Cre driver line (Pvalb, Sst, Vip, Wild-type).
    """
    records = []
    
    # Group by genotype
    for geno, grp in inventory_df.groupby("genotype"):
        n_sessions = len(grp)
        n_probes = grp["probe_count"].sum()
        total_units = grp["unit_count"].dropna().sum()
        mean_units = grp["unit_count"].dropna().mean()
        is_opto = grp["is_eligible_opto"].iloc[0]
        
        # Summarize unique brain areas
        all_areas = set()
        for areas_str in grp["brain_areas"].dropna():
            for a in str(areas_str).split(";"):
                clean_a = a.strip()
                if clean_a and clean_a != "nan":
                    all_areas.add(clean_a)
                    
        records.append({
            "Transgenic Line / Genotype": geno,
            "Optogenetic Stimulation": "Yes (ChR2/Ai32)" if is_opto else "No (Wild-type)",
            "Sessions Count": n_sessions,
            "Total Probes": int(n_probes),
            "Total Units": int(total_units),
            "Mean Units / Session": np.round(mean_units, 1) if not np.isnan(mean_units) else "Pending",
            "Unique Brain Structures Recorded": len(all_areas),
            "Key Structures": ", ".join(sorted(list(all_areas))[:8]) + ("..." if len(all_areas) > 8 else "")
        })
        
    df1 = pd.DataFrame(records)
    # Sort with opto lines on top
    df1.sort_values(by=["Optogenetic Stimulation", "Sessions Count"], ascending=[False, False], inplace=True)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df1.to_csv(output_path, index=False)
    logger.info("Saved Table 1 to %s", output_path)
    return df1


def generate_table2_feature_distributions_by_class(
    labeled_features_df: pd.DataFrame,
    output_path: str = "results/tables/Table_2_feature_distributions.csv"
) -> pd.DataFrame:
    """
    Table 2: Physiological unit feature distributions stratified by operational reference class.
    Reports median, IQR, mean, and standard deviation.
    """
    features_to_report = [
        ("baseline_rate", "Baseline Firing Rate (Hz)"),
        ("evoked_rate", "Evoked Firing Rate (Hz)"),
        ("modulation_ratio", "Modulation Ratio (Evoked / (Base + 1))"),
        ("median_latency_ms", "First-Spike Latency (ms)"),
        ("latency_sd_ms", "Latency Jitter SD (ms)"),
        ("latency_iqr_ms", "Latency IQR (ms)"),
        ("trial_reliability", "Trial Reliability (P(>=1 spk))"),
        ("sham_reliability", "Pre-onset Sham Control (FPR)"),
        ("p_value", "Permutation Test p-value"),
        ("effect_size", "Cohen's d Effect Size")
    ]
    
    classes = [
        "putatively directly optotagged",
        "light-responsive / indirect or uncertain",
        "not light responsive"
    ]
    
    rows = []
    for feat_col, feat_name in features_to_report:
        row = {"Physiological Feature": feat_name}
        for cls in classes:
            sub = labeled_features_df[labeled_features_df["reference_class"] == cls][feat_col].dropna()
            if len(sub) > 0:
                med = np.median(sub)
                q25, q75 = np.percentile(sub, [25, 75])
                row[f"{cls} [Median (IQR)]"] = f"{med:.2f} ({q25:.2f} - {q75:.2f})"
            else:
                row[f"{cls} [Median (IQR)]"] = "N/A"
        rows.append(row)
        
    df2 = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df2.to_csv(output_path, index=False)
    logger.info("Saved Table 2 to %s", output_path)
    return df2


def generate_table3_threshold_sensitivity(
    sensitivity_df: pd.DataFrame,
    output_path: str = "results/tables/Table_3_threshold_sensitivity.csv"
) -> pd.DataFrame:
    """
    Table 3: Threshold sensitivity grid across all 27 combinations.
    """
    t3 = sensitivity_df.copy()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    t3.to_csv(output_path, index=False)
    logger.info("Saved Table 3 to %s", output_path)
    return t3


def format_ml_results_table(
    results_list: List[Dict[str, Any]],
    output_path: str,
    table_name: str
) -> pd.DataFrame:
    """
    Format machine learning metrics into structured tabular format.
    """
    rows = []
    for res in results_list:
        m = res.get("overall_metrics", {})
        rows.append({
            "Evaluation Strategy": res.get("cv_mode", "N/A"),
            "Feature Set": res.get("feature_set_name", "N/A"),
            "Model Classifier": res.get("model_name", "N/A"),
            "Balanced Accuracy": m.get("balanced_accuracy", np.nan),
            "Macro F1": m.get("macro_f1", np.nan),
            "Macro Precision": m.get("macro_precision", np.nan),
            "Macro Recall": m.get("macro_recall", np.nan),
            "AUROC (OVR)": m.get("auroc", np.nan),
            "AUPRC": m.get("auprc", np.nan),
            "Brier Score": m.get("brier_score", np.nan),
            "Expected Calibration Error (ECE)": m.get("ece", np.nan)
        })
        
    df = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved %s to %s", table_name, output_path)
    return df

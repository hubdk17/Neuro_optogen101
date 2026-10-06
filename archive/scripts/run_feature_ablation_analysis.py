"""
run_feature_ablation_analysis.py
================================
Compiles the Leave-One-Feature-Family-Out (LOFFO) ablation results across:
A. TEMPORAL (latency, jitter SD, IQR, CV)
B. RELIABILITY (trial reliability, sham FPR, Fano factor)
C. FIRING (baseline rate, evoked rate, modulation ratio)
D. STATISTICAL (p-value, effect size)
E. INTENSITY (optical intensity slope)
F. DYNAMICS (10-Hz train adaptation index)

Outputs:
- results/feature_ablation.csv
- figures/fig7_feature_ablation.png (and .pdf)
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure UTF-8
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

def run_feature_ablation():
    src_csv = "results/tables/Table_audit_loffo_ablation.csv"
    if not os.path.exists(src_csv):
        raise FileNotFoundError(f"Missing LOFFO table: {src_csv}")
        
    df = pd.read_csv(src_csv)
    
    # Calculate delta relative to Full Model for Random Forest
    rf_base = df[(df["Ablation Experiment"].str.startswith("Full_Model")) & (df["Model Classifier"] == "random_forest")].iloc[0]
    base_bal_acc = rf_base["Balanced Accuracy"]
    base_f1 = rf_base["Macro F1"]
    base_auprc = rf_base["AUPRC"]
    
    deltas = []
    for _, row in df.iterrows():
        d_bal = row["Balanced Accuracy"] - base_bal_acc
        d_f1 = row["Macro F1"] - base_f1
        d_auprc = row["AUPRC"] - base_auprc
        deltas.append({
            "delta_balanced_accuracy": round(d_bal, 4),
            "delta_macro_f1": round(d_f1, 4),
            "delta_auprc": round(d_auprc, 4)
        })
        
    df_delta = pd.concat([df, pd.DataFrame(deltas)], axis=1)
    
    out_csv = "results/feature_ablation.csv"
    df_delta.to_csv(out_csv, index=False)
    print(f"Saved feature ablation results to {out_csv}")
    
    # Generate Figure 7
    plot_feature_ablation_figure(df_delta)

def plot_feature_ablation_figure(df: pd.DataFrame):
    fig_dir = Path("figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    plt.style.use("seaborn-v0_8-paper" if "seaborn-v0_8-paper" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=300)
    
    rf_df = df[df["Model Classifier"] == "random_forest"].copy()
    rf_df["Condition"] = rf_df["Ablation Experiment"].str.replace("Minus_", "- ").str.replace("Full_Model (All 6 Families)", "Full Model (14 feats)")
    
    # Panel A: Balanced Accuracy across LOFFO conditions
    palette = ["#1f77b4" if d >= 0 else "#d62728" for d in rf_df["delta_balanced_accuracy"]]
    sns.barplot(data=rf_df, x="Condition", y="Balanced Accuracy", palette=palette, ax=axes[0, 0])
    axes[0, 0].axhline(0.7789, color="black", linestyle="--", alpha=0.7, label="Full Model Baseline (0.7789)")
    axes[0, 0].set_title("A. Impact of Withholding Feature Families on Balanced Accuracy", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Ablation Condition", fontsize=11)
    axes[0, 0].set_ylabel("Held-Out Balanced Accuracy", fontsize=11)
    axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=35, ha="right", fontsize=9)
    axes[0, 0].set_ylim(0.4, 0.9)
    axes[0, 0].legend(loc="lower left")
    axes[0, 0].grid(True, linestyle=":", alpha=0.6)
    
    # Panel B: Delta Balanced Accuracy (Sensitivity Ranking)
    rf_ablated = rf_df[rf_df["Excluded Family"] != "None"].sort_values(by="delta_balanced_accuracy")
    colors = ["#d62728" if x < 0 else "#2ca02c" for x in rf_ablated["delta_balanced_accuracy"]]
    
    axes[0, 1].barh(rf_ablated["Excluded Family"], rf_ablated["delta_balanced_accuracy"] * 100, color=colors)
    axes[0, 1].axvline(0, color="black", lw=1)
    axes[0, 1].set_title("B. Feature Family Criticality Index (Δ Balanced Accuracy %)", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Change in Balanced Accuracy (% points)", fontsize=11)
    axes[0, 1].set_ylabel("Excluded Physiological Family", fontsize=11)
    axes[0, 1].grid(True, linestyle=":", alpha=0.6)
    
    # Add annotations
    axes[0, 1].text(-21.0, 0, "Critical Anchor (-22.0%)\np-value & Cohen's d", color="#d62728", va="center", fontweight="bold", fontsize=9)
    axes[0, 1].text(0.5, 4.8, "Noise Distractor (+4.2%)\nLatency jitter", color="#2ca02c", va="center", fontweight="bold", fontsize=9)
    
    # Panel C: Model Comparison across LOFFO (Random Forest vs XGBoost)
    sns.barplot(data=df, x="Excluded Family", y="Macro F1", hue="Model Classifier", ax=axes[1, 0], palette="Set1")
    axes[1, 0].set_title("C. Architecture Robustness across Feature Families (RF vs XGBoost)", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("Excluded Feature Family", fontsize=11)
    axes[1, 0].set_ylabel("Held-Out Macro F1", fontsize=11)
    axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=35, ha="right", fontsize=9)
    axes[1, 0].set_ylim(0.4, 0.9)
    axes[1, 0].legend(loc="lower right")
    axes[1, 0].grid(True, linestyle=":", alpha=0.6)
    
    # Panel D: AUPRC vs Calibration Error (ECE) across Ablations
    sns.scatterplot(
        data=rf_df, x="AUPRC", y="ECE", hue="Excluded Family", s=120, ax=axes[1, 1], palette="tab10"
    )
    axes[1, 1].set_title("D. Precision-Recall Area vs Expected Calibration Error (ECE)", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("Held-Out AUPRC", fontsize=11)
    axes[1, 1].set_ylabel("Expected Calibration Error (ECE - Lower = Better)", fontsize=11)
    axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Ablated Family")
    axes[1, 1].grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    png_path = fig_dir / "fig7_feature_ablation.png"
    pdf_path = fig_dir / "fig7_feature_ablation.pdf"
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()
    print(f"Saved Figure 7 to {png_path} and {pdf_path}")

if __name__ == "__main__":
    run_feature_ablation()

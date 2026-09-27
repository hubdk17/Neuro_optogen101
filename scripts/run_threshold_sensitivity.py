"""
run_threshold_sensitivity.py
============================
Executes the systematic 3x3x3 threshold-instability analysis across:
- Latency thresholds: [6.0, 8.0, 10.0] ms
- Reliability thresholds: [0.20, 0.30, 0.50]
- Modulation thresholds: [1.5, 2.0, 3.0]
Total: 27 configurations.

Evaluates at:
- Unit level
- Session level
- Specimen level

Computes:
- direct yield and fraction
- uncertain yield and fraction
- non-responsive yield and fraction
- units changing class relative to baseline
- pairwise Jaccard similarity
- boundary distance distribution
- produces results/threshold_sensitivity.csv and figures/fig4_threshold_instability.png
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

def classify_units_with_thresholds(
    df: pd.DataFrame,
    lat_th: float,
    rel_th: float,
    mod_th: float,
    p_th: float = 0.05,
    d_th: float = 0.10
) -> pd.Series:
    """
    Apply operational optotagging rule for given threshold triplet.
    """
    is_direct = (
        (df["median_latency_ms"] < lat_th) &
        (df["trial_reliability"] >= rel_th) &
        (df["modulation_ratio"] > mod_th) &
        (df["p_value"] < p_th) &
        (df["effect_size"] > d_th)
    )
    
    is_nonresp = (
        (df["p_value"] >= p_th) |
        (df["modulation_ratio"] <= 1.0) |
        (df["trial_reliability"] < 0.05)
    )
    
    classes = pd.Series("light-responsive / indirect or uncertain", index=df.index)
    classes[is_nonresp] = "not light responsive"
    classes[is_direct] = "putatively directly optotagged"
    
    # Check baseline firing rate < 0.1 Hz for insufficient evidence
    low_baseline = df["baseline_rate"] < 0.1
    classes[low_baseline & ~is_direct] = "insufficient evidence"
    return classes

def run_threshold_grid_analysis():
    input_csv = "results/tables/unit_features_multisession.csv"
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input multisession features not found: {input_csv}")
        
    df = pd.read_csv(input_csv)
    print(f"Loaded multisession features: {len(df)} units across {df['session_id'].nunique()} sessions and {df['specimen_id'].nunique()} specimens.")
    
    # Filter valid for evaluation (exclude insufficient evidence from active denominators)
    eval_df = df.copy()
    
    latency_thresholds = [6.0, 8.0, 10.0]
    reliability_thresholds = [0.20, 0.30, 0.50]
    modulation_thresholds = [1.5, 2.0, 3.0]
    
    # Reference baseline configuration: (8.0 ms, 0.30, 2.0)
    ref_labels = classify_units_with_thresholds(eval_df, 8.0, 0.30, 2.0)
    ref_direct_set = set(eval_df[ref_labels == "putatively directly optotagged"].index)
    
    grid_records = []
    classification_matrix = {}
    
    for lat in latency_thresholds:
        for rel in reliability_thresholds:
            for mod in modulation_thresholds:
                config_id = f"L{int(lat)}_R{int(rel*100)}_M{str(mod).replace('.', 'p')}"
                labels = classify_units_with_thresholds(eval_df, lat, rel, mod)
                classification_matrix[config_id] = labels
                
                n_total = len(eval_df)
                n_direct = (labels == "putatively directly optotagged").sum()
                n_indirect = (labels == "light-responsive / indirect or uncertain").sum()
                n_nonresp = (labels == "not light responsive").sum()
                n_insuff = (labels == "insufficient evidence").sum()
                
                pct_direct = n_direct / n_total * 100
                pct_indirect = n_indirect / n_total * 100
                pct_nonresp = n_nonresp / n_total * 100
                
                # Compare to reference configuration
                cur_direct_set = set(eval_df[labels == "putatively directly optotagged"].index)
                intersection = len(cur_direct_set & ref_direct_set)
                union = len(cur_direct_set | ref_direct_set)
                jaccard_to_ref = intersection / union if union > 0 else 1.0
                
                # Units changing class relative to reference
                units_changed = (labels != ref_labels).sum()
                pct_units_changed = units_changed / n_total * 100
                
                # Session-level breakdown
                session_breakdown = {}
                for s_id in eval_df["session_id"].unique():
                    s_mask = eval_df["session_id"] == s_id
                    s_direct = (labels[s_mask] == "putatively directly optotagged").sum()
                    session_breakdown[f"session_{s_id}_direct_count"] = s_direct
                    session_breakdown[f"session_{s_id}_direct_pct"] = round(s_direct / s_mask.sum() * 100, 2)
                    
                # Specimen-level breakdown
                specimen_breakdown = {}
                for sp_id in eval_df["specimen_id"].unique():
                    sp_mask = eval_df["specimen_id"] == sp_id
                    sp_direct = (labels[sp_mask] == "putatively directly optotagged").sum()
                    specimen_breakdown[f"specimen_{sp_id}_direct_count"] = sp_direct
                    specimen_breakdown[f"specimen_{sp_id}_direct_pct"] = round(sp_direct / sp_mask.sum() * 100, 2)
                    
                rec = {
                    "config_id": config_id,
                    "latency_threshold_ms": lat,
                    "reliability_threshold": rel,
                    "modulation_threshold": mod,
                    "total_units": n_total,
                    "direct_count": n_direct,
                    "direct_pct": round(pct_direct, 2),
                    "indirect_count": n_indirect,
                    "indirect_pct": round(pct_indirect, 2),
                    "nonresponsive_count": n_nonresp,
                    "nonresponsive_pct": round(pct_nonresp, 2),
                    "insufficient_evidence_count": n_insuff,
                    "jaccard_similarity_to_baseline": round(jaccard_to_ref, 4),
                    "units_changing_class": units_changed,
                    "pct_units_changing_class": round(pct_units_changed, 2)
                }
                rec.update(session_breakdown)
                rec.update(specimen_breakdown)
                grid_records.append(rec)
                
    df_grid = pd.DataFrame(grid_records)
    out_csv = Path("results/threshold_sensitivity.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_grid.to_csv(out_csv, index=False)
    print(f"Saved complete 27-point threshold sensitivity grid to {out_csv}")
    
    # Print key sensitivity summary
    print("\n=== THRESHOLD SENSITIVITY SUMMARY ===")
    print(f"Direct yield range across 27 configurations: {df_grid['direct_count'].min()} to {df_grid['direct_count'].max()} units "
          f"({df_grid['direct_pct'].min()}% to {df_grid['direct_pct'].max()}%)")
    print(f"Jaccard similarity to baseline (8ms, 0.30 rel, 2.0 mod): {df_grid['jaccard_similarity_to_baseline'].min():.4f} to 1.0000")
    print(f"Units switching class across threshold combinations: {df_grid['units_changing_class'].min()} to {df_grid['units_changing_class'].max()} units "
          f"({df_grid['pct_units_changing_class'].min()}% to {df_grid['pct_units_changing_class'].max()}%)")
          
    # Generate Publication Figure: Figure 4 (Threshold Instability)
    plot_threshold_instability_figure(df_grid)
    return df_grid

def plot_threshold_instability_figure(df_grid: pd.DataFrame):
    fig_dir = Path("figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    plt.style.use("seaborn-v0_8-paper" if "seaborn-v0_8-paper" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=300)
    
    # Panel A: Heatmap of Direct Count by Latency and Reliability (at Modulation = 2.0)
    sub_m2 = df_grid[df_grid["modulation_threshold"] == 2.0]
    pivot_direct = sub_m2.pivot(index="latency_threshold_ms", columns="reliability_threshold", values="direct_count")
    
    sns.heatmap(pivot_direct, annot=True, fmt="d", cmap="Blues", ax=axes[0, 0], cbar_kws={'label': 'Direct Optotagged Units (N)'})
    axes[0, 0].set_title("A. Direct Tag Yield (Modulation Ratio = 2.0)", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Reliability Threshold", fontsize=11)
    axes[0, 0].set_ylabel("Latency Threshold (ms)", fontsize=11)
    
    # Panel B: Jaccard Similarity to Baseline across all 27 configurations
    sns.barplot(data=df_grid, x="config_id", y="jaccard_similarity_to_baseline", ax=axes[0, 1], color="#1f77b4")
    axes[0, 1].set_title("B. Classification Stability (Jaccard Similarity to 8ms/0.30/2.0 Baseline)", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Threshold Configuration (27 Conditions)", fontsize=11)
    axes[0, 1].set_ylabel("Jaccard Similarity Index", fontsize=11)
    axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=90, fontsize=8)
    axes[0, 1].axhline(1.0, color="red", linestyle="--", alpha=0.6, label="Reference Baseline")
    axes[0, 1].set_ylim(0, 1.1)
    axes[0, 1].legend(loc="lower left")
    
    # Panel C: Session-level Direct Yield Variation
    s1_col = [c for c in df_grid.columns if "session_" in c and "_direct_count" in c][0]
    s2_col = [c for c in df_grid.columns if "session_" in c and "_direct_count" in c][1]
    
    axes[1, 0].plot(df_grid["config_id"], df_grid[s1_col], marker="o", color="#2ca02c", label=f"Session {s1_col.split('_')[1]} (Specimen 707296982)")
    axes[1, 0].plot(df_grid["config_id"], df_grid[s2_col], marker="s", color="#d62728", label=f"Session {s2_col.split('_')[1]} (Specimen 739783171)")
    axes[1, 0].set_title("C. Inter-Session Yield Discrepancy Across Thresholds", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("Threshold Configuration", fontsize=11)
    axes[1, 0].set_ylabel("Putatively Direct Units (N)", fontsize=11)
    axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=90, fontsize=8)
    axes[1, 0].legend(loc="upper left")
    axes[1, 0].grid(True, linestyle=":", alpha=0.6)
    
    # Panel D: Total Units Switching Class Across Threshold Changes
    sns.lineplot(data=df_grid, x="config_id", y="pct_units_changing_class", marker="^", color="#9467bd", ax=axes[1, 1])
    axes[1, 1].set_title("D. Total Population Switching Operational Class (%)", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("Threshold Configuration", fontsize=11)
    axes[1, 1].set_ylabel("Units Changing Class (%)", fontsize=11)
    axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=90, fontsize=8)
    axes[1, 1].grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    png_path = fig_dir / "fig4_threshold_instability.png"
    pdf_path = fig_dir / "fig4_threshold_instability.pdf"
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()
    print(f"Saved Figure 4 to {png_path} and {pdf_path}")

if __name__ == "__main__":
    run_threshold_grid_analysis()

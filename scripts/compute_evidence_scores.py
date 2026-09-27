"""
compute_evidence_scores.py
==========================
Computes the continuous, interpretable Optogenetic Response Evidence Score (E_i)
and comprehensive uncertainty metrics (U_i) for Neuropixels units.

Mathematical Formulation:
E_i = g_A(A_i) * [ w_S * s_S(S_i) + w_R * s_R(R_i) + w_M * s_M(M_i) + w_L * s_L(L_i) + w_J * s_J(J_i) ]

Outputs:
- results/evidence_scores.parquet
- results/evidence_scores.csv
- results/uncertainty_analysis.csv
- figures/fig5_evidence_score.png
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

def compute_evidence_and_uncertainty(df: pd.DataFrame) -> pd.DataFrame:
    res = df.copy()
    
    # 1. Sub-scores
    # S: Statistical significance (-log10(p_value))
    p_vals = np.clip(res["p_value"].fillna(1.0).values, 1e-15, 1.0)
    neg_log_p = -np.log10(p_vals)
    s_S = 1.0 / (1.0 + np.exp(-(neg_log_p - 2.0) / 1.0))
    
    # R: Trial reliability [0, 1]
    rel = np.clip(res["trial_reliability"].fillna(0.0).values, 0.0, 1.0)
    s_R = rel
    
    # M: Modulation ratio
    mod = np.clip(res["modulation_ratio"].fillna(0.0).values, 0.0, 100.0)
    # Log2 modulation for values > 1, 0 otherwise
    log2_mod = np.where(mod > 1.0, np.log2(np.maximum(mod, 1.0)), 0.0)
    s_M = np.tanh(log2_mod / 2.0)
    
    # L: Latency (smooth decay centered at 8 ms)
    lat = res["median_latency_ms"].fillna(25.0).values
    s_L = 1.0 / (1.0 + np.exp((lat - 8.0) / 2.0))
    
    # J: Temporal jitter (low jitter = high score)
    jitter = res["latency_sd_ms"].fillna(10.0).values
    s_J = np.exp(-jitter / 2.5)
    
    # Artifact gating: if artifact_fraction > 0.05, penalize
    art_frac = res["artifact_fraction"].fillna(0.0).values if "artifact_fraction" in res.columns else np.zeros(len(res))
    g_A = 1.0 - np.clip(art_frac * 2.0, 0.0, 1.0)
    
    # Weights based on LOFFO ablation findings
    w_S = 0.30
    w_R = 0.25
    w_M = 0.20
    w_L = 0.15
    w_J = 0.10
    
    raw_evidence = (w_S * s_S + w_R * s_R + w_M * s_M + w_L * s_L + w_J * s_J)
    evidence_score = g_A * raw_evidence
    res["evidence_score"] = np.round(evidence_score, 4)
    res["subscore_statistical"] = np.round(s_S, 4)
    res["subscore_reliability"] = np.round(s_R, 4)
    res["subscore_modulation"] = np.round(s_M, 4)
    res["subscore_latency"] = np.round(s_L, 4)
    res["subscore_jitter"] = np.round(s_J, 4)
    res["artifact_gating"] = np.round(g_A, 4)
    
    # 2. Uncertainty Metrics
    # A. Trial sampling variance: R * (1 - R) / 45
    trial_var = (rel * (1.0 - rel)) / 45.0
    res["trial_sampling_variance"] = np.round(trial_var, 5)
    
    # B. Latency uncertainty (CV of latency)
    res["latency_uncertainty"] = np.round(np.clip(jitter / np.maximum(lat, 1.0), 0.0, 2.0), 4)
    
    # C. Threshold boundary proximity (distance to L=8, R=0.30, M=2.0)
    d_lat = (lat - 8.0) / 4.0
    d_rel = (rel - 0.30) / 0.20
    d_mod = (log2_mod - 1.0) / 1.0
    boundary_dist = np.sqrt(d_lat**2 + d_rel**2 + d_mod**2)
    boundary_proximity = np.exp(-boundary_dist)  # 1.0 at boundary, decays far away
    res["boundary_proximity_uncertainty"] = np.round(boundary_proximity, 4)
    res["distance_to_decision_boundary"] = np.round(boundary_dist, 4)
    
    # D. Composite Uncertainty Index U_i in [0, 1]
    composite_uncertainty = 0.40 * boundary_proximity + 0.35 * res["latency_uncertainty"] + 0.25 * (trial_var / 0.0055)
    res["composite_uncertainty"] = np.round(np.clip(composite_uncertainty, 0.0, 1.0), 4)
    
    # 3. Stratification into Evidence Regimes
    regimes = []
    for _, row in res.iterrows():
        base_rate = row.get("baseline_rate", 0.0)
        e = row["evidence_score"]
        u = row["composite_uncertainty"]
        
        if base_rate < 0.1 and e < 0.30:
            regimes.append("insufficient evidence")
        elif e >= 0.65 and u < 0.50:
            regimes.append("high evidence (direct-like)")
        elif (0.35 <= e < 0.65) or (e >= 0.50 and u >= 0.50):
            regimes.append("intermediate evidence (uncertain / borderline)")
        else:
            regimes.append("low evidence (non-responsive)")
            
    res["evidence_regime"] = regimes
    return res

def plot_evidence_score_figure(res_df: pd.DataFrame):
    fig_dir = Path("figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    plt.style.use("seaborn-v0_8-paper" if "seaborn-v0_8-paper" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=300)
    
    # Panel A: Distribution of Continuous Evidence Scores by Reference Class
    sns.kdeplot(
        data=res_df[res_df["reference_class"].isin(["not light responsive", "light-responsive / indirect or uncertain", "putatively directly optotagged"])],
        x="evidence_score", hue="reference_class", common_norm=False, fill=True, alpha=0.35, ax=axes[0, 0],
        palette={"not light responsive": "#1f77b4", "light-responsive / indirect or uncertain": "#ff7f0e", "putatively directly optotagged": "#2ca02c"}
    )
    axes[0, 0].set_title("A. Graded Evidence Score Distribution Across Operational Classes", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Optogenetic Response Evidence Score E_i [0, 1]", fontsize=11)
    axes[0, 0].set_ylabel("Kernel Density", fontsize=11)
    axes[0, 0].set_xlim(-0.05, 1.05)
    
    # Panel B: Evidence Score vs Composite Uncertainty (Stratified Regimes)
    sns.scatterplot(
        data=res_df, x="evidence_score", y="composite_uncertainty", hue="evidence_regime",
        alpha=0.65, s=35, ax=axes[0, 1],
        palette={
            "high evidence (direct-like)": "#2ca02c",
            "intermediate evidence (uncertain / borderline)": "#ff7f0e",
            "low evidence (non-responsive)": "#1f77b4",
            "insufficient evidence": "#7f7f7f"
        }
    )
    axes[0, 1].axvline(0.35, color="grey", linestyle=":", alpha=0.6)
    axes[0, 1].axvline(0.65, color="grey", linestyle=":", alpha=0.6)
    axes[0, 1].set_title("B. Evidence vs Uncertainty Landscape (Borderline Identification)", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Optogenetic Response Evidence Score E_i", fontsize=11)
    axes[0, 1].set_ylabel("Composite Uncertainty Index U_i", fontsize=11)
    
    # Panel C: Sub-score Contributions in Direct vs Indirect vs Non-responsive
    subscore_cols = ["subscore_statistical", "subscore_reliability", "subscore_modulation", "subscore_latency", "subscore_jitter"]
    mean_subscores = res_df[res_df["reference_class"].isin(["not light responsive", "light-responsive / indirect or uncertain", "putatively directly optotagged"])].groupby("reference_class")[subscore_cols].mean().T
    
    mean_subscores.plot(kind="bar", ax=axes[1, 0], colormap="viridis", width=0.8)
    axes[1, 0].set_title("C. Multi-Dimensional Physiological Sub-Score Profiles", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("Physiological Evidence Dimension", fontsize=11)
    axes[1, 0].set_ylabel("Normalized Sub-Score [0, 1]", fontsize=11)
    axes[1, 0].set_xticklabels(["Statistical", "Reliability", "Modulation", "Latency", "Low Jitter"], rotation=45, ha="right", fontsize=10)
    axes[1, 0].legend(title="Operational Class", loc="upper left")
    axes[1, 0].grid(True, linestyle=":", alpha=0.6)
    
    # Panel D: Threshold Boundary Proximity vs Conventional Binary Classification Jitter
    sns.boxplot(data=res_df, x="reference_class", y="distance_to_decision_boundary", ax=axes[1, 1], palette="Set2")
    axes[1, 1].set_title("D. Distance to Conventional Decision Boundary", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("Operational Class", fontsize=11)
    axes[1, 1].set_ylabel("Normalized Euclidean Distance to Threshold Hyperplane", fontsize=11)
    axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=15, ha="right", fontsize=9)
    
    plt.tight_layout()
    png_path = fig_dir / "fig5_evidence_score.png"
    pdf_path = fig_dir / "fig5_evidence_score.pdf"
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()
    print(f"Saved Figure 5 to {png_path} and {pdf_path}")

def run_evidence_pipeline():
    in_csv = "results/tables/unit_features_multisession.csv"
    if not os.path.exists(in_csv):
        raise FileNotFoundError(f"Missing multisession features: {in_csv}")
        
    df = pd.read_csv(in_csv)
    print(f"Loaded {len(df)} units for continuous evidence scoring.")
    
    res_df = compute_evidence_and_uncertainty(df)
    
    # Save outputs
    pq_path = "results/evidence_scores.parquet"
    csv_path = "results/evidence_scores.csv"
    res_df.to_parquet(pq_path, index=False)
    res_df.to_csv(csv_path, index=False)
    print(f"Saved evidence scores to {pq_path} and {csv_path}")
    
    # Build uncertainty summary table
    unc_summary = res_df.groupby("evidence_regime").agg(
        unit_count=("unit_id", "count"),
        mean_evidence=("evidence_score", "mean"),
        std_evidence=("evidence_score", "std"),
        mean_uncertainty=("composite_uncertainty", "mean"),
        mean_reliability=("trial_reliability", "mean"),
        mean_latency=("median_latency_ms", "mean"),
        mean_boundary_dist=("distance_to_decision_boundary", "mean")
    ).reset_index()
    
    unc_path = "results/uncertainty_analysis.csv"
    unc_summary.to_csv(unc_path, index=False)
    print(f"Saved uncertainty analysis summary to {unc_path}")
    print("\nEvidence Regime Breakdown:")
    print(unc_summary.to_string())
    
    # Generate Figure 5
    plot_evidence_score_figure(res_df)

if __name__ == "__main__":
    run_evidence_pipeline()

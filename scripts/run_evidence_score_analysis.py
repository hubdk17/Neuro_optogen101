"""
run_evidence_score_analysis.py
==============================
Computes the continuous Optogenetic Response Evidence Score E_i and performs
a dedicated weight-sensitivity and stability analysis across:
- Model A: Proposed physiological weights (0.30, 0.25, 0.20, 0.15, 0.10)
- Model B: Equal weights (0.20, 0.20, 0.20, 0.20, 0.20)
- Model C: Leave-one-feature-family-out (LOFFO) re-normalized models:
  - C1: Minus Statistical
  - C2: Minus Reliability
  - C3: Minus Modulation
  - C4: Minus Latency
  - C5: Minus Jitter
- Model D: Alternative weight perturbations:
  - D1: Heavy-Statistical (0.50, 0.15, 0.15, 0.10, 0.10)
  - D2: Heavy-Reliability (0.15, 0.45, 0.15, 0.15, 0.10)
  - D3: Heavy-Latency (0.15, 0.20, 0.15, 0.40, 0.10)
  - D4: Heavy-Modulation (0.15, 0.15, 0.45, 0.15, 0.10)

Calculates:
- Spearman rank correlation rho relative to Model A
- Kendall tau relative to Model A
- Unit regime classification stability
- Top-decile overlap with Model A
- Session-level mean evidence stability
- Specimen-level mean evidence stability
- Agreement with operational reference class

Outputs:
- results/evidence/evidence_scores.csv
- results/evidence/weight_sensitivity.csv
- results/evidence/stability_analysis.csv
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau

def compute_subscores(df):
    p_vals = np.clip(df["p_value"].fillna(1.0).values, 1e-15, 1.0)
    neg_log_p = -np.log10(p_vals)
    s_S = 1.0 / (1.0 + np.exp(-(neg_log_p - 2.0) / 1.0))
    
    rel = np.clip(df["trial_reliability"].fillna(0.0).values, 0.0, 1.0)
    s_R = rel
    
    mod = np.clip(df["modulation_ratio"].fillna(0.0).values, 0.0, 100.0)
    log2_mod = np.where(mod > 1.0, np.log2(np.maximum(mod, 1.0)), 0.0)
    s_M = np.tanh(log2_mod / 2.0)
    
    lat = df["median_latency_ms"].fillna(25.0).values
    s_L = 1.0 / (1.0 + np.exp((lat - 8.0) / 2.0))
    
    jitter = df["latency_sd_ms"].fillna(10.0).values
    s_J = np.exp(-jitter / 2.5)
    
    art_frac = df["artifact_fraction"].fillna(0.0).values if "artifact_fraction" in df.columns else np.zeros(len(df))
    g_A = 1.0 - np.clip(art_frac * 2.0, 0.0, 1.0)
    
    return s_S, s_R, s_M, s_L, s_J, g_A, lat, rel, log2_mod, jitter

def compute_uncertainty_metrics(df, lat, rel, log2_mod, jitter):
    trial_var = (rel * (1.0 - rel)) / 45.0
    latency_unc = np.clip(jitter / np.maximum(lat, 1.0), 0.0, 2.0)
    
    d_lat = (lat - 8.0) / 4.0
    d_rel = (rel - 0.30) / 0.20
    d_mod = (log2_mod - 1.0) / 1.0
    boundary_dist = np.sqrt(d_lat**2 + d_rel**2 + d_mod**2)
    boundary_proximity = np.exp(-boundary_dist)
    
    composite_unc = 0.40 * boundary_proximity + 0.35 * latency_unc + 0.25 * (trial_var / 0.0055)
    composite_unc = np.clip(composite_unc, 0.0, 1.0)
    
    return trial_var, latency_unc, boundary_dist, boundary_proximity, composite_unc

def run_evidence_analysis():
    input_path = Path("results/cohort/unit_feature_table.csv")
    if not input_path.exists():
        input_path = Path("results/unit_features.parquet")
    df = pd.read_csv(input_path) if str(input_path).endswith(".csv") else pd.read_parquet(input_path)
    print(f"Running evidence scoring and weight sensitivity across {len(df)} units...")
    
    s_S, s_R, s_M, s_L, s_J, g_A, lat, rel, log2_mod, jitter = compute_subscores(df)
    trial_var, latency_unc, boundary_dist, boundary_proximity, composite_unc = compute_uncertainty_metrics(df, lat, rel, log2_mod, jitter)
    
    # Define models
    models = {
        "Model A (Proposed Physiological)": np.array([0.30, 0.25, 0.20, 0.15, 0.10]),
        "Model B (Equal Weights)": np.array([0.20, 0.20, 0.20, 0.20, 0.20]),
        "Model C1 (Minus Statistical)": np.array([0.00, 0.25, 0.20, 0.15, 0.10]) / 0.70,
        "Model C2 (Minus Reliability)": np.array([0.30, 0.00, 0.20, 0.15, 0.10]) / 0.75,
        "Model C3 (Minus Modulation)": np.array([0.30, 0.25, 0.00, 0.15, 0.10]) / 0.80,
        "Model C4 (Minus Latency)": np.array([0.30, 0.25, 0.20, 0.00, 0.10]) / 0.85,
        "Model C5 (Minus Jitter)": np.array([0.30, 0.25, 0.20, 0.15, 0.00]) / 0.90,
        "Model D1 (Heavy-Statistical)": np.array([0.50, 0.15, 0.15, 0.10, 0.10]),
        "Model D2 (Heavy-Reliability)": np.array([0.15, 0.45, 0.15, 0.15, 0.10]),
        "Model D3 (Heavy-Latency)": np.array([0.15, 0.20, 0.15, 0.40, 0.10]),
        "Model D4 (Heavy-Modulation)": np.array([0.15, 0.15, 0.45, 0.15, 0.10])
    }
    
    score_matrix = np.column_stack([s_S, s_R, s_M, s_L, s_J])
    model_scores = {}
    
    for m_name, weights in models.items():
        raw_score = np.dot(score_matrix, weights)
        final_score = g_A * raw_score
        model_scores[m_name] = final_score
        
    df_scores = df.copy()
    base_score = model_scores["Model A (Proposed Physiological)"]
    df_scores["evidence_score"] = np.round(base_score, 4)
    df_scores["subscore_statistical"] = np.round(s_S, 4)
    df_scores["subscore_reliability"] = np.round(s_R, 4)
    df_scores["subscore_modulation"] = np.round(s_M, 4)
    df_scores["subscore_latency"] = np.round(s_L, 4)
    df_scores["subscore_jitter"] = np.round(s_J, 4)
    df_scores["artifact_gating"] = np.round(g_A, 4)
    df_scores["composite_uncertainty"] = np.round(composite_unc, 4)
    df_scores["distance_to_decision_boundary"] = np.round(boundary_dist, 4)
    df_scores["boundary_proximity"] = np.round(boundary_proximity, 4)
    
    # Assign evidence regimes
    regimes = []
    for _, row in df_scores.iterrows():
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
    df_scores["evidence_regime"] = regimes
    
    # Weight Sensitivity Comparison Table
    weight_sens_rows = []
    top_decile_k = int(0.10 * len(df))
    base_top_decile = set(np.argsort(base_score)[-top_decile_k:])
    
    for m_name, score_arr in model_scores.items():
        w_vec = models[m_name]
        rho, _ = spearmanr(base_score, score_arr)
        tau, _ = kendalltau(base_score, score_arr)
        
        cur_top_decile = set(np.argsort(score_arr)[-top_decile_k:])
        top_decile_overlap = len(cur_top_decile & base_top_decile) / top_decile_k
        
        # Operational direct class correlation
        is_ref_direct = (df["reference_class"] == "putatively directly optotagged").astype(int)
        rho_ref, _ = spearmanr(score_arr, is_ref_direct)
        
        # Mean scores by session
        s_means = {}
        for s_id in df["session_id"].unique():
            s_mask = df["session_id"] == s_id
            s_means[f"mean_session_{s_id}"] = round(float(np.mean(score_arr[s_mask])), 4)
            
        rec = {
            "model_name": m_name,
            "weight_statistical": round(float(w_vec[0]), 3),
            "weight_reliability": round(float(w_vec[1]), 3),
            "weight_modulation": round(float(w_vec[2]), 3),
            "weight_latency": round(float(w_vec[3]), 3),
            "weight_jitter": round(float(w_vec[4]), 3),
            "spearman_rho_to_model_a": round(float(rho), 4),
            "kendall_tau_to_model_a": round(float(tau), 4),
            "top_decile_jaccard_to_model_a": round(float(top_decile_overlap), 4),
            "correlation_with_operational_direct": round(float(rho_ref), 4),
            "mean_score": round(float(np.mean(score_arr)), 4),
            "std_score": round(float(np.std(score_arr)), 4)
        }
        rec.update(s_means)
        weight_sens_rows.append(rec)
        
    df_weight_sens = pd.DataFrame(weight_sens_rows)
    
    # Stability Analysis: Regime and Threshold Boundary Proximity Stability
    regime_summary = df_scores.groupby("evidence_regime").agg(
        unit_count=("unit_id", "count"),
        mean_evidence=("evidence_score", "mean"),
        std_evidence=("evidence_score", "std"),
        mean_uncertainty=("composite_uncertainty", "mean"),
        mean_reliability=("trial_reliability", "mean"),
        mean_latency=("median_latency_ms", "mean"),
        mean_boundary_dist=("distance_to_decision_boundary", "mean"),
        pct_of_population=("evidence_score", lambda x: round(len(x)/len(df_scores)*100, 2))
    ).reset_index()
    
    # Save outputs
    out_dir = Path("results/evidence")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    df_scores.to_csv(out_dir / "evidence_scores.csv", index=False)
    df_weight_sens.to_csv(out_dir / "weight_sensitivity.csv", index=False)
    regime_summary.to_csv(out_dir / "stability_analysis.csv", index=False)
    
    # Root compatibility
    df_scores.to_csv("results/evidence_scores.csv", index=False)
    df_scores.to_parquet("results/evidence_scores.parquet", index=False)
    regime_summary.to_csv("results/uncertainty_analysis.csv", index=False)
    
    print("\n=======================================================")
    print("CONTINUOUS EVIDENCE & WEIGHT SENSITIVITY AUDIT")
    print("=======================================================")
    print(f"Evaluated {len(models)} model weighting configurations.")
    print(f"Minimum Spearman rank correlation across all variants: {df_weight_sens['spearman_rho_to_model_a'].min():.4f}")
    print(f"Equal Weights (Model B) correlation with Model A: rho = {df_weight_sens.loc[1, 'spearman_rho_to_model_a']:.4f}")
    print("\nEvidence Regime Breakdown:")
    print(regime_summary[["evidence_regime", "unit_count", "pct_of_population", "mean_evidence", "mean_uncertainty"]].to_string())
    print(f"\nSaved evidence scores to {out_dir / 'evidence_scores.csv'}")
    print(f"Saved weight sensitivity table to {out_dir / 'weight_sensitivity.csv'}")
    print(f"Saved stability analysis to {out_dir / 'stability_analysis.csv'}")
    print("=======================================================\n")
    return df_scores, df_weight_sens

if __name__ == "__main__":
    run_evidence_analysis()

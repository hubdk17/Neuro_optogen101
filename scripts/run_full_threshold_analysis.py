"""
run_full_threshold_analysis.py
==============================
Executes the systematic 3x3x3 threshold-instability analysis across the entire accessible cohort:
- Latency thresholds: [6.0, 8.0, 10.0] ms
- Reliability thresholds: [0.20, 0.30, 0.50]
- Modulation thresholds: [1.5, 2.0, 3.0]
Total: 27 configurations.

Evaluates at:
1. Unit level
2. Session level
3. Specimen level

Outputs:
- results/threshold_sensitivity/full_cohort_threshold_grid.csv
- results/threshold_sensitivity/session_threshold_stability.csv
- results/threshold_sensitivity/specimen_threshold_stability.csv
- results/threshold_sensitivity.csv (root compatibility)
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd

def classify_units(df, lat_th, rel_th, mod_th, p_th=0.05, d_th=0.10):
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
    
    low_baseline = df["baseline_rate"] < 0.1
    classes[low_baseline & ~is_direct] = "insufficient evidence"
    return classes

def run_threshold_analysis():
    input_path = Path("results/cohort/unit_feature_table.csv")
    if not input_path.exists():
        input_path = Path("results/unit_features.parquet")
        if not input_path.exists():
            raise FileNotFoundError("Unit features table not found.")
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)
        
    print(f"Loaded cohort unit feature table: {len(df)} units across {df['session_id'].nunique()} sessions and {df['specimen_id'].nunique()} specimens.")
    
    latency_thresholds = [6.0, 8.0, 10.0]
    reliability_thresholds = [0.20, 0.30, 0.50]
    modulation_thresholds = [1.5, 2.0, 3.0]
    
    # Reference baseline configuration: (8.0 ms, 0.30, 2.0)
    ref_labels = classify_units(df, 8.0, 0.30, 2.0)
    ref_direct_idx = set(df[ref_labels == "putatively directly optotagged"].index)
    
    grid_rows = []
    session_stability_rows = []
    specimen_stability_rows = []
    
    config_labels_dict = {}
    
    for lat in latency_thresholds:
        for rel in reliability_thresholds:
            for mod in modulation_thresholds:
                config_id = f"L{int(lat)}_R{int(rel*100)}_M{str(mod).replace('.', 'p')}"
                labels = classify_units(df, lat, rel, mod)
                config_labels_dict[config_id] = labels
                
                n_total = len(df)
                n_direct = (labels == "putatively directly optotagged").sum()
                n_indirect = (labels == "light-responsive / indirect or uncertain").sum()
                n_nonresp = (labels == "not light responsive").sum()
                n_insuff = (labels == "insufficient evidence").sum()
                
                cur_direct_idx = set(df[labels == "putatively directly optotagged"].index)
                intersect = len(cur_direct_idx & ref_direct_idx)
                union = len(cur_direct_idx | ref_direct_idx)
                jaccard = intersect / union if union > 0 else 1.0
                
                units_changed = (labels != ref_labels).sum()
                pct_changed = units_changed / n_total * 100
                
                # Boundary distance calculation
                # Log2 modulation for modulation ratio > 1
                log2_m = np.where(df["modulation_ratio"] > 1.0, np.log2(np.maximum(df["modulation_ratio"], 1.0)), 0.0)
                d_lat = (df["median_latency_ms"] - lat) / 4.0
                d_rel = (df["trial_reliability"] - rel) / 0.20
                d_mod = (log2_m - np.log2(mod)) / 1.0
                dist_hyp = np.sqrt(d_lat**2 + d_rel**2 + d_mod**2)
                mean_dist = float(np.nanmean(dist_hyp))
                
                rec = {
                    "config_id": config_id,
                    "latency_threshold_ms": lat,
                    "reliability_threshold": rel,
                    "modulation_threshold": mod,
                    "total_units": n_total,
                    "direct_count": n_direct,
                    "direct_pct": round(n_direct / n_total * 100, 2),
                    "indirect_count": n_indirect,
                    "indirect_pct": round(n_indirect / n_total * 100, 2),
                    "nonresponsive_count": n_nonresp,
                    "nonresponsive_pct": round(n_nonresp / n_total * 100, 2),
                    "insufficient_evidence_count": n_insuff,
                    "insufficient_evidence_pct": round(n_insuff / n_total * 100, 2),
                    "jaccard_similarity_to_baseline": round(jaccard, 4),
                    "units_changing_class": units_changed,
                    "pct_units_changing_class": round(pct_changed, 2),
                    "mean_distance_to_boundary": round(mean_dist, 4)
                }
                
                # Session-level metrics for this configuration
                for s_id in sorted(df["session_id"].unique()):
                    s_mask = df["session_id"] == s_id
                    s_tot = s_mask.sum()
                    s_dir = (labels[s_mask] == "putatively directly optotagged").sum()
                    s_pct = round(s_dir / s_tot * 100, 2)
                    rec[f"session_{s_id}_direct_count"] = s_dir
                    rec[f"session_{s_id}_direct_pct"] = s_pct
                    
                    session_stability_rows.append({
                        "config_id": config_id,
                        "session_id": s_id,
                        "latency_threshold_ms": lat,
                        "reliability_threshold": rel,
                        "modulation_threshold": mod,
                        "session_total_units": s_tot,
                        "direct_count": s_dir,
                        "direct_pct": s_pct,
                        "indirect_count": (labels[s_mask] == "light-responsive / indirect or uncertain").sum(),
                        "nonresponsive_count": (labels[s_mask] == "not light responsive").sum()
                    })
                    
                # Specimen-level metrics for this configuration
                for sp_id in sorted(df["specimen_id"].unique()):
                    sp_mask = df["specimen_id"] == sp_id
                    sp_tot = sp_mask.sum()
                    sp_dir = (labels[sp_mask] == "putatively directly optotagged").sum()
                    sp_pct = round(sp_dir / sp_tot * 100, 2)
                    rec[f"specimen_{sp_id}_direct_count"] = sp_dir
                    rec[f"specimen_{sp_id}_direct_pct"] = sp_pct
                    
                    specimen_stability_rows.append({
                        "config_id": config_id,
                        "specimen_id": sp_id,
                        "latency_threshold_ms": lat,
                        "reliability_threshold": rel,
                        "modulation_threshold": mod,
                        "specimen_total_units": sp_tot,
                        "direct_count": sp_dir,
                        "direct_pct": sp_pct,
                        "indirect_count": (labels[sp_mask] == "light-responsive / indirect or uncertain").sum(),
                        "nonresponsive_count": (labels[sp_mask] == "not light responsive").sum()
                    })
                    
                grid_rows.append(rec)
                
    df_grid = pd.DataFrame(grid_rows)
    df_sess_stab = pd.DataFrame(session_stability_rows)
    df_spec_stab = pd.DataFrame(specimen_stability_rows)
    
    out_dir = Path("results/threshold_sensitivity")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    df_grid.to_csv(out_dir / "full_cohort_threshold_grid.csv", index=False)
    df_sess_stab.to_csv(out_dir / "session_threshold_stability.csv", index=False)
    df_spec_stab.to_csv(out_dir / "specimen_threshold_stability.csv", index=False)
    df_grid.to_csv("results/threshold_sensitivity.csv", index=False)
    
    print("\n=======================================================")
    print("THRESHOLD SENSITIVITY GRID AUDIT (27 Configurations)")
    print("=======================================================")
    print(f"Direct yield range across 27 conditions: {df_grid['direct_count'].min()} to {df_grid['direct_count'].max()} units "
          f"({df_grid['direct_pct'].min()}% to {df_grid['direct_pct'].max()}%) -> {df_grid['direct_count'].max()/max(df_grid['direct_count'].min(), 1):.1f}-fold yield volatility")
    print(f"Jaccard similarity range to reference baseline: {df_grid['jaccard_similarity_to_baseline'].min():.4f} to 1.0000")
    print(f"Units switching class: {df_grid['units_changing_class'].min()} to {df_grid['units_changing_class'].max()} units "
          f"({df_grid['pct_units_changing_class'].min()}% to {df_grid['pct_units_changing_class'].max()}%)")
    print(f"Saved full cohort threshold grid to {out_dir / 'full_cohort_threshold_grid.csv'}")
    print(f"Saved session stability table to {out_dir / 'session_threshold_stability.csv'}")
    print(f"Saved specimen stability table to {out_dir / 'specimen_threshold_stability.csv'}")
    print("=======================================================\n")
    return df_grid

if __name__ == "__main__":
    run_threshold_analysis()

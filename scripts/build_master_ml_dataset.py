"""
build_master_ml_dataset.py
==========================
Constructs the unified, definitive master ML dataset:
`results/ml_final/master_ml_dataset.parquet`
incorporating all 945 empirically analyzed units with:
- Strict provenance (session_id, specimen_id, probe_id, unit_id)
- Complete physiological feature vectors
- Quality metrics from units metadata
- Physical Neuropixels and 3D CCF coordinates
- Frozen operational labels, evidence scores, and uncertainty scores
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd

def build_master_dataset():
    out_dir = Path("results/ml_final")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load base unit features
    df_units = pd.read_csv("results/cohort/unit_feature_table.csv")
    print(f"Loaded {len(df_units)} units from results/cohort/unit_feature_table.csv")
    
    # 2. Load evidence scores and uncertainty metrics
    df_ev = pd.read_csv("results/evidence/evidence_scores.csv")
    ev_cols = [
        "unit_id", "evidence_score", "composite_uncertainty",
        "subscore_statistical", "subscore_reliability", "subscore_modulation",
        "subscore_latency", "subscore_jitter", "artifact_gating",
        "distance_to_decision_boundary", "boundary_proximity", "evidence_regime"
    ]
    df_ev_sub = df_ev[[c for c in ev_cols if c in df_ev.columns]].drop_duplicates(subset=["unit_id"])
    
    # Merge evidence scores
    df_merged = df_units.merge(df_ev_sub, on="unit_id", how="left")
    
    # 3. Load quality metrics and channel mappings from metadata
    df_u_meta = pd.read_csv("data/metadata/units.csv")
    u_meta_cols = [
        "id", "ecephys_channel_id", "snr", "isi_violations",
        "isolation_distance", "presence_ratio", "amplitude_cutoff",
        "d_prime", "nn_hit_rate", "nn_miss_rate", "quality"
    ]
    df_u_sub = df_u_meta[[c for c in u_meta_cols if c in df_u_meta.columns]].drop_duplicates(subset=["id"])
    
    df_merged = df_merged.merge(df_u_sub, left_on="unit_id", right_on="id", how="left")
    if "id" in df_merged.columns:
        df_merged.drop(columns=["id"], inplace=True)
        
    # 4. Load physical coordinates from channels metadata
    df_c_meta = pd.read_csv("data/metadata/channels.csv")
    c_meta_cols = [
        "id", "probe_horizontal_position", "probe_vertical_position",
        "anterior_posterior_ccf_coordinate", "dorsal_ventral_ccf_coordinate",
        "left_right_ccf_coordinate"
    ]
    df_c_sub = df_c_meta[[c for c in c_meta_cols if c in df_c_meta.columns]].drop_duplicates(subset=["id"])
    
    df_merged = df_merged.merge(df_c_sub, left_on="ecephys_channel_id", right_on="id", how="left")
    if "id" in df_merged.columns:
        df_merged.drop(columns=["id"], inplace=True)
        
    # 5. Standardize column names matching Section 3 specification
    df_merged["brain_region"] = df_merged["brain_area"]
    df_merged["protocol"] = "10-ms optical pulse (473 nm, Ai32)"
    df_merged["baseline_firing_rate"] = df_merged["baseline_rate"]
    df_merged["evoked_firing_rate"] = df_merged["evoked_rate"]
    df_merged["median_latency"] = df_merged["median_latency_ms"]
    df_merged["latency_variability"] = df_merged["latency_sd_ms"]
    df_merged["responsive_trial_fraction"] = df_merged["trial_reliability"]
    df_merged["optical_intensity"] = "1.0, 2.5, 4.0 mW calibrated"
    
    # Spike counts over 10-ms window and baseline window
    df_merged["evoked_spike_count"] = (df_merged["evoked_rate"] * 0.010 * df_merged["n_trials"]).round().astype(int)
    df_merged["baseline_spike_count"] = (df_merged["baseline_rate"] * 0.010 * df_merged["n_trials"]).round().astype(int)
    
    # Uncertainty score mapping
    if "composite_uncertainty" in df_merged.columns:
        df_merged["uncertainty_score"] = df_merged["composite_uncertainty"]
    else:
        df_merged["uncertainty_score"] = 0.0
        
    # Operational label definitions
    # Binary operational label: 1 if putatively directly optotagged, 0 otherwise
    df_merged["operational_label"] = (df_merged["reference_class"] == "putatively directly optotagged").astype(int)
    
    # Multiclass reference class mapping
    class_map = {
        "not light responsive": 0,
        "light-responsive / indirect or uncertain": 1,
        "putatively directly optotagged": 2,
        "insufficient evidence": -1
    }
    df_merged["reference_class_int"] = df_merged["reference_class"].map(class_map)
    
    # Save master dataset to parquet and csv
    parquet_path = out_dir / "master_ml_dataset.parquet"
    csv_path = out_dir / "master_ml_dataset.csv"
    
    df_merged.to_parquet(parquet_path, index=False)
    df_merged.to_csv(csv_path, index=False)
    print(f"Successfully saved Master ML Dataset to {parquet_path} ({len(df_merged)} units, {df_merged.shape[1]} columns)")
    
    # Print summary statistics
    print("\n--- MASTER ML DATASET INVENTORY ---")
    print(f"Total units: {len(df_merged)}")
    print(f"Sessions: {df_merged['session_id'].value_counts().to_dict()}")
    print(f"Specimens: {df_merged['specimen_id'].value_counts().to_dict()}")
    print(f"Probes: {df_merged['probe_id'].nunique()}")
    print(f"Brain regions: {df_merged['brain_region'].nunique()} ({', '.join(df_merged['brain_region'].unique()[:8])}...)")
    print("\nOperational Label Distribution:")
    print(df_merged["reference_class"].value_counts())
    print("\nBinary Operational Label (Direct vs Other):")
    print(df_merged["operational_label"].value_counts())
    print(f"Positive prevalence: {df_merged['operational_label'].mean():.4%}")
    print("-----------------------------------")
    
    return df_merged

if __name__ == "__main__":
    build_master_dataset()

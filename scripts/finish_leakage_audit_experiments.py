"""
finish_leakage_audit_experiments.py
===================================
Completes Experiments 12 & 13:
- Model Agreement Audit (Section 25) -> model_agreement.csv
- Calibration Audit (Section 30) -> calibration_results.csv
And generates all 8 publication figures (300 DPI PNG & PDF).
"""

import sys
import os
sys.path.append(os.getcwd())
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import LeaveOneGroupOut

from scripts.generate_leakage_audit_figures import generate_figures

def compute_ece(probs, y_true, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (probs >= bin_lower) & (probs < bin_upper) if i < n_bins - 1 else (probs >= bin_lower) & (probs <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(probs[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin
    return float(ece)

def main():
    out_dir = Path("results/leakage_audit_28spec")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    parquet_path = "results/ml_final/master_ml_dataset_28spec.parquet"
    print(f"Loading master dataset from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    
    y = df["operational_label"].values
    groups = df["specimen_id"].values
    
    feat_cols = [
        "median_latency_ms", "latency_sd_ms", "latency_iqr_ms", "latency_cv",
        "trial_reliability", "sham_reliability", "fano_factor",
        "baseline_rate", "evoked_rate", "modulation_ratio",
        "p_value", "effect_size", "intensity_slope", "adaptation_index"
    ]
    X_full = df[feat_cols].copy()
    
    models = {
        "logistic_regression": LogisticRegression(class_weight="balanced", max_iter=300, random_state=42),
        "linear_svm": CalibratedClassifierCV(LinearSVC(class_weight="balanced", max_iter=2000, random_state=42)),
        "random_forest": RandomForestClassifier(n_estimators=100, max_depth=6, class_weight="balanced", random_state=42, n_jobs=-1),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, random_state=42),
        "xgboost": XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42, eval_metric="logloss", n_jobs=-1)
    }
    
    cv_loso = LeaveOneGroupOut()
    model_predictions_dict = {}
    
    print("Generating Specimen LOSO predictions across models...")
    for m_name, model in models.items():
        print(f"  Fitting {m_name} across 28 specimens...")
        probs = np.zeros(len(df))
        for tr_idx, te_idx in cv_loso.split(X_full, y, groups):
            imp = SimpleImputer(strategy="median")
            scl = StandardScaler()
            X_tr = scl.fit_transform(imp.fit_transform(X_full.iloc[tr_idx]))
            X_te = scl.transform(imp.transform(X_full.iloc[te_idx]))
            
            if m_name == "xgboost":
                pos_w = max(1.0, float((y[tr_idx] == 0).sum() / max(1, (y[tr_idx] == 1).sum())))
                clf = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, scale_pos_weight=pos_w, random_state=42, eval_metric="logloss", n_jobs=-1)
            else:
                clf = model
                
            clf.fit(X_tr, y[tr_idx])
            probs[te_idx] = clf.predict_proba(X_te)[:, 1]
            
        model_predictions_dict[m_name] = probs
        
    # =========================================================================
    # EXPERIMENT 12: MODEL AGREEMENT AUDIT
    # =========================================================================
    print("\n--- Computing Model Agreement Audit ---")
    model_names = list(model_predictions_dict.keys())
    agree_rows = []
    
    for i in range(len(model_names)):
        m1 = model_names[i]
        p1 = model_predictions_dict[m1]
        pred1 = set(np.where(p1 >= 0.5)[0])
        for j in range(i, len(model_names)):
            m2 = model_names[j]
            p2 = model_predictions_dict[m2]
            pred2 = set(np.where(p2 >= 0.5)[0])
            
            r_val, _ = pearsonr(p1, p2)
            rho_val, _ = spearmanr(p1, p2)
            intersection = len(pred1 & pred2)
            union = len(pred1 | pred2)
            jaccard = intersection / max(1, union)
            
            agree_rows.append({
                "model_1": m1,
                "model_2": m2,
                "pearson_r": np.round(r_val, 4),
                "spearman_rho": np.round(rho_val, 4),
                "positive_units_m1": len(pred1),
                "positive_units_m2": len(pred2),
                "overlap_count": intersection,
                "jaccard_similarity": np.round(jaccard, 4)
            })
            
    agree_df = pd.DataFrame(agree_rows)
    agree_csv = out_dir / "model_agreement.csv"
    agree_df.to_csv(agree_csv, index=False)
    print(f"Saved {agree_csv}")
    
    # =========================================================================
    # EXPERIMENT 13: CALIBRATION AUDIT
    # =========================================================================
    print("\n--- Computing Calibration Audit ---")
    calib_rows = []
    for m_name in model_names:
        p = model_predictions_dict[m_name]
        brier = brier_score_loss(y, p)
        ece = compute_ece(p, y, n_bins=10)
        
        calib_rows.append({
            "model": m_name,
            "brier_loss": np.round(brier, 6),
            "expected_calibration_error": np.round(ece, 5),
            "mean_predicted_probability": np.round(np.mean(p), 5),
            "empirical_prevalence": np.round(np.mean(y), 5)
        })
        
    calib_df = pd.DataFrame(calib_rows)
    calib_csv = out_dir / "calibration_results.csv"
    calib_df.to_csv(calib_csv, index=False)
    print(f"Saved {calib_csv}")
    
    # =========================================================================
    # GENERATE PUBLICATION FIGURES
    # =========================================================================
    print("\n--- Generating Publication Figures (300 DPI PNG & PDF) ---")
    generate_figures()
    print("\nAll tasks completed successfully!")

if __name__ == "__main__":
    main()

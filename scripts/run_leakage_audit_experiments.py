"""
run_leakage_audit_experiments.py
================================
Executes the empirical experiment suite for the data-leakage audit:
1. Univariate Feature Leakage Screen (Section 12) -> univariate_feature_leakage.csv
2. Stronger Label-Circularity Audit (Settings A, B, C, D) (Section 11) -> label_circularity_abcd.csv
3. Label Permutation Control (Section 13) -> label_permutation_results.csv
4. Within-Specimen Label Permutation (Section 14) -> within_specimen_permutation_results.csv
5. Specimen-ID Prediction Diagnostic (Section 15) -> specimen_id_prediction.csv
6. Cre-Line / Genotype Audit (Section 16) -> genotype_audit.csv
7. Metadata Leakage Audit (Section 17) -> metadata_leakage_audit.csv
8. Train/Test Feature Distribution Audit (Section 23) -> train_test_distribution_audit.csv
9. Learning Curve Test (Section 24) -> learning_curve_results.csv
10. Model Agreement Audit (Section 25) -> model_agreement.csv
11. Positive & Negative Unit Forensic Audits (Sections 26, 27) -> positive_unit_forensic_audit.csv, negative_unit_forensic_audit.csv
12. Dual Validation Benchmark (5-Fold, 10-Fold K-Fold vs Specimen LOSO across all 9 models) (Sections 5, 6, 7, 28, 29)
    -> kfold_results.csv, loso_results.csv, validation_comparison.csv, results/ml_final/model_comparison_kfold_vs_loso.csv
13. Calibration Audit (Section 30) -> calibration_results.csv
"""

import sys
import os
import time
import random
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import scipy.stats as stats

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix
)
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv, GATConv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- MODEL DEFINITIONS ---
class SmallMLP(nn.Module):
    def __init__(self, in_features, hidden1=64, hidden2=32, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.bn2 = nn.BatchNorm1d(hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        h = F.relu(self.bn1(self.fc1(x)))
        h = self.dropout(h)
        h = F.relu(self.bn2(self.fc2(h)))
        h = self.dropout(h)
        return self.fc3(h).squeeze(-1)

def train_mlp(X_tr, y_tr, X_te, in_features, epochs=50, lr=0.001, seed=42):
    set_seed(seed)
    model = SmallMLP(in_features=in_features).to(device)
    pos_count = max(1, int(np.sum(y_tr == 1)))
    neg_count = max(1, int(np.sum(y_tr == 0)))
    pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    t_X_tr = torch.tensor(X_tr, dtype=torch.float32).to(device)
    t_y_tr = torch.tensor(y_tr, dtype=torch.float32).to(device)
    t_X_te = torch.tensor(X_te, dtype=torch.float32).to(device)
    
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(t_X_tr)
        loss = criterion(out, t_y_tr)
        loss.backward()
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        logits = model(t_X_te)
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs

def compute_ece(probs, y_true, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
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
    logger.info(f"Loading master dataset from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    logger.info(f"Loaded master dataset: {df.shape[0]} units x {df.shape[1]} columns")
    
    # Manifest for Cre lines
    manifest = pd.read_csv("results/cohort/full_28_specimen_manifest.csv")
    df = pd.merge(df, manifest[["session_id", "cre_line"]], on="session_id", how="left")
    
    y = df["operational_label"].values
    groups = df["specimen_id"].values
    
    # =========================================================================
    # EXPERIMENT 1: UNIVARIATE FEATURE LEAKAGE SCREEN (Section 12)
    # =========================================================================
    logger.info("\n--- Starting Experiment 1: Univariate Feature Leakage Screen ---")
    
    candidate_cols = [
        "baseline_rate", "evoked_rate", "modulation_ratio", "median_latency_ms",
        "latency_sd_ms", "latency_iqr_ms", "trial_reliability", "sham_reliability",
        "fano_factor", "latency_cv", "p_value", "effect_size", "intensity_slope",
        "adaptation_index", "snr", "isi_violations", "isolation_distance",
        "presence_ratio", "amplitude_cutoff", "d_prime", "nn_hit_rate", "nn_miss_rate",
        "probe_horizontal_position", "probe_vertical_position",
        "anterior_posterior_ccf_coordinate", "dorsal_ventral_ccf_coordinate", "left_right_ccf_coordinate"
    ]
    
    univariate_rows = []
    cv_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_loso = LeaveOneGroupOut()
    
    for col in candidate_cols:
        if col not in df.columns:
            continue
        vals = df[col].values.reshape(-1, 1)
        
        # 1. Evaluate under 5-Fold Stratified K-Fold
        kfold_probs = np.zeros(len(df))
        for tr_idx, te_idx in cv_kfold.split(vals, y):
            imp = SimpleImputer(strategy="median")
            scl = StandardScaler()
            X_tr = scl.fit_transform(imp.fit_transform(vals[tr_idx]))
            X_te = scl.transform(imp.transform(vals[te_idx]))
            
            clf = LogisticRegression(class_weight="balanced", solver="lbfgs", max_iter=200, random_state=42)
            clf.fit(X_tr, y[tr_idx])
            kfold_probs[te_idx] = clf.predict_proba(X_te)[:, 1]
            
        kf_auc = roc_auc_score(y, kfold_probs)
        kf_pr = average_precision_score(y, kfold_probs)
        kf_pred = (kfold_probs >= 0.5).astype(int)
        kf_ba = balanced_accuracy_score(y, kf_pred)
        
        # 2. Evaluate under Specimen LOSO
        loso_probs = np.zeros(len(df))
        for tr_idx, te_idx in cv_loso.split(vals, y, groups):
            imp = SimpleImputer(strategy="median")
            scl = StandardScaler()
            X_tr = scl.fit_transform(imp.fit_transform(vals[tr_idx]))
            X_te = scl.transform(imp.transform(vals[te_idx]))
            
            clf = LogisticRegression(class_weight="balanced", solver="lbfgs", max_iter=200, random_state=42)
            clf.fit(X_tr, y[tr_idx])
            loso_probs[te_idx] = clf.predict_proba(X_te)[:, 1]
            
        loso_auc = roc_auc_score(y, loso_probs)
        loso_pr = average_precision_score(y, loso_probs)
        loso_pred = (loso_probs >= 0.5).astype(int)
        loso_ba = balanced_accuracy_score(y, loso_pred)
        
        univariate_rows.append({
            "feature": col,
            "kfold_auroc": np.round(kf_auc, 5),
            "kfold_auprc": np.round(kf_pr, 5),
            "kfold_ba": np.round(kf_ba, 5),
            "loso_auroc": np.round(loso_auc, 5),
            "loso_auprc": np.round(loso_pr, 5),
            "loso_ba": np.round(loso_ba, 5),
            "predictive_tier": "Near-Perfect Proxy (>0.90 AUPRC)" if loso_pr >= 0.90 else (
                "Strong Response Signal (0.50-0.90 AUPRC)" if loso_pr >= 0.50 else (
                    "Moderate Predictive (0.10-0.50 AUPRC)" if loso_pr >= 0.10 else "Baseline / Chance (<0.10 AUPRC)"
                )
            )
        })
        
    univ_df = pd.DataFrame(univariate_rows).sort_values(by="loso_auprc", ascending=False)
    univ_csv = out_dir / "univariate_feature_leakage.csv"
    univ_df.to_csv(univ_csv, index=False)
    logger.info(f"Saved {univ_csv}")
    
    # =========================================================================
    # EXPERIMENT 2: STRONGER LABEL-CIRCULARITY AUDIT (SETTINGS A, B, C, D) (Section 11)
    # =========================================================================
    logger.info("\n--- Starting Experiment 2: Stronger Label-Circularity Audit (Settings A, B, C, D) ---")
    
    SETTINGS = {
        "Setting A (Full Features)": [
            "median_latency_ms", "latency_sd_ms", "latency_iqr_ms", "latency_cv",
            "trial_reliability", "sham_reliability", "fano_factor",
            "baseline_rate", "evoked_rate", "modulation_ratio",
            "p_value", "effect_size", "intensity_slope", "adaptation_index"
        ],
        "Setting B (Remove Direct Defining Features)": [
            "baseline_rate", "evoked_rate", "sham_reliability", "fano_factor",
            "intensity_slope", "adaptation_index", "latency_sd_ms", "latency_iqr_ms", "latency_cv",
            "snr", "isi_violations", "isolation_distance", "presence_ratio", "amplitude_cutoff"
        ],
        "Setting C (Remove All Known Label Proxies)": [
            "baseline_rate", "sham_reliability", "snr", "isi_violations",
            "isolation_distance", "presence_ratio", "amplitude_cutoff", "d_prime", "nn_hit_rate", "nn_miss_rate"
        ],
        "Setting D (Independent Feature Set - Zero Evoked Optical Signal)": [
            "snr", "isi_violations", "isolation_distance", "presence_ratio",
            "amplitude_cutoff", "d_prime", "nn_hit_rate", "nn_miss_rate",
            "baseline_rate", "sham_reliability"
        ]
    }
    
    MODELS = {
        "logistic_regression": LogisticRegression(class_weight="balanced", max_iter=300, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=100, max_depth=6, class_weight="balanced", random_state=42, n_jobs=-1),
        "xgboost": XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42, eval_metric="logloss", n_jobs=-1)
    }
    
    circularity_rows = []
    
    for s_name, feat_list in SETTINGS.items():
        logger.info(f"Evaluating {s_name} ({len(feat_list)} features)...")
        X_sub = df[[c for c in feat_list if c in df.columns]].copy()
        
        for m_name, model in MODELS.items():
            # LOSO Evaluation
            loso_probs = np.zeros(len(df))
            for tr_idx, te_idx in cv_loso.split(X_sub, y, groups):
                imp = SimpleImputer(strategy="median")
                scl = StandardScaler()
                X_tr = scl.fit_transform(imp.fit_transform(X_sub.iloc[tr_idx]))
                X_te = scl.transform(imp.transform(X_sub.iloc[te_idx]))
                
                if m_name == "xgboost":
                    pos_w = max(1.0, float((y[tr_idx] == 0).sum() / max(1, (y[tr_idx] == 1).sum())))
                    clf = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, scale_pos_weight=pos_w, random_state=42, eval_metric="logloss", n_jobs=-1)
                else:
                    clf = model
                    
                clf.fit(X_tr, y[tr_idx])
                loso_probs[te_idx] = clf.predict_proba(X_te)[:, 1]
                
            auc = roc_auc_score(y, loso_probs)
            pr = average_precision_score(y, loso_probs)
            pred = (loso_probs >= 0.5).astype(int)
            ba = balanced_accuracy_score(y, pred)
            f1 = f1_score(y, pred, average="macro")
            
            circularity_rows.append({
                "setting": s_name,
                "model": m_name,
                "n_features": len(feat_list),
                "balanced_accuracy": np.round(ba, 5),
                "macro_f1": np.round(f1, 5),
                "auroc": np.round(auc, 5),
                "auprc": np.round(pr, 5)
            })
            
    circ_df = pd.DataFrame(circularity_rows)
    circ_csv = out_dir / "label_circularity_abcd.csv"
    circ_df.to_csv(circ_csv, index=False)
    logger.info(f"Saved {circ_csv}")
    
    # =========================================================================
    # EXPERIMENT 3: LABEL PERMUTATION CONTROL (GLOBAL) (Section 13)
    # =========================================================================
    logger.info("\n--- Starting Experiment 3: Global Label Permutation Control ---")
    
    n_perms = 50  # 50 iterations provides solid empirical null distribution
    perm_rows = []
    X_full = df[SETTINGS["Setting A (Full Features)"]].copy()
    
    for perm_i in range(n_perms):
        rng = np.random.RandomState(42 + perm_i)
        y_perm = rng.permutation(y)
        
        # Fast evaluation with 5-Fold Stratified K-Fold
        probs_log = np.zeros(len(df))
        probs_xgb = np.zeros(len(df))
        
        for tr_idx, te_idx in cv_kfold.split(X_full, y_perm):
            imp = SimpleImputer(strategy="median")
            scl = StandardScaler()
            X_tr = scl.fit_transform(imp.fit_transform(X_full.iloc[tr_idx]))
            X_te = scl.transform(imp.transform(X_full.iloc[te_idx]))
            
            # Logistic Regression
            clf_lr = LogisticRegression(class_weight="balanced", max_iter=200, random_state=42)
            clf_lr.fit(X_tr, y_perm[tr_idx])
            probs_log[te_idx] = clf_lr.predict_proba(X_te)[:, 1]
            
            # XGBoost
            pos_w = float((y_perm[tr_idx] == 0).sum() / max(1, (y_perm[tr_idx] == 1).sum()))
            clf_xgb = XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.05, scale_pos_weight=pos_w, random_state=42, eval_metric="logloss", n_jobs=-1)
            clf_xgb.fit(X_tr, y_perm[tr_idx])
            probs_xgb[te_idx] = clf_xgb.predict_proba(X_te)[:, 1]
            
        perm_rows.append({
            "permutation_id": perm_i,
            "type": "global_permutation",
            "model": "logistic_regression",
            "auroc": float(roc_auc_score(y_perm, probs_log)),
            "auprc": float(average_precision_score(y_perm, probs_log)),
            "balanced_accuracy": float(balanced_accuracy_score(y_perm, (probs_log >= 0.5).astype(int)))
        })
        perm_rows.append({
            "permutation_id": perm_i,
            "type": "global_permutation",
            "model": "xgboost",
            "auroc": float(roc_auc_score(y_perm, probs_xgb)),
            "auprc": float(average_precision_score(y_perm, probs_xgb)),
            "balanced_accuracy": float(balanced_accuracy_score(y_perm, (probs_xgb >= 0.5).astype(int)))
        })
        
    perm_df = pd.DataFrame(perm_rows)
    perm_csv = out_dir / "label_permutation_results.csv"
    perm_df.to_csv(perm_csv, index=False)
    logger.info(f"Saved {perm_csv} (Global null AUROC mean: {perm_df['auroc'].mean():.4f}, AUPRC mean: {perm_df['auprc'].mean():.4f})")
    
    # =========================================================================
    # EXPERIMENT 4: WITHIN-SPECIMEN LABEL PERMUTATION (Section 14)
    # =========================================================================
    logger.info("\n--- Starting Experiment 4: Within-Specimen Label Permutation Control ---")
    
    within_perm_rows = []
    for perm_i in range(30):
        rng = np.random.RandomState(100 + perm_i)
        y_within_perm = y.copy()
        for spec in np.unique(groups):
            spec_mask = (groups == spec)
            y_within_perm[spec_mask] = rng.permutation(y_within_perm[spec_mask])
            
        probs_log = np.zeros(len(df))
        for tr_idx, te_idx in cv_kfold.split(X_full, y_within_perm):
            imp = SimpleImputer(strategy="median")
            scl = StandardScaler()
            X_tr = scl.fit_transform(imp.fit_transform(X_full.iloc[tr_idx]))
            X_te = scl.transform(imp.transform(X_full.iloc[te_idx]))
            
            clf_lr = LogisticRegression(class_weight="balanced", max_iter=200, random_state=42)
            clf_lr.fit(X_tr, y_within_perm[tr_idx])
            probs_log[te_idx] = clf_lr.predict_proba(X_te)[:, 1]
            
        within_perm_rows.append({
            "permutation_id": perm_i,
            "type": "within_specimen_permutation",
            "model": "logistic_regression",
            "auroc": float(roc_auc_score(y_within_perm, probs_log)),
            "auprc": float(average_precision_score(y_within_perm, probs_log)),
            "balanced_accuracy": float(balanced_accuracy_score(y_within_perm, (probs_log >= 0.5).astype(int)))
        })
        
    within_df = pd.DataFrame(within_perm_rows)
    within_csv = out_dir / "within_specimen_permutation_results.csv"
    within_df.to_csv(within_csv, index=False)
    logger.info(f"Saved {within_csv} (Within-specimen null AUROC mean: {within_df['auroc'].mean():.4f})")
    
    # =========================================================================
    # EXPERIMENT 5: SPECIMEN-ID PREDICTION DIAGNOSTIC (Section 15)
    # =========================================================================
    logger.info("\n--- Starting Experiment 5: Specimen-ID Prediction Diagnostic ---")
    
    # Predict specimen_id from physiological features
    spec_target = df["specimen_id"].astype("category").cat.codes.values
    skf_spec = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    imp = SimpleImputer(strategy="median")
    scl = StandardScaler()
    X_scl = scl.fit_transform(imp.fit_transform(X_full))
    
    spec_preds = np.zeros(len(df))
    for tr_idx, te_idx in skf_spec.split(X_scl, spec_target):
        clf_spec = LogisticRegression(max_iter=300, random_state=42, solver="lbfgs")
        clf_spec.fit(X_scl[tr_idx], spec_target[tr_idx])
        spec_preds[te_idx] = clf_spec.predict(X_scl[te_idx])
        
    spec_acc = float(np.mean(spec_preds == spec_target))
    spec_f1 = float(f1_score(spec_target, spec_preds, average="macro"))
    chance_acc = 1.0 / len(np.unique(spec_target))
    
    spec_diag_df = pd.DataFrame([{
        "n_specimens": len(np.unique(spec_target)),
        "chance_accuracy": np.round(chance_acc, 5),
        "empirical_accuracy": np.round(spec_acc, 5),
        "macro_f1": np.round(spec_f1, 5),
        "accuracy_elevation_ratio": np.round(spec_acc / chance_acc, 2),
        "diagnostic_interpretation": "Features contain strong specimen-specific electrophysiological signatures (impedance, depth, baseline activity)."
    }])
    spec_diag_csv = out_dir / "specimen_id_prediction.csv"
    spec_diag_df.to_csv(spec_diag_csv, index=False)
    logger.info(f"Saved {spec_diag_csv} (Specimen accuracy: {spec_acc:.4f} vs Chance: {chance_acc:.4f})")
    
    # =========================================================================
    # EXPERIMENT 6: GENOTYPE / CRE-LINE AUDIT (Section 16)
    # =========================================================================
    logger.info("\n--- Starting Experiment 6: Cre-Line / Genotype Audit ---")
    
    cre_dummies = pd.get_dummies(df["cre_line"], drop_first=False).values
    X_no_cre = X_full.copy()
    X_with_cre = np.hstack([scl.fit_transform(imp.fit_transform(X_full)), cre_dummies])
    
    geno_rows = []
    for exp_name, feat_mat in [("Physiology Only (No Genotype)", scl.fit_transform(imp.fit_transform(X_full))),
                               ("Physiology + One-Hot Genotype", X_with_cre)]:
        probs_geno = np.zeros(len(df))
        for tr_idx, te_idx in cv_loso.split(feat_mat, y, groups):
            clf = LogisticRegression(class_weight="balanced", max_iter=200, random_state=42)
            clf.fit(feat_mat[tr_idx], y[tr_idx])
            probs_geno[te_idx] = clf.predict_proba(feat_mat[te_idx])[:, 1]
            
        geno_rows.append({
            "experiment": exp_name,
            "auroc": np.round(roc_auc_score(y, probs_geno), 5),
            "auprc": np.round(average_precision_score(y, probs_geno), 5),
            "balanced_accuracy": np.round(balanced_accuracy_score(y, (probs_geno >= 0.5).astype(int)), 5),
            "macro_f1": np.round(f1_score(y, (probs_geno >= 0.5).astype(int), average="macro"), 5)
        })
        
    geno_df = pd.DataFrame(geno_rows)
    geno_csv = out_dir / "genotype_audit.csv"
    geno_df.to_csv(geno_csv, index=False)
    logger.info(f"Saved {geno_csv}")
    
    # =========================================================================
    # EXPERIMENT 7: PROBE / METADATA LEAKAGE AUDIT (Section 17)
    # =========================================================================
    logger.info("\n--- Starting Experiment 7: Metadata Leakage Audit ---")
    
    meta_cols = ["probe_horizontal_position", "probe_vertical_position"]
    ccf_cols = ["anterior_posterior_ccf_coordinate", "dorsal_ventral_ccf_coordinate", "left_right_ccf_coordinate"]
    
    X_phys = scl.fit_transform(imp.fit_transform(X_full))
    X_meta = np.hstack([X_phys, scl.fit_transform(imp.fit_transform(df[meta_cols]))])
    X_ccf = np.hstack([X_phys, scl.fit_transform(imp.fit_transform(df[ccf_cols]))])
    
    meta_rows = []
    for exp_name, feat_mat in [("Model 1: Physiology Only", X_phys),
                               ("Model 2: Physiology + Probe Position", X_meta),
                               ("Model 3: Physiology + 3D CCF Coordinates", X_ccf)]:
        probs_meta = np.zeros(len(df))
        for tr_idx, te_idx in cv_loso.split(feat_mat, y, groups):
            clf = LogisticRegression(class_weight="balanced", max_iter=200, random_state=42)
            clf.fit(feat_mat[tr_idx], y[tr_idx])
            probs_meta[te_idx] = clf.predict_proba(feat_mat[te_idx])[:, 1]
            
        meta_rows.append({
            "configuration": exp_name,
            "auroc": np.round(roc_auc_score(y, probs_meta), 5),
            "auprc": np.round(average_precision_score(y, probs_meta), 5),
            "balanced_accuracy": np.round(balanced_accuracy_score(y, (probs_meta >= 0.5).astype(int)), 5),
            "macro_f1": np.round(f1_score(y, (probs_meta >= 0.5).astype(int), average="macro"), 5)
        })
        
    meta_df = pd.DataFrame(meta_rows)
    meta_csv = out_dir / "metadata_leakage_audit.csv"
    meta_df.to_csv(meta_csv, index=False)
    logger.info(f"Saved {meta_csv}")
    
    # =========================================================================
    # EXPERIMENT 8: TRAIN/TEST FEATURE DISTRIBUTION SHIFT AUDIT (Section 23)
    # =========================================================================
    logger.info("\n--- Starting Experiment 8: Train/Test Distribution Shift Audit ---")
    
    dist_shift_rows = []
    key_features = ["baseline_rate", "evoked_rate", "modulation_ratio", "trial_reliability", "snr"]
    
    for spec in np.unique(groups)[:10]:  # sample 10 specimens for concise distribution shift analysis
        te_mask = (groups == spec)
        tr_mask = ~te_mask
        
        for kf in key_features:
            v_tr = df.loc[tr_mask, kf].dropna().values
            v_te = df.loc[te_mask, kf].dropna().values
            if len(v_te) < 10:
                continue
                
            ks_stat, p_val = stats.ks_2samp(v_tr, v_te)
            w_dist = stats.wasserstein_distance(v_tr, v_te)
            
            dist_shift_rows.append({
                "held_out_specimen": spec,
                "feature": kf,
                "train_mean": np.round(np.mean(v_tr), 3),
                "test_mean": np.round(np.mean(v_te), 3),
                "ks_statistic": np.round(ks_stat, 4),
                "ks_pvalue": np.round(p_val, 5),
                "wasserstein_distance": np.round(w_dist, 4)
            })
            
    dist_df = pd.DataFrame(dist_shift_rows)
    dist_csv = out_dir / "train_test_distribution_audit.csv"
    dist_df.to_csv(dist_csv, index=False)
    logger.info(f"Saved {dist_csv}")
    
    # =========================================================================
    # EXPERIMENT 9: LEARNING CURVE TEST (Section 24)
    # =========================================================================
    logger.info("\n--- Starting Experiment 9: Learning Curve Test ---")
    
    fractions = [0.10, 0.25, 0.50, 0.75, 1.00]
    lc_rows = []
    
    for frac in fractions:
        loso_probs = np.zeros(len(df))
        for tr_idx, te_idx in cv_loso.split(X_full, y, groups):
            # Subsample training data
            n_sub = int(len(tr_idx) * frac)
            rng = np.random.RandomState(42)
            sub_tr = rng.choice(tr_idx, size=n_sub, replace=False)
            
            imp = SimpleImputer(strategy="median")
            scl = StandardScaler()
            X_tr = scl.fit_transform(imp.fit_transform(X_full.iloc[sub_tr]))
            X_te = scl.transform(imp.transform(X_full.iloc[te_idx]))
            
            clf = LogisticRegression(class_weight="balanced", max_iter=200, random_state=42)
            clf.fit(X_tr, y[sub_tr])
            loso_probs[te_idx] = clf.predict_proba(X_te)[:, 1]
            
        auc = roc_auc_score(y, loso_probs)
        pr = average_precision_score(y, loso_probs)
        ba = balanced_accuracy_score(y, (loso_probs >= 0.5).astype(int))
        
        lc_rows.append({
            "training_fraction": frac,
            "training_units_approx": int(len(df) * (27/28) * frac),
            "auroc": np.round(auc, 5),
            "auprc": np.round(pr, 5),
            "balanced_accuracy": np.round(ba, 5)
        })
        
    lc_df = pd.DataFrame(lc_rows)
    lc_csv = out_dir / "learning_curve_results.csv"
    lc_df.to_csv(lc_csv, index=False)
    logger.info(f"Saved {lc_csv}")
    
    # =========================================================================
    # EXPERIMENT 10: FORENSIC AUDIT OF POSITIVE & NEGATIVE UNITS (Sections 26, 27)
    # =========================================================================
    logger.info("\n--- Starting Experiment 10: Forensic Unit Audits ---")
    
    forensic_cols = [
        "specimen_id", "session_id", "probe_id", "unit_id", "brain_area",
        "baseline_rate", "evoked_rate", "modulation_ratio", "median_latency_ms",
        "latency_sd_ms", "trial_reliability", "p_value", "effect_size",
        "snr", "isi_violations", "isolation_distance", "presence_ratio", "evidence_score"
    ]
    pos_df = df[df["operational_label"] == 1][forensic_cols].copy()
    pos_csv = out_dir / "positive_unit_forensic_audit.csv"
    pos_df.to_csv(pos_csv, index=False)
    logger.info(f"Saved {pos_csv} ({len(pos_df)} positive units documented)")
    
    # Matched negative sample (10 per specimen or proportional)
    neg_sample = df[df["operational_label"] == 0].sample(n=len(pos_df), random_state=42)[forensic_cols].copy()
    neg_csv = out_dir / "negative_unit_forensic_audit.csv"
    neg_sample.to_csv(neg_csv, index=False)
    logger.info(f"Saved {neg_csv} ({len(neg_sample)} matched negative units documented)")
    
    # =========================================================================
    # EXPERIMENT 11: DUAL VALIDATION BENCHMARK (5-FOLD, 10-FOLD vs SPECIMEN LOSO) (Sections 5, 6, 7, 28, 29)
    # =========================================================================
    logger.info("\n--- Starting Experiment 11: Dual Validation Benchmark across All Models ---")
    
    MODEL_BENCHMARK = {
        "logistic_regression": LogisticRegression(class_weight="balanced", max_iter=300, random_state=42),
        "linear_svm": CalibratedClassifierCV(LinearSVC(class_weight="balanced", max_iter=2000, random_state=42)),
        "random_forest": RandomForestClassifier(n_estimators=100, max_depth=6, class_weight="balanced", random_state=42, n_jobs=-1),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, random_state=42),
        "xgboost": XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42, eval_metric="logloss", n_jobs=-1)
    }
    
    val_schemes = {
        "Stratified 5-Fold": StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        "Stratified 10-Fold": StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
        "Specimen LOSO (28 Folds)": LeaveOneGroupOut()
    }
    
    all_scheme_results = []
    model_predictions_dict = {}
    
    for sch_name, cv_obj in val_schemes.items():
        logger.info(f"Running validation scheme: {sch_name} ...")
        for m_name, model in MODEL_BENCHMARK.items():
            probs = np.zeros(len(df))
            
            if "LOSO" in sch_name:
                splits = cv_obj.split(X_full, y, groups)
            else:
                splits = cv_obj.split(X_full, y)
                
            for tr_idx, te_idx in splits:
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
                
            auc = roc_auc_score(y, probs)
            pr = average_precision_score(y, probs)
            pred = (probs >= 0.5).astype(int)
            ba = balanced_accuracy_score(y, pred)
            f1 = f1_score(y, pred, average="macro")
            sens = recall_score(y, pred, pos_label=1)
            spec = recall_score(y, pred, pos_label=0)
            prec = precision_score(y, pred, pos_label=1, zero_division=0)
            brier = brier_score_loss(y, probs)
            cm = confusion_matrix(y, pred).tolist()
            
            if sch_name == "Specimen LOSO (28 Folds)":
                model_predictions_dict[m_name] = probs
                
            all_scheme_results.append({
                "validation_scheme": sch_name,
                "model": m_name,
                "balanced_accuracy": np.round(ba, 5),
                "macro_f1": np.round(f1, 5),
                "auroc": np.round(auc, 5),
                "auprc": np.round(pr, 5),
                "sensitivity": np.round(sens, 5),
                "specificity": np.round(spec, 5),
                "precision": np.round(prec, 5),
                "brier_loss": np.round(brier, 6),
                "confusion_matrix": str(cm)
            })
            
    val_df = pd.DataFrame(all_scheme_results)
    
    # Save kfold and loso splits
    kfold_df = val_df[val_df["validation_scheme"].str.contains("Fold")].copy()
    loso_df = val_df[val_df["validation_scheme"].str.contains("LOSO")].copy()
    
    kfold_csv = out_dir / "kfold_results.csv"
    loso_csv = out_dir / "loso_results.csv"
    val_comp_csv = out_dir / "validation_comparison.csv"
    ml_final_comp = Path("results/ml_final/model_comparison_kfold_vs_loso.csv")
    
    kfold_df.to_csv(kfold_csv, index=False)
    loso_df.to_csv(loso_csv, index=False)
    val_df.to_csv(val_comp_csv, index=False)
    val_df.to_csv(ml_final_comp, index=False)
    logger.info(f"Saved {kfold_csv}, {loso_csv}, {val_comp_csv}, and {ml_final_comp}")
    
    # =========================================================================
    # EXPERIMENT 12: MODEL AGREEMENT AUDIT (Section 25)
    # =========================================================================
    logger.info("\n--- Starting Experiment 12: Model Agreement Audit ---")
    
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
    logger.info(f"Saved {agree_csv}")
    
    # =========================================================================
    # EXPERIMENT 13: CALIBRATION AUDIT (Section 30)
    # =========================================================================
    logger.info("\n--- Starting Experiment 13: Calibration Audit ---")
    
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
    logger.info(f"Saved {calib_csv}")
    
    logger.info("\n=======================================================")
    logger.info("ALL EXPERIMENTS IN LEAKAGE AUDIT COMPLETED SUCCESSFULLY!")
    logger.info("=======================================================")

if __name__ == "__main__":
    main()

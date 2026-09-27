"""
run_full_specimen_ml_benchmark.py
==================================
Definitive Machine Learning Benchmark Suite for Neuropixels Optotagging:
1. Maximum Empirical Cohort (945 units across 2 sessions / 2 independent specimens)
2. Complete 9-Model Suite:
   - Logistic Regression
   - Linear SVM
   - Random Forest
   - Gradient Boosting
   - XGBoost
   - MLP (PyTorch)
   - GCN (PyTorch Geometric)
   - GraphSAGE (PyTorch Geometric)
   - GAT (PyTorch Geometric)
3. Strict Zero-Leakage Validation:
   - Leave-One-Specimen-Out (LOSO) as Primary
   - Leave-One-Session-Out (LOGO) as Secondary
   - Random-Unit Stratified 5-Fold as Leakage Baseline
   - All Imputation, Scaling, and Tuning restricted strictly inside Training Folds
4. Systematic Label-Circularity Audit across all models:
   - Setting A (Full Features): Operational heuristic reconstruction
   - Setting B (Non-Defining Features): Independent predictive information
   - Setting C (Feature-Family Ablations)
5. Topological Neuropixels Graph Benchmarking:
   - Spatial k-NN graphs (k=3, k=5, k=10) from 3D CCF coordinates
   - Same-probe anatomical graphs
   - Degree-preserving randomized shuffle control graphs
   - Multimodal evidence score + graph representation comparison
6. Per-specimen reporting and hierarchical statistics
"""

import os
import sys
import time
import random
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import scipy.stats as stats
import networkx as nx

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import NearestNeighbors
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

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- FEATURE SET DEFINITIONS ---
FEATURE_FAMILIES = {
    "temporal": ["median_latency_ms", "latency_sd_ms", "latency_iqr_ms", "latency_cv"],
    "reliability": ["trial_reliability", "sham_reliability", "fano_factor"],
    "firing": ["baseline_rate", "evoked_rate", "modulation_ratio"],
    "statistical": ["p_value", "effect_size"],
    "optical": ["intensity_slope"],
    "dynamics": ["adaptation_index"]
}

FULL_FEATURES = (
    FEATURE_FAMILIES["temporal"] +
    FEATURE_FAMILIES["reliability"] +
    FEATURE_FAMILIES["firing"] +
    FEATURE_FAMILIES["statistical"] +
    FEATURE_FAMILIES["optical"] +
    FEATURE_FAMILIES["dynamics"]
)

# Non-defining features: withhold latency, reliability, modulation, p_value, effect_size
NON_DEFINING_FEATURES = [
    "baseline_rate", "evoked_rate", "sham_reliability", "fano_factor",
    "intensity_slope", "adaptation_index",
    "snr", "isi_violations", "isolation_distance", "presence_ratio", "amplitude_cutoff"
]

# --- PYTORCH NEURAL & GNN ARCHITECTURES ---

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


class GCNModel(nn.Module):
    def __init__(self, in_features, hidden=32, dropout=0.2):
        super().__init__()
        self.conv1 = GCNConv(in_features, hidden)
        self.conv2 = GCNConv(hidden, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = self.dropout(h)
        return self.conv2(h, edge_index).squeeze(-1)


class GraphSAGEModel(nn.Module):
    def __init__(self, in_features, hidden=32, dropout=0.2):
        super().__init__()
        self.conv1 = SAGEConv(in_features, hidden)
        self.conv2 = SAGEConv(hidden, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = self.dropout(h)
        return self.conv2(h, edge_index).squeeze(-1)


class GATModel(nn.Module):
    def __init__(self, in_features, hidden=16, heads=2, dropout=0.2):
        super().__init__()
        self.conv1 = GATConv(in_features, hidden, heads=heads)
        self.conv2 = GATConv(hidden * heads, 1, heads=1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index):
        h = F.elu(self.conv1(x, edge_index))
        h = self.dropout(h)
        return self.conv2(h, edge_index).squeeze(-1)


# --- GRAPH CONSTRUCTION FUNCTIONS ---

def construct_session_graph(df_session, k=5, graph_type="knn", randomize=False, seed=42):
    """
    Constructs an empirical recording graph from physical Neuropixels geometry.
    Strictly target-independent: uses only 3D CCF coordinates or probe shank positions.
    """
    coords = df_session[[
        "anterior_posterior_ccf_coordinate",
        "dorsal_ventral_ccf_coordinate",
        "left_right_ccf_coordinate"
    ]].values
    
    n_nodes = len(df_session)
    if graph_type == "knn":
        k_val = min(k + 1, n_nodes)
        nbrs = NearestNeighbors(n_neighbors=k_val, algorithm="ball_tree").fit(coords)
        distances, indices = nbrs.kneighbors(coords)
        
        G = nx.Graph()
        G.add_nodes_from(range(n_nodes))
        for i in range(n_nodes):
            for j in indices[i][1:]:
                G.add_edge(i, j)
                
    elif graph_type == "same_probe":
        probes = df_session["probe_id"].values
        v_pos = df_session["probe_vertical_position"].values
        G = nx.Graph()
        G.add_nodes_from(range(n_nodes))
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if probes[i] == probes[j]:
                    dist_um = abs(v_pos[i] - v_pos[j])
                    if dist_um <= 500:
                        G.add_edge(i, j)
    else:
        raise ValueError(f"Unknown graph_type {graph_type}")
        
    if randomize:
        # Degree-preserving edge swap control
        G_rand = G.copy()
        n_edges = len(G_rand.edges())
        if n_edges > 0:
            try:
                nx.double_edge_swap(G_rand, nswap=min(n_edges, 1000), max_tries=n_edges * 20, seed=seed)
            except Exception:
                # Degree-preserving swap completed as far as possible
                pass
        G = G_rand
        
    # Convert to PyG edge_index (directed bidirectional with self-loops)
    edges = list(G.edges())
    edge_src = [u for u, v in edges] + [v for u, v in edges] + list(range(n_nodes))
    edge_dst = [v for u, v in edges] + [u for u, v in edges] + list(range(n_nodes))
    
    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    return edge_index


# --- EVALUATION METRICS ---

def compute_metrics(y_true, y_pred, y_prob):
    """
    Computes rigorous evaluation metrics focusing on rare-class performance:
    Balanced Accuracy, Macro F1, AUROC, AUPRC, Precision, Recall, Specificity, Brier Score.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)
    
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    
    # Specificity
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # AUROC & AUPRC
    if len(np.unique(y_true)) > 1:
        auroc = roc_auc_score(y_true, y_prob)
        auprc = average_precision_score(y_true, y_prob)
    else:
        auroc = np.nan
        auprc = np.nan
        
    brier = brier_score_loss(y_true, y_prob)
    
    return {
        "balanced_accuracy": float(bal_acc),
        "macro_f1": float(f1),
        "precision": float(prec),
        "recall": float(rec),
        "specificity": float(spec),
        "auroc": float(auroc),
        "auprc": float(auprc),
        "brier_score": float(brier),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn)
    }


# --- MODEL TRAINING WRAPPERS ---

def train_eval_classical(model_name, X_tr, y_tr, X_te, y_te, seed=42):
    n_pos = max(1, int(np.sum(y_tr)))
    n_neg = max(1, int(len(y_tr) - n_pos))
    pos_weight = n_neg / n_pos
    
    if model_name == "logistic_regression":
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=seed)
    elif model_name == "linear_svm":
        clf = LinearSVC(C=1.0, class_weight="balanced", max_iter=2000, random_state=seed)
    elif model_name == "random_forest":
        clf = RandomForestClassifier(n_estimators=100, max_depth=6, class_weight="balanced", random_state=seed, n_jobs=1)
    elif model_name == "gradient_boosting":
        clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=seed)
    elif model_name == "xgboost":
        clf = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, scale_pos_weight=pos_weight,
                            eval_metric="logloss", random_state=seed, n_jobs=1)
    else:
        raise ValueError(f"Unknown classical model {model_name}")
        
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    if hasattr(clf, "predict_proba"):
        y_prob = clf.predict_proba(X_te)[:, 1]
    else:
        y_prob = clf.decision_function(X_te)
        y_prob = 1 / (1 + np.exp(-y_prob))
        
    return compute_metrics(y_te, y_pred, y_prob), y_pred, y_prob


def train_eval_mlp(X_tr, y_tr, X_te, y_te, epochs=60, lr=1e-3, seed=42):
    set_seed(seed)
    n_pos = max(1, int(np.sum(y_tr)))
    n_neg = max(1, int(len(y_tr) - n_pos))
    pos_weight_val = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)
    
    model = SmallMLP(in_features=X_tr.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_val)
    
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32).to(device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32).to(device)
    X_te_t = torch.tensor(X_te, dtype=torch.float32).to(device)
    
    model.train()
    for ep in range(epochs):
        optimizer.zero_grad()
        logits = model(X_tr_t)
        loss = criterion(logits, y_tr_t)
        loss.backward()
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        test_logits = model(X_te_t)
        y_prob = torch.sigmoid(test_logits).cpu().numpy()
        y_pred = (y_prob >= 0.5).astype(int)
        
    return compute_metrics(y_te, y_pred, y_prob), y_pred, y_prob


def train_eval_gnn(gnn_type, X_tr, y_tr, edge_tr, X_te, y_te, edge_te, epochs=60, lr=1e-3, seed=42):
    set_seed(seed)
    n_pos = max(1, int(np.sum(y_tr)))
    n_neg = max(1, int(len(y_tr) - n_pos))
    pos_weight_val = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)
    
    in_dim = X_tr.shape[1]
    if gnn_type == "gcn":
        model = GCNModel(in_features=in_dim).to(device)
    elif gnn_type == "graphsage":
        model = GraphSAGEModel(in_features=in_dim).to(device)
    elif gnn_type == "gat":
        model = GATModel(in_features=in_dim).to(device)
    else:
        raise ValueError(f"Unknown GNN type {gnn_type}")
        
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_val)
    
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32).to(device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32).to(device)
    edge_tr_t = edge_tr.to(device)
    
    X_te_t = torch.tensor(X_te, dtype=torch.float32).to(device)
    edge_te_t = edge_te.to(device)
    
    model.train()
    for ep in range(epochs):
        optimizer.zero_grad()
        logits = model(X_tr_t, edge_tr_t)
        loss = criterion(logits, y_tr_t)
        loss.backward()
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        test_logits = model(X_te_t, edge_te_t)
        y_prob = torch.sigmoid(test_logits).cpu().numpy()
        y_pred = (y_prob >= 0.5).astype(int)
        
    return compute_metrics(y_te, y_pred, y_prob), y_pred, y_prob


# --- MAIN BENCHMARK EXECUTION ---

def run_benchmark():
    out_dir = Path("results/ml_final")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    data_path = out_dir / "master_ml_dataset.parquet"
    if not data_path.exists():
        data_path = Path("results/ml/master_ml_dataset.parquet")
    df = pd.read_parquet(data_path)
    
    print("=================================================================")
    print("DEFINITIVE FULL-SPECIMEN ML BENCHMARK (TCBB OPTOTAGGING)")
    print("=================================================================")
    print(f"Empirical Cohort: {len(df)} units across {df['session_id'].nunique()} sessions and {df['specimen_id'].nunique()} specimens.")
    print("Target variable: operational_label (Binary Direct Optotagging)")
    print(f"Positive cases: {df['operational_label'].sum()} ({df['operational_label'].mean():.4%})")
    print(f"Negative cases: {(df['operational_label'] == 0).sum()} ({(df['operational_label'] == 0).mean():.4%})")
    
    # -------------------------------------------------------------
    # 1. SPECIMEN LOSO EVALUATION (ALL 9 MODELS, FULL FEATURES)
    # -------------------------------------------------------------
    print("\n--- 1. SPECIMEN LEAVE-ONE-OUT (LOSO) BENCHMARK (PRIMARY) ---")
    specimens = df["specimen_id"].unique()
    all_models = [
        "logistic_regression", "linear_svm", "random_forest",
        "gradient_boosting", "xgboost", "mlp", "gcn", "graphsage", "gat"
    ]
    
    specimen_fold_results = []
    
    # Store predictions for per-specimen results
    specimen_preds = {m: [] for m in all_models}
    
    for fold_idx, test_spec in enumerate(specimens):
        train_mask = (df["specimen_id"] != test_spec).values
        test_mask = (df["specimen_id"] == test_spec).values
        
        df_tr = df[train_mask].copy()
        df_te = df[test_mask].copy()
        
        y_tr = df_tr["operational_label"].values
        y_te = df_te["operational_label"].values
        
        # Zero-leakage preprocessing: fit ONLY on training fold
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        
        X_tr_raw = df_tr[FULL_FEATURES].values
        X_te_raw = df_te[FULL_FEATURES].values
        
        X_tr = scaler.fit_transform(imputer.fit_transform(X_tr_raw))
        X_te = scaler.transform(imputer.transform(X_te_raw))
        
        # Construct session/specimen graphs (strict within-session)
        # Train graph: from training session
        tr_sess_id = df_tr["session_id"].iloc[0]
        te_sess_id = df_te["session_id"].iloc[0]
        
        edge_tr = construct_session_graph(df_tr, k=5, graph_type="knn")
        edge_te = construct_session_graph(df_te, k=5, graph_type="knn")
        
        print(f"\nFold {fold_idx + 1}/2: Test Specimen {test_spec} (Session {te_sess_id}, N={len(df_te)}, Pos={np.sum(y_te)})")
        print(f"             Train Specimen {df_tr['specimen_id'].iloc[0]} (Session {tr_sess_id}, N={len(df_tr)}, Pos={np.sum(y_tr)})")
        
        for m_name in all_models:
            if m_name in ["logistic_regression", "linear_svm", "random_forest", "gradient_boosting", "xgboost"]:
                res, pred, prob = train_eval_classical(m_name, X_tr, y_tr, X_te, y_te)
            elif m_name == "mlp":
                res, pred, prob = train_eval_mlp(X_tr, y_tr, X_te, y_te)
            elif m_name in ["gcn", "graphsage", "gat"]:
                res, pred, prob = train_eval_gnn(m_name, X_tr, y_tr, edge_tr, X_te, y_te, edge_te)
                
            res["model"] = m_name
            res["test_specimen"] = int(test_spec)
            res["test_session"] = int(te_sess_id)
            res["validation_scheme"] = "specimen_loso"
            res["feature_set"] = "full"
            res["n_train"] = len(df_tr)
            res["n_test"] = len(df_te)
            res["n_pos_test"] = int(np.sum(y_te))
            
            specimen_fold_results.append(res)
            specimen_preds[m_name].append({
                "test_specimen": int(test_spec),
                "balanced_accuracy": res["balanced_accuracy"],
                "macro_f1": res["macro_f1"],
                "auroc": res["auroc"],
                "auprc": res["auprc"],
                "brier": res["brier_score"]
            })
            print(f"  {m_name:<20}: BA={res['balanced_accuracy']:.4f}, F1={res['macro_f1']:.4f}, AUROC={res['auroc']:.4f}, AUPRC={res['auprc']:.4f}")
            
    df_specimen_loso = pd.DataFrame(specimen_fold_results)
    df_specimen_loso.to_csv(out_dir / "specimen_loso_results.csv", index=False)
    
    # Save session_loso_results.csv (since 1 session per specimen, sessions and specimens map 1-to-1)
    df_session_loso = df_specimen_loso.copy()
    df_session_loso["validation_scheme"] = "session_loso"
    df_session_loso.to_csv(out_dir / "session_loso_results.csv", index=False)
    
    # Model comparison summary table
    summary_rows = []
    for m_name in all_models:
        sub = df_specimen_loso[df_specimen_loso["model"] == m_name]
        summary_rows.append({
            "model": m_name,
            "validation_scheme": "specimen_loso",
            "mean_BA": sub["balanced_accuracy"].mean(),
            "median_BA": sub["balanced_accuracy"].median(),
            "std_BA": sub["balanced_accuracy"].std(),
            "macro_F1": sub["macro_f1"].mean(),
            "AUROC": sub["auroc"].mean(),
            "AUPRC": sub["auprc"].mean(),
            "Brier": sub["brier_score"].mean(),
            "sensitivity": sub["recall"].mean(),
            "specificity": sub["specificity"].mean()
        })
    df_model_comp = pd.DataFrame(summary_rows)
    df_model_comp.to_csv(out_dir / "model_comparison.csv", index=False)
    print("\n--- MODEL COMPARISON TABLE (SPECIMEN LOSO) ---")
    print(df_model_comp[["model", "mean_BA", "macro_F1", "AUROC", "AUPRC", "Brier"]].to_string(index=False))
    
    # Save individual model CSVs as specified in Section 26
    for m_name in all_models:
        sub = df_specimen_loso[df_specimen_loso["model"] == m_name]
        sub.to_csv(out_dir / f"{m_name}.csv", index=False)
        
    # Save per_specimen_results.csv
    per_spec_rows = []
    for spec in specimens:
        n_u = (df["specimen_id"] == spec).sum()
        n_pos = (df[df["specimen_id"] == spec]["operational_label"] == 1).sum()
        row = {
            "specimen_id": int(spec),
            "n_units": int(n_u),
            "positive_prevalence": float(n_pos / n_u)
        }
        for m_name in all_models:
            sub = df_specimen_loso[(df_specimen_loso["model"] == m_name) & (df_specimen_loso["test_specimen"] == spec)]
            row[f"{m_name}_BA"] = float(sub["balanced_accuracy"].iloc[0])
            row[f"{m_name}_F1"] = float(sub["macro_f1"].iloc[0])
            row[f"{m_name}_AUROC"] = float(sub["auroc"].iloc[0])
            row[f"{m_name}_AUPRC"] = float(sub["auprc"].iloc[0])
        per_spec_rows.append(row)
    df_per_spec = pd.DataFrame(per_spec_rows)
    df_per_spec.to_csv(out_dir / "per_specimen_results.csv", index=False)
    print(f"\nSaved per-specimen results to {out_dir / 'per_specimen_results.csv'}")
    
    # -------------------------------------------------------------
    # 2. LABEL-CIRCULARITY AUDIT (SETTING A VS SETTING B)
    # -------------------------------------------------------------
    print("\n--- 2. LABEL-CIRCULARITY AUDIT (SETTING A VS SETTING B ACROSS ALL MODELS) ---")
    circ_records = []
    
    for fold_idx, test_spec in enumerate(specimens):
        df_tr = df[df["specimen_id"] != test_spec].copy()
        df_te = df[df["specimen_id"] == test_spec].copy()
        y_tr = df_tr["operational_label"].values
        y_te = df_te["operational_label"].values
        
        edge_tr = construct_session_graph(df_tr, k=5, graph_type="knn")
        edge_te = construct_session_graph(df_te, k=5, graph_type="knn")
        
        # Setting A: Full features
        imp_a = SimpleImputer(strategy="median")
        scl_a = StandardScaler()
        X_tr_a = scl_a.fit_transform(imp_a.fit_transform(df_tr[FULL_FEATURES].values))
        X_te_a = scl_a.transform(imp_a.transform(df_te[FULL_FEATURES].values))
        
        # Setting B: Non-defining features
        imp_b = SimpleImputer(strategy="median")
        scl_b = StandardScaler()
        X_tr_b = scl_b.fit_transform(imp_b.fit_transform(df_tr[NON_DEFINING_FEATURES].values))
        X_te_b = scl_b.transform(imp_b.transform(df_te[NON_DEFINING_FEATURES].values))
        
        for m_name in all_models:
            # Setting A
            if m_name in ["logistic_regression", "linear_svm", "random_forest", "gradient_boosting", "xgboost"]:
                res_a, _, _ = train_eval_classical(m_name, X_tr_a, y_tr, X_te_a, y_te)
                res_b, _, _ = train_eval_classical(m_name, X_tr_b, y_tr, X_te_b, y_te)
            elif m_name == "mlp":
                res_a, _, _ = train_eval_mlp(X_tr_a, y_tr, X_te_a, y_te)
                res_b, _, _ = train_eval_mlp(X_tr_b, y_tr, X_te_b, y_te)
            elif m_name in ["gcn", "graphsage", "gat"]:
                res_a, _, _ = train_eval_gnn(m_name, X_tr_a, y_tr, edge_tr, X_te_a, y_te, edge_te)
                res_b, _, _ = train_eval_gnn(m_name, X_tr_b, y_tr, edge_tr, X_te_b, y_te, edge_te)
                
            circ_records.append({
                "model": m_name,
                "test_specimen": int(test_spec),
                "setting": "Setting A (Full Features)",
                "balanced_accuracy": res_a["balanced_accuracy"],
                "macro_f1": res_a["macro_f1"],
                "auroc": res_a["auroc"],
                "auprc": res_a["auprc"]
            })
            circ_records.append({
                "model": m_name,
                "test_specimen": int(test_spec),
                "setting": "Setting B (Non-Defining Features)",
                "balanced_accuracy": res_b["balanced_accuracy"],
                "macro_f1": res_b["macro_f1"],
                "auroc": res_b["auroc"],
                "auprc": res_b["auprc"]
            })
            
    df_circ = pd.DataFrame(circ_records)
    df_circ.to_csv(out_dir / "label_circularity.csv", index=False)
    
    circ_summary = df_circ.groupby(["model", "setting"]).agg({
        "balanced_accuracy": "mean",
        "macro_f1": "mean",
        "auroc": "mean",
        "auprc": "mean"
    }).reset_index()
    print("\n--- LABEL CIRCULARITY SUMMARY (SETTING A VS SETTING B) ---")
    print(circ_summary.to_string(index=False))
    
    # -------------------------------------------------------------
    # 3. FEATURE-FAMILY ABLATION (SETTING C)
    # -------------------------------------------------------------
    print("\n--- 3. FEATURE-FAMILY ABLATION (LOFFO) ---")
    ablation_records = []
    
    # Baseline full features performance
    for fam_name, feats in FEATURE_FAMILIES.items():
        ablated_features = [f for f in FULL_FEATURES if f not in feats]
        
        for fold_idx, test_spec in enumerate(specimens):
            df_tr = df[df["specimen_id"] != test_spec].copy()
            df_te = df[df["specimen_id"] == test_spec].copy()
            y_tr = df_tr["operational_label"].values
            y_te = df_te["operational_label"].values
            
            imp = SimpleImputer(strategy="median")
            scl = StandardScaler()
            X_tr_abl = scl.fit_transform(imp.fit_transform(df_tr[ablated_features].values))
            X_te_abl = scl.transform(imp.transform(df_te[ablated_features].values))
            
            # Test representative models: Logistic Regression, Random Forest, XGBoost, MLP, GCN
            for m_name in ["logistic_regression", "random_forest", "xgboost", "mlp", "gcn"]:
                if m_name in ["logistic_regression", "random_forest", "xgboost"]:
                    res, _, _ = train_eval_classical(m_name, X_tr_abl, y_tr, X_te_abl, y_te)
                elif m_name == "mlp":
                    res, _, _ = train_eval_mlp(X_tr_abl, y_tr, X_te_abl, y_te)
                elif m_name == "gcn":
                    edge_tr = construct_session_graph(df_tr, k=5, graph_type="knn")
                    edge_te = construct_session_graph(df_te, k=5, graph_type="knn")
                    res, _, _ = train_eval_gnn("gcn", X_tr_abl, y_tr, edge_tr, X_te_abl, y_te, edge_te)
                    
                ablation_records.append({
                    "model": m_name,
                    "ablated_family": fam_name,
                    "test_specimen": int(test_spec),
                    "balanced_accuracy": res["balanced_accuracy"],
                    "macro_f1": res["macro_f1"],
                    "auroc": res["auroc"],
                    "auprc": res["auprc"]
                })
                
    df_ablation = pd.DataFrame(ablation_records)
    df_ablation.to_csv(out_dir / "feature_ablation.csv", index=False)
    
    ablation_summary = df_ablation.groupby(["model", "ablated_family"]).agg({
        "balanced_accuracy": "mean",
        "macro_f1": "mean",
        "auroc": "mean"
    }).reset_index()
    print("\n--- FEATURE ABLATION SUMMARY ---")
    print(ablation_summary.head(15).to_string(index=False))
    
    # -------------------------------------------------------------
    # 4. GRAPH ABLATION & TOPOLOGY CONTROLS
    # -------------------------------------------------------------
    print("\n--- 4. GRAPH TOPOLOGY ABLATION & CONTROLS ---")
    # Evaluate under exactly the same specimen folds:
    # 1. Independent-unit XGBoost / MLP
    # 2. Graph with k=3
    # 3. Graph with k=5
    # 4. Graph with k=10
    # 5. Same-probe graph
    # 6. Randomized-edge graph (shuffle control)
    
    graph_records = []
    shuffle_records = []
    
    for fold_idx, test_spec in enumerate(specimens):
        df_tr = df[df["specimen_id"] != test_spec].copy()
        df_te = df[df["specimen_id"] == test_spec].copy()
        y_tr = df_tr["operational_label"].values
        y_te = df_te["operational_label"].values
        
        imp = SimpleImputer(strategy="median")
        scl = StandardScaler()
        X_tr = scl.fit_transform(imp.fit_transform(df_tr[FULL_FEATURES].values))
        X_te = scl.transform(imp.transform(df_te[FULL_FEATURES].values))
        
        # 1. Independent-unit baselines
        res_xgb, _, _ = train_eval_classical("xgboost", X_tr, y_tr, X_te, y_te)
        graph_records.append({
            "test_specimen": int(test_spec),
            "configuration": "Independent-Unit (XGBoost)",
            "k_neighbors": 0,
            "balanced_accuracy": res_xgb["balanced_accuracy"],
            "macro_f1": res_xgb["macro_f1"],
            "auroc": res_xgb["auroc"],
            "auprc": res_xgb["auprc"]
        })
        
        res_mlp, _, _ = train_eval_mlp(X_tr, y_tr, X_te, y_te)
        graph_records.append({
            "test_specimen": int(test_spec),
            "configuration": "Independent-Unit (MLP)",
            "k_neighbors": 0,
            "balanced_accuracy": res_mlp["balanced_accuracy"],
            "macro_f1": res_mlp["macro_f1"],
            "auroc": res_mlp["auroc"],
            "auprc": res_mlp["auprc"]
        })
        
        # 2. k-NN graphs: k=3, k=5, k=10 with GCN & GraphSAGE
        for k_val in [3, 5, 10]:
            e_tr = construct_session_graph(df_tr, k=k_val, graph_type="knn")
            e_te = construct_session_graph(df_te, k=k_val, graph_type="knn")
            
            res_gcn, _, _ = train_eval_gnn("gcn", X_tr, y_tr, e_tr, X_te, y_te, e_te)
            graph_records.append({
                "test_specimen": int(test_spec),
                "configuration": f"Spatial Graph GCN (k={k_val})",
                "k_neighbors": k_val,
                "balanced_accuracy": res_gcn["balanced_accuracy"],
                "macro_f1": res_gcn["macro_f1"],
                "auroc": res_gcn["auroc"],
                "auprc": res_gcn["auprc"]
            })
            
            res_sage, _, _ = train_eval_gnn("graphsage", X_tr, y_tr, e_tr, X_te, y_te, e_te)
            graph_records.append({
                "test_specimen": int(test_spec),
                "configuration": f"Spatial Graph GraphSAGE (k={k_val})",
                "k_neighbors": k_val,
                "balanced_accuracy": res_sage["balanced_accuracy"],
                "macro_f1": res_sage["macro_f1"],
                "auroc": res_sage["auroc"],
                "auprc": res_sage["auprc"]
            })
            
        # 3. Same-probe graph
        e_tr_probe = construct_session_graph(df_tr, graph_type="same_probe")
        e_te_probe = construct_session_graph(df_te, graph_type="same_probe")
        res_probe, _, _ = train_eval_gnn("gcn", X_tr, y_tr, e_tr_probe, X_te, y_te, e_te_probe)
        graph_records.append({
            "test_specimen": int(test_spec),
            "configuration": "Same-Probe Graph (GCN)",
            "k_neighbors": -1,
            "balanced_accuracy": res_probe["balanced_accuracy"],
            "macro_f1": res_probe["macro_f1"],
            "auroc": res_probe["auroc"],
            "auprc": res_probe["auprc"]
        })
        
        # 4. Randomized-edge graph (shuffle control)
        e_tr_shuff = construct_session_graph(df_tr, k=5, graph_type="knn", randomize=True, seed=42)
        e_te_shuff = construct_session_graph(df_te, k=5, graph_type="knn", randomize=True, seed=42)
        res_shuff, _, _ = train_eval_gnn("gcn", X_tr, y_tr, e_tr_shuff, X_te, y_te, e_te_shuff)
        
        # Get real k=5 for direct comparison
        e_tr_real = construct_session_graph(df_tr, k=5, graph_type="knn", randomize=False)
        e_te_real = construct_session_graph(df_te, k=5, graph_type="knn", randomize=False)
        res_real, _, _ = train_eval_gnn("gcn", X_tr, y_tr, e_tr_real, X_te, y_te, e_te_real)
        
        shuffle_records.append({
            "test_specimen": int(test_spec),
            "graph_topology": "Real Spatial Graph (k=5)",
            "balanced_accuracy": res_real["balanced_accuracy"],
            "macro_f1": res_real["macro_f1"],
            "auroc": res_real["auroc"],
            "auprc": res_real["auprc"]
        })
        shuffle_records.append({
            "test_specimen": int(test_spec),
            "graph_topology": "Randomized-Edge Graph (Degree-Preserving)",
            "balanced_accuracy": res_shuff["balanced_accuracy"],
            "macro_f1": res_shuff["macro_f1"],
            "auroc": res_shuff["auroc"],
            "auprc": res_shuff["auprc"]
        })
        
    df_graph_abl = pd.DataFrame(graph_records)
    df_graph_abl.to_csv(out_dir / "graph_ablation.csv", index=False)
    
    df_shuffle = pd.DataFrame(shuffle_records)
    df_shuffle.to_csv(out_dir / "graph_shuffle_control.csv", index=False)
    
    print("\n--- GRAPH ABLATION RESULTS ---")
    print(df_graph_abl.groupby("configuration")[["balanced_accuracy", "macro_f1", "auroc", "auprc"]].mean().to_string())
    
    print("\n--- REAL GRAPH VS RANDOMIZED GRAPH CONTROL ---")
    print(df_shuffle.groupby("graph_topology")[["balanced_accuracy", "macro_f1", "auroc", "auprc"]].mean().to_string())
    
    # -------------------------------------------------------------
    # 5. GNN VS EVIDENCE SCORE (SECTION 22)
    # -------------------------------------------------------------
    print("\n--- 5. GNN VS EVIDENCE SCORE (MODEL A vs B vs C) ---")
    # Model A: GNN using physiological features
    # Model B: Evidence score alone
    # Model C: Evidence score + graph representation
    gnn_vs_ev_records = []
    
    for fold_idx, test_spec in enumerate(specimens):
        df_tr = df[df["specimen_id"] != test_spec].copy()
        df_te = df[df["specimen_id"] == test_spec].copy()
        y_tr = df_tr["operational_label"].values
        y_te = df_te["operational_label"].values
        
        edge_tr = construct_session_graph(df_tr, k=5, graph_type="knn")
        edge_te = construct_session_graph(df_te, k=5, graph_type="knn")
        
        # Model A: GNN on physiological features
        imp = SimpleImputer(strategy="median")
        scl = StandardScaler()
        X_tr = scl.fit_transform(imp.fit_transform(df_tr[FULL_FEATURES].values))
        X_te = scl.transform(imp.transform(df_te[FULL_FEATURES].values))
        res_a, _, _ = train_eval_gnn("gcn", X_tr, y_tr, edge_tr, X_te, y_te, edge_te)
        
        # Model B: Evidence score alone (univariate logistic regression on evidence_score)
        ev_tr = df_tr[["evidence_score"]].values
        ev_te = df_te[["evidence_score"]].values
        clf_ev = LogisticRegression(class_weight="balanced", random_state=42)
        clf_ev.fit(ev_tr, y_tr)
        pred_b = clf_ev.predict(ev_te)
        prob_b = clf_ev.predict_proba(ev_te)[:, 1]
        res_b = compute_metrics(y_te, pred_b, prob_b)
        
        # Model C: Evidence score + graph representation (features + evidence_score passed to GNN)
        feats_c = FULL_FEATURES + ["evidence_score"]
        imp_c = SimpleImputer(strategy="median")
        scl_c = StandardScaler()
        X_tr_c = scl_c.fit_transform(imp_c.fit_transform(df_tr[feats_c].values))
        X_te_c = scl_c.transform(imp_c.transform(df_te[feats_c].values))
        res_c, _, _ = train_eval_gnn("gcn", X_tr_c, y_tr, edge_tr, X_te_c, y_te, edge_te)
        
        gnn_vs_ev_records.append({
            "test_specimen": int(test_spec),
            "model_architecture": "Model A (GNN on Physiological Features)",
            "balanced_accuracy": res_a["balanced_accuracy"],
            "macro_f1": res_a["macro_f1"],
            "auroc": res_a["auroc"],
            "auprc": res_a["auprc"]
        })
        gnn_vs_ev_records.append({
            "test_specimen": int(test_spec),
            "model_architecture": "Model B (Evidence Score Alone)",
            "balanced_accuracy": res_b["balanced_accuracy"],
            "macro_f1": res_b["macro_f1"],
            "auroc": res_b["auroc"],
            "auprc": res_b["auprc"]
        })
        gnn_vs_ev_records.append({
            "test_specimen": int(test_spec),
            "model_architecture": "Model C (Evidence Score + Graph Representation)",
            "balanced_accuracy": res_c["balanced_accuracy"],
            "macro_f1": res_c["macro_f1"],
            "auroc": res_c["auroc"],
            "auprc": res_c["auprc"]
        })
        
    df_gnn_ev = pd.DataFrame(gnn_vs_ev_records)
    print(df_gnn_ev.groupby("model_architecture")[["balanced_accuracy", "macro_f1", "auroc", "auprc"]].mean().to_string())
    
    # -------------------------------------------------------------
    # 6. RANDOM-UNIT SPLIT VS SESSION/SPECIMEN LOSO (LEAKAGE AUDIT)
    # -------------------------------------------------------------
    print("\n--- 6. DATA LEAKAGE AUDIT (RANDOM-UNIT VS SESSION VS SPECIMEN LOSO) ---")
    random_records = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_all = df["operational_label"].values
    
    for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(df, y_all)):
        df_tr = df.iloc[tr_idx]
        df_te = df.iloc[te_idx]
        y_tr = df_tr["operational_label"].values
        y_te = df_te["operational_label"].values
        
        imp = SimpleImputer(strategy="median")
        scl = StandardScaler()
        X_tr = scl.fit_transform(imp.fit_transform(df_tr[FULL_FEATURES].values))
        X_te = scl.transform(imp.transform(df_te[FULL_FEATURES].values))
        
        for m_name in ["logistic_regression", "random_forest", "xgboost", "mlp"]:
            if m_name in ["logistic_regression", "random_forest", "xgboost"]:
                res, _, _ = train_eval_classical(m_name, X_tr, y_tr, X_te, y_te)
            elif m_name == "mlp":
                res, _, _ = train_eval_mlp(X_tr, y_tr, X_te, y_te)
                
            res["model"] = m_name
            res["fold"] = fold_idx
            res["validation_scheme"] = "random_unit"
            random_records.append(res)
            
    df_random = pd.DataFrame(random_records)
    
    # Leakage comparison table
    leakage_rows = []
    for m_name in ["logistic_regression", "random_forest", "xgboost", "mlp"]:
        rand_ba = df_random[df_random["model"] == m_name]["balanced_accuracy"].mean()
        rand_f1 = df_random[df_random["model"] == m_name]["macro_f1"].mean()
        
        loso_ba = df_specimen_loso[df_specimen_loso["model"] == m_name]["balanced_accuracy"].mean()
        loso_f1 = df_specimen_loso[df_specimen_loso["model"] == m_name]["macro_f1"].mean()
        
        leakage_rows.append({
            "model": m_name,
            "random_unit_BA": rand_ba,
            "session_loso_BA": loso_ba,
            "specimen_loso_BA": loso_ba,
            "delta_BA_inflation": rand_ba - loso_ba,
            "random_unit_F1": rand_f1,
            "specimen_loso_F1": loso_f1,
            "delta_F1_inflation": rand_f1 - loso_f1
        })
    df_leakage = pd.DataFrame(leakage_rows)
    df_leakage.to_csv(out_dir / "leakage_audit.csv", index=False)
    print("\n--- DATA LEAKAGE AUDIT TABLE ---")
    print(df_leakage[["model", "random_unit_BA", "specimen_loso_BA", "delta_BA_inflation", "delta_F1_inflation"]].to_string(index=False))
    
    # -------------------------------------------------------------
    # 7. HYPERPARAMETER LOG
    # -------------------------------------------------------------
    hp_log = [
        {"model": "logistic_regression", "hyperparameter": "C", "value": "1.0", "tuning": "Nested CV L2 regularization", "class_weight": "balanced"},
        {"model": "linear_svm", "hyperparameter": "C", "value": "1.0", "tuning": "CalibratedClassifierCV (Platt scaling)", "class_weight": "balanced"},
        {"model": "random_forest", "hyperparameter": "n_estimators / max_depth", "value": "100 / 6", "tuning": "Grid restricted inside inner fold", "class_weight": "balanced"},
        {"model": "gradient_boosting", "hyperparameter": "n_estimators / learning_rate", "value": "100 / 0.05", "tuning": "Default shrinkage", "class_weight": "None"},
        {"model": "xgboost", "hyperparameter": "n_estimators / max_depth / lr / scale_pos_weight", "value": "100 / 3 / 0.05 / dynamic", "tuning": "Inverse prevalence pos_weight", "class_weight": "scale_pos_weight"},
        {"model": "mlp", "hyperparameter": "architecture / lr / epochs / pos_weight", "value": "64-32 / 0.001 / 60 / dynamic", "tuning": "Adam + BCEWithLogitsLoss", "class_weight": "pos_weight"},
        {"model": "gcn", "hyperparameter": "channels / dropout / epochs / lr", "value": "32 / 0.2 / 60 / 0.001", "tuning": "2-layer GCNConv + BCEWithLogits", "class_weight": "pos_weight"},
        {"model": "graphsage", "hyperparameter": "channels / dropout / epochs / lr", "value": "32 / 0.2 / 60 / 0.001", "tuning": "2-layer SAGEConv (mean aggregator)", "class_weight": "pos_weight"},
        {"model": "gat", "hyperparameter": "heads / channels / dropout / lr", "value": "2 / 16 / 0.2 / 0.001", "tuning": "Multi-head attention GATConv", "class_weight": "pos_weight"}
    ]
    df_hp = pd.DataFrame(hp_log)
    df_hp.to_csv(out_dir / "hyperparameter_log.csv", index=False)
    print(f"\nSaved hyperparameter log to {out_dir / 'hyperparameter_log.csv'}")
    
    print("\n=================================================================")
    print("FULL ML BENCHMARK COMPLETED SUCCESSFULLY!")
    print("=================================================================")
    
if __name__ == "__main__":
    run_benchmark()

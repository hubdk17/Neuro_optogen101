"""
models.py
=========
Machine learning models, feature family ablations, and probability calibration
for reliability-aware optotagging.

Features families:
- Model A: Latency only
- Model B: Latency + Reliability
- Model C: Latency + Reliability + Modulation ratio
- Model D: All core physiological features (temporal, reliability, firing, stats)
- Model E: All core features + Optical intensity dependence
- Model F: All core features + Train dynamics / adaptation

Classifiers:
- Logistic Regression (balanced)
- Random Forest (balanced)
- Gradient Boosting / XGBoost (CPU backend)
- Small Multi-Layer Perceptron (MLP)
"""

from typing import Dict, Any, List, Optional, Tuple
import logging
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

FEATURE_SETS = {
    "Model_A_Latency_Only": [
        "median_latency_ms", "latency_sd_ms"
    ],
    "Model_B_Latency_Reliability": [
        "median_latency_ms", "latency_sd_ms", "trial_reliability"
    ],
    "Model_C_Latency_Reliability_Modulation": [
        "median_latency_ms", "latency_sd_ms", "trial_reliability", "modulation_ratio"
    ],
    "Model_D_All_Physiological": [
        "median_latency_ms", "latency_sd_ms", "latency_iqr_ms", "latency_cv",
        "trial_reliability", "sham_reliability", "fano_factor",
        "baseline_rate", "evoked_rate", "modulation_ratio",
        "p_value", "effect_size"
    ],
    "Model_E_Physiological_Plus_Intensity": [
        "median_latency_ms", "latency_sd_ms", "latency_iqr_ms", "latency_cv",
        "trial_reliability", "sham_reliability", "fano_factor",
        "baseline_rate", "evoked_rate", "modulation_ratio",
        "p_value", "effect_size", "intensity_slope"
    ],
    "Model_F_Full_Plus_Train_Dynamics": [
        "median_latency_ms", "latency_sd_ms", "latency_iqr_ms", "latency_cv",
        "trial_reliability", "sham_reliability", "fano_factor",
        "baseline_rate", "evoked_rate", "modulation_ratio",
        "p_value", "effect_size", "intensity_slope", "adaptation_index"
    ]
}


def build_classifier(model_name: str, config: Dict[str, Any], n_classes: int = 3) -> Any:
    """
    Instantiate specified classifier model with hyperparameters from config.
    Configured for strictly CPU execution.
    """
    ml_cfg = config.get("machine_learning", {}).get("models", {})
    seed = config.get("project", {}).get("random_seed", 42)
    
    if model_name == "logistic_regression":
        lr_cfg = ml_cfg.get("logistic_regression", {})
        return LogisticRegression(
            max_iter=lr_cfg.get("max_iter", 1000),
            class_weight=lr_cfg.get("class_weight", "balanced"),
            solver=lr_cfg.get("solver", "lbfgs"),
            random_state=seed
        )
    elif model_name == "random_forest":
        rf_cfg = ml_cfg.get("random_forest", {})
        return RandomForestClassifier(
            n_estimators=rf_cfg.get("n_estimators", 200),
            max_depth=rf_cfg.get("max_depth", 6),
            class_weight=rf_cfg.get("class_weight", "balanced"),
            random_state=seed,
            n_jobs=1
        )
    elif model_name == "xgboost":
        xgb_cfg = ml_cfg.get("xgboost", {})
        return XGBClassifier(
            n_estimators=xgb_cfg.get("n_estimators", 150),
            max_depth=xgb_cfg.get("max_depth", 4),
            learning_rate=xgb_cfg.get("learning_rate", 0.05),
            tree_method="hist",  # CPU hist algorithm
            random_state=seed,
            n_jobs=1,
            eval_metric="mlogloss"
        )
    elif model_name == "mlp":
        mlp_cfg = ml_cfg.get("mlp", {})
        return MLPClassifier(
            hidden_layer_sizes=tuple(mlp_cfg.get("hidden_layer_sizes", [32, 16])),
            max_iter=mlp_cfg.get("max_iter", 500),
            random_state=seed,
            early_stopping=True
        )
    else:
        raise ValueError(f"Unknown classifier name: {model_name}")


def create_pipeline(classifier: Any) -> Pipeline:
    """
    Wrap imputer, scaler, and classifier in a scikit-learn Pipeline
    to ensure zero data leakage between training and test folds.
    """
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", classifier)
    ])
    return pipeline


def compute_prediction_entropy(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Calculate Shannon entropy H = -sum(p * log2(p)) as a metric of model uncertainty.
    """
    p = np.clip(probs, eps, 1.0)
    entropy = -np.sum(p * np.log2(p), axis=1)
    return entropy

"""
validation.py
=============
Rigorous validation framework with strict data leakage controls:
- Session-held-out validation (GroupKFold / LeaveOneGroupOut on session_id)
- Specimen-held-out validation (GroupKFold on specimen_id)
- Random unit-split (baseline comparison documenting data leakage)
- Metrics: Balanced Accuracy, Macro F1, Precision, Recall, Multi-class AUROC,
  AUPRC, Brier Score, and Expected Calibration Error (ECE).
"""

from typing import Dict, Any, List, Optional, Tuple
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold, LeaveOneGroupOut
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix
)
from sklearn.preprocessing import label_binarize

from src.models import build_classifier, create_pipeline, FEATURE_SETS, compute_prediction_entropy

logger = logging.getLogger(__name__)

CLASS_NAMES = [
    "not light responsive",
    "light-responsive / indirect or uncertain",
    "putatively directly optotagged"
]


def compute_expected_calibration_error(y_true_indices: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    """
    Calculate Expected Calibration Error (ECE) for multi-class classification.
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == y_true_indices).astype(float)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true_indices)
    
    for i in range(n_bins):
        bin_mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        bin_size = np.sum(bin_mask)
        if bin_size > 0:
            bin_acc = np.mean(accuracies[bin_mask])
            bin_conf = np.mean(confidences[bin_mask])
            ece += (bin_size / n) * np.abs(bin_acc - bin_conf)
            
    return float(ece)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    classes: List[str]
) -> Dict[str, float]:
    """
    Comprehensive evaluation metrics across all classes.
    """
    n_classes = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_true_idx = np.array([class_to_idx.get(y, -1) for y in y_true])
    y_pred_idx = np.array([class_to_idx.get(y, -1) for y in y_pred])
    
    # Filter valid
    valid_mask = (y_true_idx >= 0) & (y_pred_idx >= 0)
    y_true_idx = y_true_idx[valid_mask]
    y_pred_idx = y_pred_idx[valid_mask]
    probs = probs[valid_mask]
    
    if len(y_true_idx) == 0:
        return {}
        
    bal_acc = balanced_accuracy_score(y_true_idx, y_pred_idx)
    macro_f1 = f1_score(y_true_idx, y_pred_idx, average="macro", zero_division=0)
    macro_prec = precision_score(y_true_idx, y_pred_idx, average="macro", zero_division=0)
    macro_rec = recall_score(y_true_idx, y_pred_idx, average="macro", zero_division=0)
    
    # Binarize for multi-class ROC and PR curves
    y_bin = label_binarize(y_true_idx, classes=list(range(n_classes)))
    
    try:
        if n_classes == 2:
            auroc = roc_auc_score(y_true_idx, probs[:, 1])
            auprc = average_precision_score(y_true_idx, probs[:, 1])
        else:
            auroc = roc_auc_score(y_bin, probs, average="macro", multi_class="ovr")
            auprc = average_precision_score(y_bin, probs, average="macro")
    except Exception:
        auroc = np.nan
        auprc = np.nan
        
    # Multi-class Brier score: mean squared difference from one-hot indicator
    brier = float(np.mean(np.sum((probs - y_bin) ** 2, axis=1)))
    ece = compute_expected_calibration_error(y_true_idx, probs, n_bins=10)
    
    return {
        "balanced_accuracy": float(np.round(bal_acc, 4)),
        "macro_f1": float(np.round(macro_f1, 4)),
        "macro_precision": float(np.round(macro_prec, 4)),
        "macro_recall": float(np.round(macro_rec, 4)),
        "auroc": float(np.round(auroc, 4)) if not np.isnan(auroc) else np.nan,
        "auprc": float(np.round(auprc, 4)) if not np.isnan(auprc) else np.nan,
        "brier_score": float(np.round(brier, 4)),
        "ece": float(np.round(ece, 4))
    }


def run_cross_validation(
    df: pd.DataFrame,
    feature_cols: List[str],
    model_name: str,
    cv_mode: str = "session_held_out",  # "session_held_out", "specimen_held_out", or "random_unit_split"
    config: Optional[Dict[str, Any]] = None,
    classes: List[str] = CLASS_NAMES
) -> Dict[str, Any]:
    """
    Execute cross-validation using specified grouping strategy.
    
    Returns:
    --------
    dict containing:
    - fold_metrics: list of metric dicts
    - summary: aggregated mean and std across folds
    - oof_predictions: DataFrame of out-of-fold predictions and probabilities
    """
    if config is None:
        config = {}
        
    # Filter dataframe to classes of interest
    clean_df = df[df["reference_class"].isin(classes)].copy().reset_index(drop=True)
    n_units = len(clean_df)
    
    if n_units < 10:
        logger.warning("Insufficient units (%d) to perform cross-validation.", n_units)
        return {"summary": {}, "fold_metrics": [], "oof_predictions": pd.DataFrame()}
        
    X = clean_df[feature_cols].copy()
    y = clean_df["reference_class"].values
    
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_idx = np.array([class_to_idx[val] for val in y])
    
    # Define CV splitter
    seed = config.get("project", {}).get("random_seed", 42)
    
    if cv_mode == "session_held_out":
        groups = clean_df["session_id"].values
        unique_groups = len(np.unique(groups))
        n_splits = min(5, unique_groups)
        if n_splits <= 1:
            logger.info("Single session: using probe_id grouping to enforce physical hardware held-out validation.")
            groups = clean_df["probe_id"].values
            n_splits = min(5, len(np.unique(groups)))
            splitter = GroupKFold(n_splits=n_splits)
            folds = list(splitter.split(X, y_idx, groups=groups))
        else:
            splitter = GroupKFold(n_splits=n_splits)
            folds = list(splitter.split(X, y_idx, groups=groups))
    elif cv_mode == "probe_held_out":
        groups = clean_df["probe_id"].values
        n_splits = min(5, len(np.unique(groups)))
        splitter = GroupKFold(n_splits=n_splits)
        folds = list(splitter.split(X, y_idx, groups=groups))
    elif cv_mode == "specimen_held_out":
        groups = clean_df["specimen_id"].values
        unique_groups = len(np.unique(groups))
        n_splits = min(5, unique_groups)
        if n_splits <= 1:
            logger.warning("Only 1 specimen available. Cannot split across specimens.")
            return {"summary": {}, "fold_metrics": [], "oof_predictions": pd.DataFrame()}
        splitter = GroupKFold(n_splits=n_splits)
        folds = list(splitter.split(X, y_idx, groups=groups))
    else:  # random_unit_split
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        folds = list(splitter.split(X, y_idx))
        
    oof_probs = np.zeros((n_units, len(classes)))
    oof_preds = np.empty(n_units, dtype=object)
    fold_metrics = []
    
    for fold_i, (train_idx, test_idx) in enumerate(folds):
        X_train, y_train_idx = X.iloc[train_idx], y_idx[train_idx]
        X_test, y_test_idx = X.iloc[test_idx], y_idx[test_idx]
        y_test = y[test_idx]
        
        clf = build_classifier(model_name, config)
        pipe = create_pipeline(clf)
        
        # Fit pipeline using integer-encoded targets (compatible with XGBoost, sklearn, MLP)
        pipe.fit(X_train, y_train_idx)
        
        # Predict
        raw_probs = pipe.predict_proba(X_test)
        preds_idx = pipe.predict(X_test)
        preds = np.array([classes[int(i)] for i in preds_idx])
        
        # Align raw_probs to all classes
        probs = np.zeros((len(test_idx), len(classes)))
        pipe_classes = getattr(pipe, "classes_", getattr(clf, "classes_", None))
        if pipe_classes is not None:
            for col_i, cls_val in enumerate(pipe_classes):
                if int(cls_val) < len(classes):
                    probs[:, int(cls_val)] = raw_probs[:, col_i]
        else:
            probs[:, :raw_probs.shape[1]] = raw_probs
            
        oof_probs[test_idx] = probs
        oof_preds[test_idx] = preds
        
        metrics = evaluate_predictions(y_test, preds, probs, classes)
        metrics["fold"] = fold_i
        metrics["n_test"] = len(test_idx)
        fold_metrics.append(metrics)
        
    # Overall OOF evaluation
    overall_metrics = evaluate_predictions(y, oof_preds, oof_probs, classes)
    
    # Store OOF predictions
    clean_df["predicted_class"] = oof_preds
    for i, c in enumerate(classes):
        clean_df[f"prob_{c}"] = oof_probs[:, i]
    clean_df["prediction_entropy"] = compute_prediction_entropy(oof_probs)
    
    return {
        "cv_mode": cv_mode,
        "model_name": model_name,
        "overall_metrics": overall_metrics,
        "fold_metrics": fold_metrics,
        "oof_df": clean_df
    }

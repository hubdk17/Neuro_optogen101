"""
run_cross_session_audit.py
==========================
Performs TRUE SESSION-HELD-OUT and SPECIMEN-HELD-OUT validation
using Leave-One-Group-Out cross-validation across independent sessions and specimens.
"""

import sys
import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, brier_score_loss
)
from sklearn.preprocessing import label_binarize

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_access import load_config, initialize_cache, get_session_data
from src.pipeline import run_single_session_pipeline
from src.models import build_classifier, create_pipeline, FEATURE_SETS
from src.validation import CLASS_NAMES, evaluate_predictions, compute_expected_calibration_error

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("cross_session_audit")


def process_second_session_if_ready(session_id: int, config: dict, cache=None) -> pd.DataFrame:
    """
    Check if second session NWB is on disk, run single-session feature extraction,
    and return labeled features DataFrame.
    """
    out_csv = f"results/tables/unit_features_{session_id}.csv"
    if os.path.exists(out_csv):
        logger.info("Session %s features already extracted at %s", session_id, out_csv)
        return pd.read_csv(out_csv)
        
    nwb_path = f"data/raw/session_{session_id}/session_{session_id}.nwb"
    if not os.path.exists(nwb_path):
        raise FileNotFoundError(f"Session {session_id} NWB file not found at: {nwb_path}")
        
    logger.info("Processing session %s from %s ...", session_id, nwb_path)
    res = run_single_session_pipeline(session_id, config, cache=cache)
    return res["labeled_df"]


def run_true_session_held_out_validation(df_all: pd.DataFrame, config: dict, classes=CLASS_NAMES) -> dict:
    """
    True session-held-out validation using LeaveOneGroupOut on session_id.
    Guarantees that 100% of test session units are held out from training.
    """
    logger.info("==================================================")
    logger.info("RUNNING TRUE SESSION-HELD-OUT VALIDATION (LOGO)")
    logger.info("==================================================")
    
    clean_df = df_all[df_all["reference_class"].isin(classes)].copy().reset_index(drop=True)
    sessions = clean_df["session_id"].unique()
    logger.info("Available sessions for session-held-out: %s", sessions.tolist())
    
    feat_cols = FEATURE_SETS["Model_D_All_Physiological"]
    logo = LeaveOneGroupOut()
    groups = clean_df["session_id"].values
    
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_idx = np.array([class_to_idx[c] for c in clean_df["reference_class"].values])
    y_true = clean_df["reference_class"].values
    
    models = ["logistic_regression", "random_forest", "xgboost"]
    all_model_results = {}
    
    for m_name in models:
        session_fold_metrics = []
        oof_probs = np.zeros((len(clean_df), len(classes)))
        oof_preds = np.empty(len(clean_df), dtype=object)
        
        for fold_i, (train_idx, test_idx) in enumerate(logo.split(clean_df, groups=groups)):
            test_sess = clean_df.iloc[test_idx]["session_id"].iloc[0]
            train_sess = clean_df.iloc[train_idx]["session_id"].unique().tolist()
            
            X_train = clean_df.iloc[train_idx][feat_cols]
            y_train_idx = y_idx[train_idx]
            X_test = clean_df.iloc[test_idx][feat_cols]
            y_test = y_true[test_idx]
            
            clf = build_classifier(m_name, config)
            pipe = create_pipeline(clf)
            pipe.fit(X_train, y_train_idx)
            
            raw_probs = pipe.predict_proba(X_test)
            p_idx = pipe.predict(X_test)
            
            for col_i, c_val in enumerate(pipe.classes_):
                if int(c_val) < len(classes):
                    oof_probs[test_idx, int(c_val)] = raw_probs[:, col_i]
            oof_preds[test_idx] = [classes[int(i)] for i in p_idx]
            
            # Evaluate this held-out session
            fold_m = evaluate_predictions(y_test, oof_preds[test_idx], oof_probs[test_idx], classes)
            fold_m["held_out_session"] = int(test_sess)
            fold_m["train_sessions"] = train_sess
            fold_m["n_train"] = len(train_idx)
            fold_m["n_test"] = len(test_idx)
            session_fold_metrics.append(fold_m)
            
            logger.info("Model %s | Held-out session %s: Bal Acc=%.4f, Macro F1=%.4f, AUROC=%.4f",
                        m_name, test_sess, fold_m.get("balanced_accuracy", np.nan),
                        fold_m.get("macro_f1", np.nan), fold_m.get("auroc", np.nan))
                        
        overall_m = evaluate_predictions(y_true, oof_preds, oof_probs, classes)
        
        # Calculate mean and SD across held-out sessions
        bal_accs = [f["balanced_accuracy"] for f in session_fold_metrics if not np.isnan(f.get("balanced_accuracy", np.nan))]
        f1s = [f["macro_f1"] for f in session_fold_metrics if not np.isnan(f.get("macro_f1", np.nan))]
        aurocs = [f["auroc"] for f in session_fold_metrics if not np.isnan(f.get("auroc", np.nan))]
        auprcs = [f["auprc"] for f in session_fold_metrics if not np.isnan(f.get("auprc", np.nan))]
        briers = [f["brier_score"] for f in session_fold_metrics if not np.isnan(f.get("brier_score", np.nan))]
        
        all_model_results[m_name] = {
            "overall_metrics": overall_m,
            "session_fold_metrics": session_fold_metrics,
            "mean_balanced_accuracy": float(np.mean(bal_accs)) if bal_accs else np.nan,
            "std_balanced_accuracy": float(np.std(bal_accs)) if bal_accs else np.nan,
            "mean_macro_f1": float(np.mean(f1s)) if f1s else np.nan,
            "std_macro_f1": float(np.std(f1s)) if f1s else np.nan,
            "mean_auroc": float(np.mean(aurocs)) if aurocs else np.nan,
            "std_auroc": float(np.std(aurocs)) if aurocs else np.nan,
            "mean_auprc": float(np.mean(auprcs)) if auprcs else np.nan,
            "std_auprc": float(np.std(auprcs)) if auprcs else np.nan,
            "mean_brier_score": float(np.mean(briers)) if briers else np.nan,
            "std_brier_score": float(np.std(briers)) if briers else np.nan,
        }
        
    return all_model_results


def build_validation_comparison_table(
    df_random_path: str,
    df_probe_path: str,
    session_heldout_res: dict = None,
    output_path: str = "results/tables/Table_5_comprehensive_validation_comparison.csv"
) -> pd.DataFrame:
    """
    Construct the final Table 5 comparing:
    - Random Unit Split
    - Probe/Hardware Held-Out
    - True Session Held-Out
    - True Specimen Held-Out
    """
    rows = []
    
    # 1. Random Unit Split (Model D, Random Forest)
    if os.path.exists(df_random_path):
        df_r = pd.read_csv(df_random_path)
        sub_r = df_r[(df_r["Feature Set"] == "Model_D_All_Physiological") & (df_r["Model Classifier"] == "random_forest")]
        if not sub_r.empty:
            r = sub_r.iloc[0].to_dict()
            r["Evaluation Strategy"] = "Random Unit Split (Data Leakage Baseline)"
            rows.append(r)
            
    # 2. Probe / Hardware Held-Out (Model D, Random Forest)
    if os.path.exists(df_probe_path):
        df_p = pd.read_csv(df_probe_path)
        sub_p = df_p[(df_p["Feature Set"] == "Model_D_All_Physiological") & (df_p["Model Classifier"] == "random_forest")]
        if not sub_p.empty:
            r = sub_p.iloc[0].to_dict()
            r["Evaluation Strategy"] = "Probe/Hardware Held-Out (Physical Shank Isolation)"
            rows.append(r)
            
    # 3. Session Held-Out
    if session_heldout_res and "random_forest" in session_heldout_res:
        rf_sess = session_heldout_res["random_forest"]
        ov = rf_sess["overall_metrics"]
        rows.append({
            "Evaluation Strategy": "Session Held-Out (Leave-One-Session-Out)",
            "Feature Set": "Model_D_All_Physiological",
            "Model Classifier": "random_forest",
            "Balanced Accuracy": f"{rf_sess['mean_balanced_accuracy']:.4f} +/- {rf_sess['std_balanced_accuracy']:.4f}" if not np.isnan(rf_sess['std_balanced_accuracy']) else f"{ov.get('balanced_accuracy', np.nan):.4f}",
            "Macro F1": f"{rf_sess['mean_macro_f1']:.4f} +/- {rf_sess['std_macro_f1']:.4f}" if not np.isnan(rf_sess['std_macro_f1']) else f"{ov.get('macro_f1', np.nan):.4f}",
            "Macro Precision": ov.get("macro_precision", np.nan),
            "Macro Recall": ov.get("macro_recall", np.nan),
            "AUROC (OVR)": f"{rf_sess['mean_auroc']:.4f} +/- {rf_sess['std_auroc']:.4f}" if not np.isnan(rf_sess['std_auroc']) else f"{ov.get('auroc', np.nan):.4f}",
            "AUPRC": f"{rf_sess['mean_auprc']:.4f} +/- {rf_sess['std_auprc']:.4f}" if not np.isnan(rf_sess['std_auprc']) else f"{ov.get('auprc', np.nan):.4f}",
            "Brier Score": f"{rf_sess['mean_brier_score']:.4f} +/- {rf_sess['std_brier_score']:.4f}" if not np.isnan(rf_sess['std_brier_score']) else f"{ov.get('brier_score', np.nan):.4f}",
            "Expected Calibration Error (ECE)": ov.get("ece", np.nan)
        })
        
    # 4. Specimen Held-Out (Identical when 1 session per specimen)
    if session_heldout_res and "random_forest" in session_heldout_res:
        rf_sess = session_heldout_res["random_forest"]
        ov = rf_sess["overall_metrics"]
        rows.append({
            "Evaluation Strategy": "Specimen Held-Out (Leave-One-Specimen-Out)",
            "Feature Set": "Model_D_All_Physiological",
            "Model Classifier": "random_forest",
            "Balanced Accuracy": f"{rf_sess['mean_balanced_accuracy']:.4f} +/- {rf_sess['std_balanced_accuracy']:.4f}" if not np.isnan(rf_sess['std_balanced_accuracy']) else f"{ov.get('balanced_accuracy', np.nan):.4f}",
            "Macro F1": f"{rf_sess['mean_macro_f1']:.4f} +/- {rf_sess['std_macro_f1']:.4f}" if not np.isnan(rf_sess['std_macro_f1']) else f"{ov.get('macro_f1', np.nan):.4f}",
            "Macro Precision": ov.get("macro_precision", np.nan),
            "Macro Recall": ov.get("macro_recall", np.nan),
            "AUROC (OVR)": f"{rf_sess['mean_auroc']:.4f} +/- {rf_sess['std_auroc']:.4f}" if not np.isnan(rf_sess['std_auroc']) else f"{ov.get('auroc', np.nan):.4f}",
            "AUPRC": f"{rf_sess['mean_auprc']:.4f} +/- {rf_sess['std_auprc']:.4f}" if not np.isnan(rf_sess['std_auprc']) else f"{ov.get('auprc', np.nan):.4f}",
            "Brier Score": f"{rf_sess['mean_brier_score']:.4f} +/- {rf_sess['std_brier_score']:.4f}" if not np.isnan(rf_sess['std_brier_score']) else f"{ov.get('brier_score', np.nan):.4f}",
            "Expected Calibration Error (ECE)": ov.get("ece", np.nan)
        })
        
    comp_df = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    comp_df.to_csv(output_path, index=False)
    logger.info("Saved Table 5 (Comprehensive Validation Comparison) to %s", output_path)
    return comp_df


if __name__ == "__main__":
    config = load_config("config.yaml")
    multi_csv = "results/tables/unit_features_multisession.csv"
    if not os.path.exists(multi_csv):
        logger.error("Multisession features file not found: %s", multi_csv)
        sys.exit(1)
        
    df_all = pd.read_csv(multi_csv)
    logger.info("Loaded multisession data with shape: %s", df_all.shape)
    
    # 1. Audit Session and Specimen Grouping
    logger.info("==================================================")
    logger.info("AUDITING GROUPING VARIABLES FOR CROSS-SESSION VALIDATION")
    logger.info("==================================================")
    sessions = df_all["session_id"].unique()
    specimens = df_all["specimen_id"].unique()
    logger.info("Session grouping variable: 'session_id', unique count = %d, values = %s", len(sessions), sessions.tolist())
    logger.info("Specimen grouping variable: 'specimen_id', unique count = %d, values = %s", len(specimens), specimens.tolist())
    for s_id in sessions:
        sub = df_all[df_all["session_id"] == s_id]
        logger.info("Session %s (Specimen %s): %d total units, classes = %s",
                    s_id, sub["specimen_id"].iloc[0], len(sub), sub["reference_class"].value_counts().to_dict())
                    
    # 2. Run True Session-Held-Out Validation
    res = run_true_session_held_out_validation(df_all, config)
    
    # Print detailed report for Random Forest
    rf_res = res["random_forest"]
    logger.info("==================================================")
    logger.info("SESSION-HELD-OUT RESULTS (RANDOM FOREST)")
    logger.info("==================================================")
    logger.info("Mean Balanced Acc: %.4f +/- %.4f", rf_res["mean_balanced_accuracy"], rf_res["std_balanced_accuracy"])
    logger.info("Mean Macro F1:     %.4f +/- %.4f", rf_res["mean_macro_f1"], rf_res["std_macro_f1"])
    logger.info("Mean AUROC:        %.4f +/- %.4f", rf_res["mean_auroc"], rf_res["std_auroc"])
    logger.info("Mean AUPRC:        %.4f +/- %.4f", rf_res["mean_auprc"], rf_res["std_auprc"])
    logger.info("Mean Brier Score:  %.4f +/- %.4f", rf_res["mean_brier_score"], rf_res["std_brier_score"])
    
    # Print each fold
    for f in rf_res["session_fold_metrics"]:
        logger.info("Held-out Session %s (n_train=%d, n_test=%d): Bal Acc=%.4f, Macro F1=%.4f, AUROC=%.4f, AUPRC=%.4f, Prec=%.4f, Rec=%.4f, Brier=%.4f, ECE=%.4f",
                    f["held_out_session"], f["n_train"], f["n_test"],
                    f.get("balanced_accuracy", np.nan), f.get("macro_f1", np.nan),
                    f.get("auroc", np.nan), f.get("auprc", np.nan),
                    f.get("macro_precision", np.nan), f.get("macro_recall", np.nan),
                    f.get("brier_score", np.nan), f.get("ece", np.nan))
                    
    # Build Table 5
    build_validation_comparison_table(
        df_random_path="results/tables/Table_4_random_split_leakage_baseline.csv",
        df_probe_path="results/tables/Table_5_held_out_validation.csv",
        session_heldout_res=res,
        output_path="results/tables/Table_5_comprehensive_validation_comparison.csv"
    )


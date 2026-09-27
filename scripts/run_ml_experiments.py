"""
run_ml_experiments.py
=====================
Executes Phase 3 machine learning benchmarking, data leakage contrast,
feature ablation studies, and generates publication Tables 4-7 and Figures 4-7.
"""

import sys
import os
import json
import logging
import platform
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_access import load_config
from src.models import FEATURE_SETS
from src.validation import run_cross_validation, CLASS_NAMES
from src.tables import format_ml_results_table
from src.visualization import (
    generate_figure4_model_comparisons,
    generate_figure5_cross_session_generalization,
    generate_figure6_feature_ablation,
    generate_figure7_calibration_and_uncertainty
)
from src.pipeline import save_run_metadata

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ml_experiments")


def main():
    config = load_config("config.yaml")
    features_csv = "results/tables/unit_features_721123822.csv"
    
    if not os.path.exists(features_csv):
        features_csv = "results/tables/unit_features.csv"
        
    logger.info("Loading unit features from: %s", features_csv)
    df = pd.read_csv(features_csv)
    logger.info("Loaded %d units (valid reference classes: %s)",
                len(df), df["reference_class"].value_counts().to_dict())
    
    models = ["logistic_regression", "random_forest", "xgboost"]
    feature_set_names = list(FEATURE_SETS.keys())
    
    # ----------------------------------------------------
    # 1. Random Unit Split CV (Data Leakage Baseline)
    # ----------------------------------------------------
    logger.info("==================================================")
    logger.info("RUNNING RANDOM UNIT SPLIT CV (LEAKAGE BASELINE)")
    logger.info("==================================================")
    random_results = []
    
    for f_name in feature_set_names:
        f_cols = FEATURE_SETS[f_name]
        for m_name in models:
            logger.info("Evaluating Random Split: %s | %s", m_name, f_name)
            res = run_cross_validation(
                df=df,
                feature_cols=f_cols,
                model_name=m_name,
                cv_mode="random_unit_split",
                config=config,
                classes=CLASS_NAMES
            )
            res["feature_set_name"] = f_name
            random_results.append(res)
            
    df_random = format_ml_results_table(
        random_results,
        output_path="results/tables/Table_4_random_split_leakage_baseline.csv",
        table_name="Table 4 (Random Unit Split Leakage Baseline)"
    )
    
    # ----------------------------------------------------
    # 2. Hardware-Held-Out CV (Leakage-Free Evaluation)
    # ----------------------------------------------------
    logger.info("==================================================")
    logger.info("RUNNING HARDWARE-HELD-OUT CV (LEAKAGE-FREE)")
    logger.info("==================================================")
    heldout_results = []
    best_oof_df = None
    
    for f_name in feature_set_names:
        f_cols = FEATURE_SETS[f_name]
        for m_name in models:
            logger.info("Evaluating Hardware-Held-Out: %s | %s", m_name, f_name)
            res = run_cross_validation(
                df=df,
                feature_cols=f_cols,
                model_name=m_name,
                cv_mode="probe_held_out",
                config=config,
                classes=CLASS_NAMES
            )
            res["feature_set_name"] = f_name
            heldout_results.append(res)
            
            # Save full physiological model predictions for calibration analysis
            if f_name == "Model_D_All_Physiological" and m_name == "random_forest":
                best_oof_df = res["oof_df"]
                
    df_heldout = format_ml_results_table(
        heldout_results,
        output_path="results/tables/Table_5_held_out_validation.csv",
        table_name="Table 5 (Hardware-Held-Out Validation)"
    )
    
    # ----------------------------------------------------
    # 3. Progressive Feature Ablation Table (Table 6)
    # ----------------------------------------------------
    logger.info("==================================================")
    logger.info("GENERATING TABLE 6: FEATURE ABLATION STUDY")
    logger.info("==================================================")
    # Filter held-out results for ablation
    ablation_records = []
    for res in heldout_results:
        m = res["overall_metrics"]
        ablation_records.append({
            "Feature Set": res["feature_set_name"],
            "Model Classifier": res["model_name"],
            "Features Included": ", ".join(FEATURE_SETS[res["feature_set_name"]]),
            "Num Features": len(FEATURE_SETS[res["feature_set_name"]]),
            "Balanced Accuracy": m.get("balanced_accuracy", np.nan),
            "Macro F1": m.get("macro_f1", np.nan),
            "AUROC (OVR)": m.get("auroc", np.nan),
            "AUPRC": m.get("auprc", np.nan),
            "Brier Score": m.get("brier_score", np.nan),
            "ECE": m.get("ece", np.nan)
        })
    df_ablation = pd.DataFrame(ablation_records)
    df_ablation.to_csv("results/tables/Table_6_feature_ablation_study.csv", index=False)
    logger.info("Saved Table 6 to results/tables/Table_6_feature_ablation_study.csv")
    
    # ----------------------------------------------------
    # 4. Publication Figures (Figures 4, 5, 6, 7)
    # ----------------------------------------------------
    logger.info("==================================================")
    logger.info("GENERATING PUBLICATION FIGURES (4 - 7)")
    logger.info("==================================================")
    
    # Figure 4: Model comparison under full physiological features
    full_phys_heldout = df_heldout[df_heldout["Feature Set"] == "Model_D_All_Physiological"].copy()
    generate_figure4_model_comparisons(full_phys_heldout, "results/figures/Figure_4_model_comparisons.png")
    
    # Figure 5: Leakage comparison (Random Unit Split vs Hardware-Held-Out)
    comp_random = df_random[df_random["Feature Set"] == "Model_D_All_Physiological"].copy()
    comp_random["Evaluation Strategy"] = "Random Unit Split (Data Leakage)"
    comp_heldout = df_heldout[df_heldout["Feature Set"] == "Model_D_All_Physiological"].copy()
    comp_heldout["Evaluation Strategy"] = "Hardware-Held-Out (Leakage-Free)"
    
    leakage_df = pd.concat([comp_random, comp_heldout], ignore_index=True)
    generate_figure5_cross_session_generalization(leakage_df, "results/figures/Figure_5_leakage_comparison.png")
    
    # Figure 6: Feature ablation curves
    generate_figure6_feature_ablation(df_ablation, "results/figures/Figure_6_feature_ablation.png")
    
    # Figure 7: Calibration & Uncertainty analysis
    if best_oof_df is not None:
        generate_figure7_calibration_and_uncertainty(best_oof_df, "results/figures/Figure_7_calibration_and_uncertainty.png")
        
    # ----------------------------------------------------
    # 5. Save Run Reproducibility Metadata
    # ----------------------------------------------------
    save_run_metadata(config, "results/run_metadata.json")
    logger.info("ALL MACHINE LEARNING EXPERIMENTS COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()

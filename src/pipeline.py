"""
pipeline.py
===========
End-to-end research execution pipeline for:
"A Reliability-Aware Multifeature Framework for Automated Optotagging of Neuropixels Units"

Supports:
- Phase 0: Discovery & session inventory
- Phase 1: Single representative session validation (Session 721123822)
- Phase 2: Population feature extraction and structured Parquet generation
- Phase 3: Machine learning, feature ablation, session-held-out validation
- Phase 4: Publication figures (Figures 1-7) and tables (Tables 1-7)
"""

import sys
import os
import argparse
import logging
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data_access import load_config, initialize_cache, get_session_data
from src.session_discovery import build_session_inventory, print_discovery_summary
from src.opto_trials import parse_opto_trials, get_trials_by_type, summarize_opto_conditions
from src.spike_alignment import build_session_trial_responses, save_trial_responses_parquet
from src.artifact_control import ArtifactController
from src.feature_extraction import extract_unit_features_for_10ms_pulses, compute_train_adaptation_index
from src.labeling import label_unit_features_table, run_threshold_sensitivity_analysis
from src.models import FEATURE_SETS
from src.validation import run_cross_validation, evaluate_predictions, CLASS_NAMES
from src.tables import (
    generate_table1_dataset_characteristics,
    generate_table2_feature_distributions_by_class,
    generate_table3_threshold_sensitivity,
    format_ml_results_table
)
from src.visualization import (
    plot_unit_raster_and_psth, generate_figure1_pipeline_schematic,
    generate_figure3_feature_distributions
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("pipeline")


def run_single_session_pipeline(session_id: int, config: dict, cache=None) -> dict:
    """
    Execute Phase 1 on a single representative session.
    """
    logger.info("==================================================")
    logger.info("STARTING PHASE 1: SINGLE-SESSION VALIDATION (%s)", session_id)
    logger.info("==================================================")
    
    if cache is None:
        cache = initialize_cache(config)
        
    session = get_session_data(cache, session_id)
    
    # Ensure channel schema compatibility for AllenSDK lazy properties
    if "structure_acronym" in session.channels.columns and "ecephys_structure_acronym" not in session.channels.columns:
        session.channels["ecephys_structure_acronym"] = session.channels["structure_acronym"]
        
    try:
        specimen_id = session.metadata.get("specimen_id", session_id)
    except Exception:
        from src.data_access import load_session_table
        try:
            sessions_df = load_session_table(cache, config)
            if sessions_df is not None and session_id in sessions_df.index and "specimen_id" in sessions_df.columns:
                specimen_id = int(sessions_df.loc[session_id, "specimen_id"])
            else:
                specimen_id = session_id
        except Exception:
            specimen_id = session_id
    
    # 1. Parse optogenetic stimulation trials
    raw_opto = session.optogenetic_stimulation_epochs
    trials_df = parse_opto_trials(raw_opto)
    summary_opto = summarize_opto_conditions(trials_df)
    logger.info("Session %s opto conditions: %s", session_id, summary_opto["trial_counts"])
    
    # Filter for primary condition (10-ms pulses)
    trials_10ms = get_trials_by_type(trials_df, stimulus_type="10ms_pulse")
    if trials_10ms.empty:
        logger.warning("No 10-ms pulse trials found for session %s. Using all available square pulses.", session_id)
        trials_10ms = trials_df[trials_df["duration"] <= 0.015].copy()
        
    logger.info("Identified %d primary 10-ms optical pulse trials", len(trials_10ms))
    
    # 2. Extract unit spike times
    units_df = session.units
    n_probes = len(session.probes) if hasattr(session, "probes") else 1
    logger.info("Session contains %d total units across %d probes", len(units_df), n_probes)
    
    units_dict = {}
    for uid in units_df.index:
        units_dict[uid] = session.spike_times[uid]
        
    # 3. Align spikes and compute trial responses
    trial_responses_df, aligned_spikes_dict = build_session_trial_responses(
        session_id=session_id,
        specimen_id=specimen_id,
        units_dict=units_dict,
        trials_df=trials_10ms,
        config=config
    )
    
    # 4. Artifact control and pre-onset sham analysis
    artifact_ctrl = ArtifactController(config)
    unit_features_list = []
    
    # Check secondary train trials for adaptation index
    train_trials = get_trials_by_type(trials_df, stimulus_type="2.5ms_train")
    
    for uid in units_df.index:
        aligned_list = aligned_spikes_dict[uid]
        art_record = artifact_ctrl.check_spike_timing_artifacts(session_id, uid, aligned_list)
        
        # Unit metadata
        u_row = units_df.loc[uid]
        probe_id = u_row.get("probe_id", u_row.get("ecephys_probe_id", "unknown"))
        brain_area = u_row.get("structure_acronym", u_row.get("ecephys_structure_acronym", "unknown"))
        
        u_trials = trial_responses_df[trial_responses_df["unit_id"] == uid]
        feats = extract_unit_features_for_10ms_pulses(
            unit_id=uid,
            session_id=session_id,
            specimen_id=specimen_id,
            probe_id=probe_id,
            brain_area=brain_area,
            unit_trials_df=u_trials,
            artifact_record=art_record,
            config=config
        )
        
        # Calculate train adaptation index if available
        if not train_trials.empty:
            adapt_idx, _ = compute_train_adaptation_index(uid, train_trials, units_dict[uid])
            feats["adaptation_index"] = adapt_idx
        else:
            feats["adaptation_index"] = np.nan
            
        unit_features_list.append(feats)
        
    feats_df = pd.DataFrame(unit_features_list)
    
    # 5. Operational reference labeling
    primary_thresh = config["thresholds"]["primary"]
    labeled_df = label_unit_features_table(
        feats_df,
        lat_thresh_ms=primary_thresh["latency_ms"],
        rel_thresh=primary_thresh["reliability"],
        mod_thresh=primary_thresh["modulation_ratio"],
        p_thresh=primary_thresh["p_value"]
    )
    
    class_counts = labeled_df["reference_class"].value_counts().to_dict()
    logger.info("Operational labeling results for session %s: %s", session_id, class_counts)
    
    # 6. Sensitivity analysis (27-parameter sweep)
    sens_cfg = config["thresholds"]["sensitivity"]
    sens_df = run_threshold_sensitivity_analysis(
        feats_df,
        latency_sweep=sens_cfg["latency_sweep_ms"],
        reliability_sweep=sens_cfg["reliability_sweep"],
        modulation_sweep=sens_cfg["modulation_sweep"],
        p_thresh=primary_thresh["p_value"]
    )
    
    # 7. Generate representative raster & PSTH plots
    fig_dir = Path(config["paths"]["figures_dir"]) / "representative_units"
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # Pick direct, indirect, and non-responsive examples
    direct_units = labeled_df[labeled_df["reference_class"] == "putatively directly optotagged"]
    indirect_units = labeled_df[labeled_df["reference_class"] == "light-responsive / indirect or uncertain"]
    non_resp_units = labeled_df[labeled_df["reference_class"] == "not light responsive"]
    
    plot_uids = []
    if not direct_units.empty:
        plot_uids.append((direct_units.iloc[0]["unit_id"], "putatively directly optotagged"))
    if not indirect_units.empty:
        plot_uids.append((indirect_units.iloc[0]["unit_id"], "light-responsive / indirect or uncertain"))
    if not non_resp_units.empty:
        plot_uids.append((non_resp_units.iloc[0]["unit_id"], "not light responsive"))
        
    for uid, r_cls in plot_uids:
        safe_cls = r_cls.replace(" ", "_").replace("/", "_")
        plot_path = fig_dir / f"session_{session_id}_unit_{uid}_{safe_cls}.png"
        plot_unit_raster_and_psth(aligned_spikes_dict[uid], uid, r_cls, output_path=str(plot_path))
        
    # Save exclusion log
    artifact_ctrl.save_exclusion_log(config["paths"]["exclusion_log"])
    
    # Save structured trial responses parquet
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    trial_resp_path = f"data/processed/unit_trial_responses_{session_id}.parquet"
    trial_responses_df.to_parquet(trial_resp_path, index=False)
    trial_responses_df.to_parquet("data/processed/unit_trial_responses.parquet", index=False)
    logger.info("Saved trial responses to %s and unit_trial_responses.parquet", trial_resp_path)
    
    # Save unit features table
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    feats_csv_path = f"results/tables/unit_features_{session_id}.csv"
    labeled_df.to_csv(feats_csv_path, index=False)
    labeled_df.to_csv("results/tables/unit_features.csv", index=False)
    logger.info("Saved unit features table to %s and unit_features.csv", feats_csv_path)
    
    # Generate Table 2 and Table 3
    generate_table2_feature_distributions_by_class(labeled_df, "results/tables/Table_2_feature_distributions.csv")
    generate_table3_threshold_sensitivity(sens_df, "results/tables/Table_3_threshold_sensitivity.csv")
    
    # Generate Figure 3
    fig3_path = "results/figures/Figure_3_feature_distributions.png"
    generate_figure3_feature_distributions(labeled_df, fig3_path)
    
    return {
        "session_id": session_id,
        "labeled_df": labeled_df,
        "trial_responses_df": trial_responses_df,
        "sensitivity_df": sens_df,
        "aligned_spikes_dict": aligned_spikes_dict,
        "class_counts": class_counts
    }


def save_run_metadata(config: dict, output_path: str):
    """Record environment and library versions for reproducibility."""
    import platform
    import numpy as np
    import scipy
    import pandas as pd
    import sklearn
    import allensdk
    
    meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "python_version": sys.version,
        "operating_system": platform.platform(),
        "platform_machine": platform.machine(),
        "allensdk_version": allensdk.__version__,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "pandas_version": pd.__version__,
        "sklearn_version": sklearn.__version__,
        "config_parameters": config
    }
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logger.info("Saved run metadata to %s", output_path)

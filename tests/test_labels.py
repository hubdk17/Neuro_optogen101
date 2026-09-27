"""
test_labels.py
==============
Unit tests for operational rule-based labeling and sensitivity sweep.
"""

import numpy as np
import pandas as pd
import pytest
from src.labeling import (
    assign_reference_label,
    label_unit_features_table,
    run_threshold_sensitivity_analysis,
    CLASS_DIRECT,
    CLASS_INDIRECT,
    CLASS_NON_RESPONSIVE,
    CLASS_INSUFFICIENT
)


def test_assign_reference_label():
    # 1. Putatively direct
    row_direct = pd.Series({
        "n_trials": 50,
        "baseline_rate": 2.0,
        "evoked_rate": 25.0,
        "modulation_ratio": 8.33,
        "median_latency_ms": 3.8,
        "latency_sd_ms": 1.1,
        "trial_reliability": 0.75,
        "p_value": 0.0001,
        "effect_size": 1.5,
        "artifact_flag": False
    })
    cls_d, conf_d, reason_d = assign_reference_label(row_direct)
    assert cls_d == CLASS_DIRECT
    assert conf_d >= 0.70
    assert "Passes direct criteria" in reason_d
    
    # 2. Indirect due to delayed latency (>8ms)
    row_delayed = row_direct.copy()
    row_delayed["median_latency_ms"] = 12.5
    cls_del, _, reason_del = assign_reference_label(row_delayed)
    assert cls_del == CLASS_INDIRECT
    assert "Delayed latency" in reason_del
    
    # 3. Indirect due to weak reliability (<0.30)
    row_unreliable = row_direct.copy()
    row_unreliable["trial_reliability"] = 0.18
    cls_unrel, _, reason_unrel = assign_reference_label(row_unreliable)
    assert cls_unrel == CLASS_INDIRECT
    assert "Sub-threshold reliability" in reason_unrel
    
    # 4. Not light responsive
    row_non = pd.Series({
        "n_trials": 50,
        "baseline_rate": 5.0,
        "evoked_rate": 4.8,
        "modulation_ratio": 0.80,
        "median_latency_ms": np.nan,
        "latency_sd_ms": np.nan,
        "trial_reliability": 0.04,
        "p_value": 0.65,
        "effect_size": -0.05,
        "artifact_flag": False
    })
    cls_non, _, reason_non = assign_reference_label(row_non)
    assert cls_non == CLASS_NON_RESPONSIVE
    
    # 5. Insufficient trials
    row_few = row_direct.copy()
    row_few["n_trials"] = 5
    cls_few, _, _ = assign_reference_label(row_few)
    assert cls_few == CLASS_INSUFFICIENT


def test_run_threshold_sensitivity_analysis():
    # Synthetic cohort of 4 units
    units = [
        {"unit_id": 1, "n_trials": 50, "baseline_rate": 1.0, "evoked_rate": 30.0, "modulation_ratio": 15.0, "median_latency_ms": 4.0, "trial_reliability": 0.8, "p_value": 0.001, "effect_size": 2.0, "artifact_flag": False},
        {"unit_id": 2, "n_trials": 50, "baseline_rate": 2.0, "evoked_rate": 20.0, "modulation_ratio": 6.0, "median_latency_ms": 7.5, "trial_reliability": 0.25, "p_value": 0.001, "effect_size": 1.0, "artifact_flag": False},
        {"unit_id": 3, "n_trials": 50, "baseline_rate": 5.0, "evoked_rate": 5.0, "modulation_ratio": 0.8, "median_latency_ms": np.nan, "trial_reliability": 0.02, "p_value": 0.8, "effect_size": 0.0, "artifact_flag": False},
        {"unit_id": 4, "n_trials": 50, "baseline_rate": 1.0, "evoked_rate": 15.0, "modulation_ratio": 7.5, "median_latency_ms": 9.2, "trial_reliability": 0.6, "p_value": 0.001, "effect_size": 1.2, "artifact_flag": False}
    ]
    df = pd.DataFrame(units)
    
    sens_df = run_threshold_sensitivity_analysis(
        df,
        latency_sweep=[6.0, 8.0, 10.0],
        reliability_sweep=[0.20, 0.30, 0.50],
        modulation_sweep=[1.5, 2.0, 3.0]
    )
    
    # 3 * 3 * 3 = 27 combinations
    assert len(sens_df) == 27
    assert "pct_putatively_direct" in sens_df.columns
    # Check that unit counts sum to total
    for _, row in sens_df.iterrows():
        total = row["n_putatively_direct"] + row["n_indirect_uncertain"] + row["n_not_responsive"] + row["n_insufficient"]
        assert total == 4

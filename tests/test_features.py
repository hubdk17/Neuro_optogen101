"""
test_features.py
================
Unit tests for feature extraction, modulation ratio, latency statistics,
and 10-Hz train dynamics.
"""

import numpy as np
import pandas as pd
import pytest
from src.feature_extraction import extract_unit_features_for_10ms_pulses, compute_train_adaptation_index


def test_extract_unit_features_responsive_vs_nonresponsive():
    config = {
        "project": {"random_seed": 42},
        "windows_seconds": {
            "baseline": [-0.020, -0.005],  # 15 ms
            "evoked": [0.001, 0.009],      # 8 ms
            "sham": [-0.018, -0.010]
        }
    }
    
    n_trials = 50
    # Responsive unit: 1 evoked spike on 40 of 50 trials at 3.5ms latency, baseline has 0 spikes
    resp_rows = []
    for t in range(n_trials):
        has_spk = (t < 40)
        resp_rows.append({
            "trial_id": t,
            "baseline_count": 0,
            "evoked_count": 1 if has_spk else 0,
            "sham_count": 0,
            "has_evoked_spike": has_spk,
            "first_spike_latency_ms": 3.5 if has_spk else np.nan,
            "optical_level": 1.0
        })
    df_resp = pd.DataFrame(resp_rows)
    
    feats_resp = extract_unit_features_for_10ms_pulses(
        unit_id=101,
        session_id=1,
        specimen_id=1,
        probe_id=1,
        brain_area="VISp",
        unit_trials_df=df_resp,
        artifact_record={"artifact_flag": False},
        config=config
    )
    
    # 40 spikes in 50 trials of 8ms -> (40/50) / 0.008 = 100 Hz evoked rate
    assert np.isclose(feats_resp["evoked_rate"], 100.0, atol=1.0)
    assert np.isclose(feats_resp["baseline_rate"], 0.0, atol=0.1)
    # Modulation ratio = 100 / (0 + 1) = 100
    assert np.isclose(feats_resp["modulation_ratio"], 100.0, atol=1.0)
    assert feats_resp["trial_reliability"] == 0.80
    assert np.isclose(feats_resp["median_latency_ms"], 3.5, atol=0.1)
    assert feats_resp["p_value"] < 0.05
    assert feats_resp["effect_size"] > 1.0
    
    # Non-responsive unit: spontaneous firing equally across baseline and evoked
    non_rows = []
    for t in range(n_trials):
        non_rows.append({
            "trial_id": t,
            "baseline_count": 0,
            "evoked_count": 0,
            "sham_count": 0,
            "has_evoked_spike": False,
            "first_spike_latency_ms": np.nan,
            "optical_level": 1.0
        })
    df_non = pd.DataFrame(non_rows)
    
    feats_non = extract_unit_features_for_10ms_pulses(
        unit_id=102,
        session_id=1,
        specimen_id=1,
        probe_id=1,
        brain_area="VISp",
        unit_trials_df=df_non,
        artifact_record={"artifact_flag": False},
        config=config
    )
    
    assert feats_non["trial_reliability"] == 0.0
    assert feats_non["modulation_ratio"] == 0.0
    assert np.isnan(feats_non["median_latency_ms"])


def test_compute_train_adaptation_index():
    # 1 trial starting at t = 0.0s
    train_trials = pd.DataFrame([{"start_time": 0.0}])
    
    # Spikes for pulse 0 (at 0.003s) and pulse 1 (at 0.103s), but decaying later
    spk_times = np.array([0.003, 0.103, 0.203])  # responses on first 3 pulses, none on pulse 9
    
    adapt_idx, probs = compute_train_adaptation_index(
        unit_id=101,
        train_trials_df=train_trials,
        spike_times=spk_times,
        n_pulses=10,
        pulse_freq_hz=10.0
    )
    
    assert len(probs) == 10
    assert probs[0] == 1.0
    assert probs[9] == 0.0
    # adaptation = 1 - (0 / 1) = 1.0
    assert np.isclose(adapt_idx, 1.0, atol=0.01)

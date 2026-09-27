"""
test_alignment.py
=================
Unit tests for spike alignment, window definitions, and trial metric computation.
"""

import numpy as np
import pytest
from src.spike_alignment import align_unit_spikes_to_trials, compute_trial_metrics


def test_align_unit_spikes_to_trials():
    # 2 trials at t = 10.0s and t = 20.0s
    trial_starts = np.array([10.0, 20.0])
    
    # Spikes around trial 1: 9.990s (-10ms), 10.003s (+3ms), 10.025s (+25ms)
    # Spikes around trial 2: 20.005s (+5ms)
    # Distant spike: 50.0s
    spike_times = np.array([9.990, 10.003, 10.025, 20.005, 50.0])
    
    aligned = align_unit_spikes_to_trials(spike_times, trial_starts, window_s=(-0.020, 0.030))
    
    assert len(aligned) == 2
    # Trial 1: -0.010, +0.003, +0.025
    assert len(aligned[0]) == 3
    np.testing.assert_allclose(aligned[0], [-0.010, 0.003, 0.025], atol=1e-5)
    
    # Trial 2: +0.005
    assert len(aligned[1]) == 1
    np.testing.assert_allclose(aligned[1], [0.005], atol=1e-5)


def test_compute_trial_metrics_artifact_exclusion():
    # Trial 1:
    # Spike at -0.010s (baseline: in [-20ms, -5ms])
    # Spike at +0.0005s (onset guard: in [0, 1ms) -> MUST NOT be in evoked!)
    # Spike at +0.004s (evoked: in [+1ms, +9ms])
    # Spike at +0.010s (offset guard: in [9ms, 11ms] -> MUST NOT be in evoked!)
    trial_1 = np.array([-0.010, 0.0005, 0.004, 0.010])
    
    # Trial 2: No spikes
    trial_2 = np.empty(0)
    
    # Trial 3: Multiple evoked spikes
    trial_3 = np.array([0.002, 0.006])
    
    aligned_list = [trial_1, trial_2, trial_3]
    
    b_counts, e_counts, s_counts, has_evoked, first_lat_ms = compute_trial_metrics(
        aligned_list,
        baseline_win=(-0.020, -0.005),
        evoked_win=(0.001, 0.009),
        sham_win=(-0.018, -0.010)
    )
    
    # Trial 1 checks
    assert b_counts[0] == 1
    assert e_counts[0] == 1  # only the 0.004s spike, NOT the 0.0005s or 0.010s spikes!
    assert has_evoked[0] is True or has_evoked[0] == 1
    assert np.isclose(first_lat_ms[0], 4.0, atol=1e-3)
    
    # Trial 2 checks
    assert b_counts[1] == 0
    assert e_counts[1] == 0
    assert has_evoked[1] == False
    assert np.isnan(first_lat_ms[1])
    
    # Trial 3 checks
    assert e_counts[2] == 2
    assert np.isclose(first_lat_ms[2], 2.0, atol=1e-3)  # first spike latency is 2.0 ms

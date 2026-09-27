"""
feature_extraction.py
======================
Unit-level feature engineering from trial responses for optotagging analysis.

Computes:
- Baseline & evoked firing rates (Hz)
- Modulation ratio: evoked_rate / (baseline_rate + 1.0)
- Median first-spike latency, SD, and IQR (ms)
- Trial reliability (fraction of trials with >=1 evoked spike)
- Statistical response significance (permutation p-value, Cohen's d, rank-biserial effect size)
- Trial-to-trial count variability (Fano factor) & latency CV
- Intensity tuning (per-level rate, reliability, modulation, latency slope)
- 10-Hz train dynamics & adaptation index (when available)
"""

from typing import Dict, Any, List, Optional, Tuple
import logging
import json
import numpy as np
import pandas as pd
from scipy.stats import linregress

from src.statistics import paired_permutation_test, cohens_d_paired, rank_biserial_effect_size

logger = logging.getLogger(__name__)


def extract_unit_features_for_10ms_pulses(
    unit_id: int,
    session_id: int,
    specimen_id: int,
    probe_id: Any,
    brain_area: str,
    unit_trials_df: pd.DataFrame,
    artifact_record: Dict[str, Any],
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Extract comprehensive multifeature representation for a single unit
    under the primary 10-ms pulse stimulation condition.
    
    Parameters:
    -----------
    unit_id : int
    session_id : int
    specimen_id : int
    probe_id : Any
    brain_area : str
    unit_trials_df : DataFrame of trial responses for this unit (filtered for 10ms pulses)
    artifact_record : dict from ArtifactController
    config : config dictionary
    """
    windows = config.get("windows_seconds", {})
    b_win = windows.get("baseline", [-0.020, -0.005])
    e_win = windows.get("evoked", [0.001, 0.009])
    
    b_dur = b_win[1] - b_win[0]  # 0.015 s
    e_dur = e_win[1] - e_win[0]  # 0.008 s
    
    n_trials = len(unit_trials_df)
    if n_trials == 0:
        return {
            "session_id": session_id,
            "specimen_id": specimen_id,
            "probe_id": probe_id,
            "unit_id": unit_id,
            "brain_area": brain_area,
            "n_trials": 0,
            "baseline_rate": 0.0,
            "evoked_rate": 0.0,
            "modulation_ratio": 0.0,
            "median_latency_ms": np.nan,
            "latency_sd_ms": np.nan,
            "latency_iqr_ms": np.nan,
            "trial_reliability": 0.0,
            "sham_reliability": 0.0,
            "fano_factor": np.nan,
            "latency_cv": np.nan,
            "p_value": 1.0,
            "effect_size": 0.0,
            "intensity_slope": np.nan,
            "response_at_each_light_level": "{}",
            "artifact_flag": False
        }
        
    b_counts = unit_trials_df["baseline_count"].values
    e_counts = unit_trials_df["evoked_count"].values
    has_evoked = unit_trials_df["has_evoked_spike"].values
    latencies = unit_trials_df["first_spike_latency_ms"].dropna().values
    
    # 1. Firing rates (Hz)
    mean_b_count = np.mean(b_counts)
    mean_e_count = np.mean(e_counts)
    baseline_rate = float(mean_b_count / b_dur)
    evoked_rate = float(mean_e_count / e_dur)
    
    # 2. Modulation ratio
    # Rationale for +1 stabilizer: regularizes against division by zero in quiescent units
    modulation_ratio = float(evoked_rate / (baseline_rate + 1.0))
    
    # 3. Latency & variability
    if len(latencies) >= 1:
        median_lat_ms = float(np.median(latencies))
        lat_sd_ms = float(np.std(latencies, ddof=1)) if len(latencies) > 1 else 0.0
        q75, q25 = np.percentile(latencies, [75, 25])
        lat_iqr_ms = float(q75 - q25)
        lat_cv = float(lat_sd_ms / (np.mean(latencies) + 1e-6))
    else:
        median_lat_ms = np.nan
        lat_sd_ms = np.nan
        lat_iqr_ms = np.nan
        lat_cv = np.nan
        
    # 4. Trial reliability
    trial_reliability = float(np.sum(has_evoked) / n_trials)
    
    # 5. Sham reliability (false-positive rate in matched pre-onset window)
    sham_counts = unit_trials_df["sham_count"].values if "sham_count" in unit_trials_df.columns else np.zeros(n_trials)
    sham_reliability = float(np.sum(sham_counts >= 1) / n_trials)
    
    # 6. Fano factor of evoked counts
    var_e = np.var(e_counts, ddof=1) if n_trials > 1 else 0.0
    fano_factor = float(var_e / (mean_e_count + 1e-6))
    
    # 7. Statistical significance: paired permutation test
    # Scale baseline count to identical 8 ms duration for paired comparison
    b_counts_scaled = b_counts * (e_dur / b_dur)
    _, p_val = paired_permutation_test(e_counts, b_counts_scaled, n_permutations=1000, random_state=config.get("project", {}).get("random_seed", 42))
    eff_size = cohens_d_paired(e_counts, b_counts_scaled)
    
    # 8. Optical intensity dependence
    level_summary = {}
    intensity_slope = np.nan
    
    if "optical_level" in unit_trials_df.columns:
        levels = unit_trials_df["optical_level"].unique()
        levels_sorted = sorted(levels, key=lambda x: float(x) if str(x).replace('.','',1).isdigit() else str(x))
        
        level_rates = []
        numeric_levels = []
        
        for lvl in levels_sorted:
            sub = unit_trials_df[unit_trials_df["optical_level"] == lvl]
            sub_e_rate = float(np.mean(sub["evoked_count"].values) / e_dur)
            sub_b_rate = float(np.mean(sub["baseline_count"].values) / b_dur)
            sub_rel = float(np.sum(sub["has_evoked_spike"].values) / max(1, len(sub)))
            sub_lats = sub["first_spike_latency_ms"].dropna().values
            sub_lat = float(np.median(sub_lats)) if len(sub_lats) > 0 else np.nan
            sub_mod = float(sub_e_rate / (sub_b_rate + 1.0))
            
            level_summary[str(lvl)] = {
                "evoked_rate_hz": np.round(sub_e_rate, 2),
                "baseline_rate_hz": np.round(sub_b_rate, 2),
                "reliability": np.round(sub_rel, 3),
                "modulation_ratio": np.round(sub_mod, 2),
                "median_latency_ms": np.round(sub_lat, 2) if not np.isnan(sub_lat) else None
            }
            
            try:
                num_lvl = float(lvl)
                numeric_levels.append(num_lvl)
                level_rates.append(sub_e_rate)
            except ValueError:
                pass
                
        if len(numeric_levels) >= 3 and len(set(numeric_levels)) >= 3:
            slope, intercept, r_value, p_value, std_err = linregress(numeric_levels, level_rates)
            intensity_slope = float(slope)
            
    # 9. Artifact flag
    artifact_flag = bool(artifact_record.get("artifact_flag", False))
    
    feature_dict = {
        "session_id": session_id,
        "specimen_id": specimen_id,
        "probe_id": probe_id,
        "unit_id": unit_id,
        "brain_area": brain_area,
        "n_trials": n_trials,
        "baseline_rate": np.round(baseline_rate, 3),
        "evoked_rate": np.round(evoked_rate, 3),
        "modulation_ratio": np.round(modulation_ratio, 3),
        "median_latency_ms": np.round(median_lat_ms, 3) if not np.isnan(median_lat_ms) else np.nan,
        "latency_sd_ms": np.round(lat_sd_ms, 3) if not np.isnan(lat_sd_ms) else np.nan,
        "latency_iqr_ms": np.round(lat_iqr_ms, 3) if not np.isnan(lat_iqr_ms) else np.nan,
        "trial_reliability": np.round(trial_reliability, 3),
        "sham_reliability": np.round(sham_reliability, 3),
        "fano_factor": np.round(fano_factor, 3) if not np.isnan(fano_factor) else np.nan,
        "latency_cv": np.round(lat_cv, 3) if not np.isnan(lat_cv) else np.nan,
        "p_value": float(np.round(p_val, 5)),
        "effect_size": float(np.round(eff_size, 3)),
        "intensity_slope": float(np.round(intensity_slope, 4)) if not np.isnan(intensity_slope) else np.nan,
        "response_at_each_light_level": json.dumps(level_summary),
        "artifact_flag": artifact_flag
    }
    return feature_dict


def compute_train_adaptation_index(
    unit_id: int,
    train_trials_df: pd.DataFrame,
    spike_times: np.ndarray,
    n_pulses: int = 10,
    pulse_freq_hz: float = 10.0,
    pulse_duration_s: float = 0.0025,
    evoked_window_s: Tuple[float, float] = (0.001, 0.009)
) -> Tuple[float, List[float]]:
    """
    Calculate pulse-by-pulse response probability and adaptation index for 10-Hz pulse trains.
    
    Adaptation formula:
    adaptation = 1.0 - (response_probability_last / (response_probability_first + 1e-4))
    """
    if train_trials_df.empty or len(spike_times) == 0:
        return np.nan, [np.nan] * n_pulses
        
    trial_starts = train_trials_df["start_time"].values
    isi = 1.0 / pulse_freq_hz  # 0.100 s for 10 Hz
    
    pulse_responses = np.zeros(n_pulses, dtype=int)
    n_train_trials = len(trial_starts)
    
    for t_start in trial_starts:
        for p in range(n_pulses):
            p_onset = t_start + p * isi
            e_left = p_onset + evoked_window_s[0]
            e_right = p_onset + evoked_window_s[1]
            
            # Fast check if spike exists in pulse evoked window
            i_left = np.searchsorted(spike_times, e_left, side="left")
            i_right = np.searchsorted(spike_times, e_right, side="right")
            if i_right > i_left:
                pulse_responses[p] += 1
                
    pulse_probs = [float(pulse_responses[p] / n_train_trials) for p in range(n_pulses)]
    p_first = pulse_probs[0]
    p_last = pulse_probs[-1]
    
    # Adaptation index: 1 - p_last / (p_first + epsilon)
    adapt_idx = float(1.0 - (p_last / (p_first + 1e-4)))
    return np.round(adapt_idx, 3), [np.round(p, 3) for p in pulse_probs]

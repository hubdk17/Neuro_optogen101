"""
spike_alignment.py
==================
Vectorized spike-trial alignment and structured intermediate representation.

Computes:
- Baseline spike count in [-20 ms, -5 ms]
- Evoked spike count in [+1 ms, +9 ms]
- Matched pre-onset sham count in [-18 ms, -10 ms]
- First-spike latency (ms)
- Evoked spike occurrence indicator (>= 1 spike)
- Raster-ready relative spike times in [-20 ms, +30 ms]

Utilizes vectorized np.searchsorted for fast processing on CPU.
Outputs trial-level table for serialization to Parquet.
"""

from typing import Dict, Any, List, Optional, Tuple
import logging
from pathlib import Path
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def align_unit_spikes_to_trials(
    spike_times: np.ndarray,
    trial_starts: np.ndarray,
    window_s: Tuple[float, float] = (-0.020, 0.030)
) -> List[np.ndarray]:
    """
    Vectorized extraction of relative spike times for each trial within [t_start + win[0], t_start + win[1]].
    
    Parameters:
    -----------
    spike_times : 1D np.ndarray, sorted spike timestamps (seconds)
    trial_starts : 1D np.ndarray, trial onset timestamps (seconds)
    window_s : tuple (win_start, win_stop) in seconds relative to trial onset
    
    Returns:
    --------
    List of 1D np.ndarrays, one per trial, with spike times relative to trial onset (in seconds).
    """
    if len(spike_times) == 0 or len(trial_starts) == 0:
        return [np.empty(0, dtype=np.float64) for _ in range(len(trial_starts))]
        
    win_start, win_stop = window_s
    left_bounds = trial_starts + win_start
    right_bounds = trial_starts + win_stop
    
    # Fast searchsorted to find slice indices
    left_idxs = np.searchsorted(spike_times, left_bounds, side="left")
    right_idxs = np.searchsorted(spike_times, right_bounds, side="right")
    
    aligned_list = []
    for i in range(len(trial_starts)):
        spk_slice = spike_times[left_idxs[i]:right_idxs[i]]
        rel_spikes = spk_slice - trial_starts[i]
        aligned_list.append(rel_spikes)
        
    return aligned_list


def compute_trial_metrics(
    aligned_spikes_list: List[np.ndarray],
    baseline_win: Tuple[float, float] = (-0.020, -0.005),
    evoked_win: Tuple[float, float] = (0.001, 0.009),
    sham_win: Tuple[float, float] = (-0.018, -0.010)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate per-trial physiological response measurements.
    
    Returns:
    --------
    baseline_counts : np.ndarray (int)
    evoked_counts : np.ndarray (int)
    sham_counts : np.ndarray (int)
    has_evoked_spike : np.ndarray (bool)
    first_spike_latencies_ms : np.ndarray (float, NaN where no evoked spike)
    """
    n_trials = len(aligned_spikes_list)
    b_counts = np.zeros(n_trials, dtype=np.int32)
    e_counts = np.zeros(n_trials, dtype=np.int32)
    s_counts = np.zeros(n_trials, dtype=np.int32)
    has_evoked = np.zeros(n_trials, dtype=bool)
    first_lat_ms = np.full(n_trials, np.nan, dtype=np.float64)
    
    b_start, b_stop = baseline_win
    e_start, e_stop = evoked_win
    s_start, s_stop = sham_win
    
    for i, rel_spk in enumerate(aligned_spikes_list):
        if len(rel_spk) == 0:
            continue
            
        # Baseline count
        b_mask = (rel_spk >= b_start) & (rel_spk < b_stop)
        b_counts[i] = np.sum(b_mask)
        
        # Sham count
        s_mask = (rel_spk >= s_start) & (rel_spk < s_stop)
        s_counts[i] = np.sum(s_mask)
        
        # Evoked count & latency
        e_mask = (rel_spk >= e_start) & (rel_spk < e_stop)
        n_ev = np.sum(e_mask)
        e_counts[i] = n_ev
        
        if n_ev > 0:
            has_evoked[i] = True
            # First evoked spike relative to stimulus onset (in ms)
            first_spk = rel_spk[e_mask][0]
            first_lat_ms[i] = first_spk * 1000.0
            
    return b_counts, e_counts, s_counts, has_evoked, first_lat_ms


def build_session_trial_responses(
    session_id: int,
    specimen_id: int,
    units_dict: Dict[int, np.ndarray],
    trials_df: pd.DataFrame,
    config: Dict[str, Any]
) -> Tuple[pd.DataFrame, Dict[int, List[np.ndarray]]]:
    """
    Build structured unit x trial responses for a single session across all valid units.
    
    Parameters:
    -----------
    session_id : int
    specimen_id : int
    units_dict : dict mapping unit_id -> 1D np.ndarray of spike timestamps
    trials_df : standardized trials DataFrame (from opto_trials)
    config : configuration dictionary
    
    Returns:
    --------
    trial_responses_df : pd.DataFrame
    all_aligned_spikes : dict mapping unit_id -> List of aligned relative spike times per trial
    """
    windows = config.get("windows_seconds", {})
    baseline_win = tuple(windows.get("baseline", [-0.020, -0.005]))
    evoked_win = tuple(windows.get("evoked", [0.001, 0.009]))
    sham_win = tuple(windows.get("sham", [-0.018, -0.010]))
    psth_win = tuple(windows.get("psth", [-0.020, 0.030]))
    
    trial_starts = trials_df["start_time"].values
    trial_ids = trials_df["trial_id"].values
    stim_types = trials_df["stimulus_type"].values
    optical_levels = trials_df["optical_level"].values
    n_trials = len(trials_df)
    
    all_rows = []
    all_aligned_spikes = {}
    
    for unit_id, spike_times in units_dict.items():
        aligned = align_unit_spikes_to_trials(spike_times, trial_starts, window_s=psth_win)
        all_aligned_spikes[unit_id] = aligned
        
        b_cnt, e_cnt, s_cnt, has_ev, lat_ms = compute_trial_metrics(
            aligned, baseline_win=baseline_win, evoked_win=evoked_win, sham_win=sham_win
        )
        
        for t in range(n_trials):
            all_rows.append({
                "session_id": session_id,
                "specimen_id": specimen_id,
                "unit_id": unit_id,
                "trial_id": int(trial_ids[t]),
                "stimulus_type": stim_types[t],
                "optical_level": optical_levels[t],
                "baseline_count": int(b_cnt[t]),
                "evoked_count": int(e_cnt[t]),
                "sham_count": int(s_cnt[t]),
                "has_evoked_spike": bool(has_ev[t]),
                "first_spike_latency_ms": float(lat_ms[t]) if not np.isnan(lat_ms[t]) else np.nan
            })
            
    df_trials = pd.DataFrame(all_rows)
    logger.info("Aligned %d units across %d trials (%d total unit-trial rows)",
                len(units_dict), n_trials, len(df_trials))
    return df_trials, all_aligned_spikes


def save_trial_responses_parquet(df: pd.DataFrame, filepath: str):
    """Save trial responses to Apache Parquet format."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine="pyarrow")
    logger.info("Saved trial responses to %s (%d rows)", path, len(df))

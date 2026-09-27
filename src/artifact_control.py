"""
artifact_control.py
===================
Stimulation artifact detection, guard zone filtering, pre-onset sham control
analysis, and exclusion logging for optogenetic Neuropixels recordings.

Artifact Policy:
1. Photovoltaic transients (photoelectric effect) frequently induce rapid,
   synchronous voltage deflections at light transition edges:
   - Light onset (t = 0 ms): Guard zone [0.0 ms, 1.0 ms)
   - Light offset (t = 10 ms for 10-ms pulse): Guard zone [9.0 ms, 11.0 ms]
2. The physiological evoked window is strictly restricted to [+1.0 ms, +9.0 ms],
   preventing onset and offset edge artifact contamination from being treated as physiological spikes.
3. Pre-onset sham analysis:
   - Uses an identical 8-ms duration window [-18.0 ms, -10.0 ms].
   - Compares evoked metrics against pre-onset sham metrics to estimate spontaneous false-positive rates.
4. Suspicious short-latency detection:
   - First-spike latency < 1.2 ms with jitter < 0.1 ms indicates probable photoelectric artifact.
5. All flags and exclusions are logged to results/tables/exclusion_log.csv.
"""

from typing import Dict, Any, List, Optional, Tuple
import logging
from pathlib import Path
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ArtifactController:
    """Manages artifact detection, sham controls, and logging."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        windows = config.get("windows_seconds", {})
        self.baseline_win = windows.get("baseline", [-0.020, -0.005])
        self.evoked_win = windows.get("evoked", [0.001, 0.009])
        self.sham_win = windows.get("sham", [-0.018, -0.010])
        self.onset_guard = windows.get("artifact_guard_onset", [0.000, 0.001])
        self.offset_guard = windows.get("artifact_guard_offset", [0.009, 0.011])
        
        self.exclusions: List[Dict[str, Any]] = []

    def check_spike_timing_artifacts(
        self,
        session_id: int,
        unit_id: int,
        aligned_spikes_list: List[np.ndarray]
    ) -> Dict[str, Any]:
        """
        Analyze aligned spikes across all trials for a unit to detect artifact flags.
        
        Parameters:
        -----------
        session_id : int
        unit_id : int
        aligned_spikes_list : List of 1D numpy arrays containing spike times (seconds) relative to onset.
        
        Returns:
        --------
        dict of artifact flags and diagnostic metrics.
        """
        all_aligned = np.concatenate(aligned_spikes_list) if len(aligned_spikes_list) > 0 else np.array([])
        n_trials = len(aligned_spikes_list)
        
        if len(all_aligned) == 0:
            return {
                "artifact_flag": False,
                "onset_guard_spike_count": 0,
                "offset_guard_spike_count": 0,
                "suspicious_short_latency": False,
                "sham_false_positive_rate": 0.0,
                "notes": "No spikes recorded in analysis window"
            }
            
        # Count spikes in onset guard zone [0, 1 ms)
        onset_spikes = np.sum((all_aligned >= self.onset_guard[0]) & (all_aligned < self.onset_guard[1]))
        offset_spikes = np.sum((all_aligned >= self.offset_guard[0]) & (all_aligned <= self.offset_guard[1]))
        
        # Calculate evoked first latencies
        evoked_first_latencies = []
        sham_has_spike_count = 0
        
        for trial_spikes in aligned_spikes_list:
            # Sham check [-18ms, -10ms]
            sham_mask = (trial_spikes >= self.sham_win[0]) & (trial_spikes < self.sham_win[1])
            if np.any(sham_mask):
                sham_has_spike_count += 1
                
            # Evoked check [+1ms, +9ms]
            evoked_mask = (trial_spikes >= self.evoked_win[0]) & (trial_spikes < self.evoked_win[1])
            if np.any(evoked_mask):
                evoked_first_latencies.append(trial_spikes[evoked_mask][0])
                
        sham_fpr = sham_has_spike_count / max(1, n_trials)
        
        suspicious_short_latency = False
        notes = []
        
        if len(evoked_first_latencies) >= 5:
            med_lat_ms = np.median(evoked_first_latencies) * 1000.0
            sd_lat_ms = np.std(evoked_first_latencies) * 1000.0
            
            # If spikes occur < 1.2 ms with near-zero jitter (< 0.1 ms), suspicious
            if med_lat_ms < 1.2 and sd_lat_ms < 0.10:
                suspicious_short_latency = True
                notes.append(f"Suspicious sub-1.2ms latency ({med_lat_ms:.2f}ms +/- {sd_lat_ms:.2f}ms)")
                
        # Excessive onset guard zone spikes compared to baseline
        onset_ratio = onset_spikes / max(1, n_trials)
        if onset_ratio > 0.5 and suspicious_short_latency:
            notes.append(f"High onset guard contamination ({onset_spikes} spikes)")
            
        artifact_flag = suspicious_short_latency or (onset_ratio > 0.8)
        
        record = {
            "session_id": session_id,
            "unit_id": unit_id,
            "n_trials": n_trials,
            "onset_guard_spike_count": int(onset_spikes),
            "offset_guard_spike_count": int(offset_spikes),
            "sham_false_positive_rate": float(np.round(sham_fpr, 4)),
            "suspicious_short_latency": suspicious_short_latency,
            "artifact_flag": artifact_flag,
            "notes": "; ".join(notes) if notes else "Clean"
        }
        
        if artifact_flag or len(notes) > 0:
            self.exclusions.append(record)
            
        return record

    def save_exclusion_log(self, filepath: Optional[str] = None):
        """Save exclusion and artifact flags to CSV."""
        if filepath is None:
            filepath = self.config["paths"]["exclusion_log"]
            
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if len(self.exclusions) > 0:
            df = pd.DataFrame(self.exclusions)
        else:
            df = pd.DataFrame(columns=[
                "session_id", "unit_id", "n_trials",
                "onset_guard_spike_count", "offset_guard_spike_count",
                "sham_false_positive_rate", "suspicious_short_latency",
                "artifact_flag", "notes"
            ])
            
        df.to_csv(path, index=False)
        logger.info("Saved exclusion log (%d records) to %s", len(df), path)
        return df

"""
opto_trials.py
==============
Extraction, classification, and parsing of optogenetic stimulation trials
from Allen Neuropixels sessions.

Dynamically adapts to the schema of session.optogenetic_stimulation_epochs.
Identifies primary (single 10-ms pulses) and secondary stimulation conditions
(5-ms pulses, 2.5-ms pulse trains, 1-s raised-cosine ramps).
"""

from typing import Dict, Any, List, Optional, Tuple
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def inspect_opto_table_schema(opto_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Introspect the optogenetic stimulation epochs DataFrame schema.
    Returns column mapping and unique stimulus descriptions.
    """
    cols = list(opto_df.columns)
    logger.info("Opto epochs table columns: %s", cols)
    
    # Identify time columns
    start_col = next((c for c in cols if "start" in c.lower()), None)
    stop_col = next((c for c in cols if "stop" in c.lower() or "end" in c.lower()), None)
    duration_col = next((c for c in cols if "duration" in c.lower()), None)
    
    # Identify condition / name / stimulus columns
    condition_col = next((c for c in cols if any(k in c.lower() for k in ["condition", "stimulus", "name", "type"])), None)
    
    # Identify optical level / power column
    level_col = next((c for c in cols if any(k in c.lower() for k in ["level", "power", "intensity", "amplitude"])), None)
    
    return {
        "columns": cols,
        "start_col": start_col,
        "stop_col": stop_col,
        "duration_col": duration_col,
        "condition_col": condition_col,
        "level_col": level_col,
        "n_rows": len(opto_df)
    }


def parse_opto_trials(opto_df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize optogenetic stimulation epochs into a consistent trial DataFrame.
    
    Standardized columns:
    - trial_id: int
    - start_time: float (seconds)
    - stop_time: float (seconds)
    - duration: float (seconds)
    - condition_raw: str
    - stimulus_type: str ('10ms_pulse', '5ms_pulse', '2.5ms_train', '1s_ramp', 'other')
    - optical_level: float / str
    """
    if opto_df is None or len(opto_df) == 0:
        logger.warning("Empty optogenetic stimulation epochs table provided.")
        return pd.DataFrame(columns=[
            "trial_id", "start_time", "stop_time", "duration",
            "condition_raw", "stimulus_type", "optical_level"
        ])
    
    df = opto_df.copy()
    schema = inspect_opto_table_schema(df)
    
    # 1. Start & Stop time
    start_col = schema["start_col"] or "start_time"
    stop_col = schema["stop_col"] or "stop_time"
    
    if start_col not in df.columns:
        raise ValueError(f"Could not identify start_time column in opto table. Columns: {df.columns}")
        
    start_times = df[start_col].values.astype(float)
    if stop_col in df.columns:
        stop_times = df[stop_col].values.astype(float)
    else:
        stop_times = start_times + 0.010  # fallback assumption
        
    # 2. Duration
    if schema["duration_col"] and schema["duration_col"] in df.columns:
        durations = df[schema["duration_col"]].values.astype(float)
    else:
        durations = np.round(stop_times - start_times, 5)
        
    # 3. Raw condition
    cond_col = schema["condition_col"]
    if cond_col and cond_col in df.columns:
        raw_conds = df[cond_col].astype(str).values
    else:
        raw_conds = np.array(["unknown"] * len(df))
        
    # 4. Optical level / power
    lvl_col = schema["level_col"]
    if lvl_col and lvl_col in df.columns:
        levels = df[lvl_col].values
    else:
        levels = np.array([1.0] * len(df))
        
    # 5. Classify stimulus type
    stim_names = df["stimulus_name"].astype(str).str.lower().values if "stimulus_name" in df.columns else [""] * len(df)
    stim_types = []
    for i in range(len(df)):
        dur = durations[i]
        cond_str = raw_conds[i].lower()
        s_name = stim_names[i]
        
        # Check 2.5ms train or 10 Hz train (must precede 5ms pulse check to avoid '2.5 ms' substring match)
        if ("fast_pulses" in s_name) or ("2.5" in cond_str) or ("train" in cond_str) or ("10 hz" in cond_str):
            stim_types.append("2.5ms_train")
        # Check single 10ms pulse
        elif (np.isclose(dur, 0.010, atol=0.002)) or ("10 ms" in cond_str and "train" not in cond_str) or ("10ms" in cond_str and "train" not in cond_str):
            stim_types.append("10ms_pulse")
        # Check single 5ms pulse
        elif (np.isclose(dur, 0.005, atol=0.0015)) or (("5 ms" in cond_str or "5ms" in cond_str) and "2.5" not in cond_str and "train" not in cond_str):
            stim_types.append("5ms_pulse")
        # Check 1s raised cosine / ramp
        elif ("raised_cosine" in s_name) or ("cosine" in cond_str) or ("ramp" in cond_str) or (np.isclose(dur, 1.0, atol=0.05) and "cosine" in cond_str):
            stim_types.append("1s_ramp")
        else:
            stim_types.append("other")
            
    standardized = pd.DataFrame({
        "trial_id": np.arange(len(df)),
        "start_time": start_times,
        "stop_time": stop_times,
        "duration": durations,
        "condition_raw": raw_conds,
        "stimulus_type": stim_types,
        "optical_level": levels
    })
    
    counts = standardized["stimulus_type"].value_counts().to_dict()
    logger.info("Parsed opto trials: Total=%d, Stimulus types=%s", len(standardized), counts)
    return standardized


def get_trials_by_type(standardized_trials: pd.DataFrame, stimulus_type: str = "10ms_pulse") -> pd.DataFrame:
    """Filter trials for a specific stimulus type."""
    subset = standardized_trials[standardized_trials["stimulus_type"] == stimulus_type].copy()
    subset.reset_index(drop=True, inplace=True)
    return subset


def summarize_opto_conditions(standardized_trials: pd.DataFrame) -> Dict[str, Any]:
    """Generate high-level summary of available optical stimulation conditions."""
    if standardized_trials.empty:
        return {
            "has_10ms_pulse": False,
            "has_5ms_pulse": False,
            "has_2p5ms_train": False,
            "has_1s_ramp": False,
            "optical_levels": [],
            "trial_counts": {}
        }
        
    counts = standardized_trials["stimulus_type"].value_counts().to_dict()
    unique_levels = list(standardized_trials["optical_level"].unique())
    
    return {
        "has_10ms_pulse": counts.get("10ms_pulse", 0) > 0,
        "has_5ms_pulse": counts.get("5ms_pulse", 0) > 0,
        "has_2p5ms_train": counts.get("2.5ms_train", 0) > 0,
        "has_1s_ramp": counts.get("1s_ramp", 0) > 0,
        "optical_levels": unique_levels,
        "trial_counts": counts
    }

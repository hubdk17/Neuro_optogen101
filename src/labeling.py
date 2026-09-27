"""
labeling.py
===========
Operational rule-based reference labeling and threshold sensitivity analysis.

IMPORTANT:
These labels are OPERATIONAL REFERENCE CRITERIA derived from electrophysiological
heuristics. They do NOT represent ground-truth biological cell identity.

Classes:
- 'putatively directly optotagged'
- 'light-responsive / indirect or uncertain'
- 'not light responsive'
- 'insufficient evidence'
"""

from typing import Dict, Any, List, Optional, Tuple
import logging
from itertools import product
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CLASS_DIRECT = "putatively directly optotagged"
CLASS_INDIRECT = "light-responsive / indirect or uncertain"
CLASS_NON_RESPONSIVE = "not light responsive"
CLASS_INSUFFICIENT = "insufficient evidence"


def assign_reference_label(
    row: pd.Series,
    lat_thresh_ms: float = 8.0,
    rel_thresh: float = 0.30,
    mod_thresh: float = 2.0,
    p_thresh: float = 0.05
) -> Tuple[str, float, str]:
    """
    Assign operational reference label for a single unit according to configured thresholds.
    
    Returns:
    --------
    (reference_class, label_confidence, label_reasons)
    """
    n_trials = row.get("n_trials", 0)
    baseline_rate = row.get("baseline_rate", 0.0)
    evoked_rate = row.get("evoked_rate", 0.0)
    mod_ratio = row.get("modulation_ratio", 0.0)
    med_lat = row.get("median_latency_ms", np.nan)
    lat_sd = row.get("latency_sd_ms", np.nan)
    rel = row.get("trial_reliability", 0.0)
    p_val = row.get("p_value", 1.0)
    eff_size = row.get("effect_size", 0.0)
    artifact_flag = bool(row.get("artifact_flag", False))
    
    # Check for insufficient data
    if n_trials < 10 or (baseline_rate == 0.0 and evoked_rate == 0.0):
        return CLASS_INSUFFICIENT, 0.0, "Insufficient trial count (<10) or zero spikes recorded"
        
    # Artifact contamination override
    if artifact_flag:
        return CLASS_INDIRECT, 0.30, "Artifact flag triggered (suspicious timing or photoelectric onset)"
        
    # Statistical significance of response
    is_significant = (p_val < p_thresh) and (eff_size > 0.10)
    
    # Check non-responsive
    if not is_significant or mod_ratio <= 1.0 or rel < 0.05:
        reasons = []
        if not is_significant:
            reasons.append(f"Not statistically significant (p={p_val:.4f} >= {p_thresh})")
        if mod_ratio <= 1.0:
            reasons.append(f"No positive modulation (mod_ratio={mod_ratio:.2f} <= 1.0)")
        if rel < 0.05:
            reasons.append(f"Very low reliability ({rel:.2f} < 0.05)")
        # Confidence higher when p-value is far above threshold and mod is near 1
        conf = float(np.clip(1.0 - rel, 0.5, 0.95))
        return CLASS_NON_RESPONSIVE, conf, "; ".join(reasons)
        
    # Check putatively direct
    passes_latency = (not np.isnan(med_lat)) and (med_lat < lat_thresh_ms)
    passes_reliability = (rel >= rel_thresh)
    passes_modulation = (mod_ratio > mod_thresh)
    
    if passes_latency and passes_reliability and passes_modulation and is_significant:
        reasons = [
            f"Passes direct criteria: mod={mod_ratio:.2f}>{mod_thresh}",
            f"rel={rel:.2f}>={rel_thresh}",
            f"lat={med_lat:.2f}<{lat_thresh_ms}ms",
            f"p={p_val:.4f}<{p_thresh}"
        ]
        # Confidence increases with higher reliability, lower latency, large effect size
        conf_score = 0.5 + 0.25 * min(1.0, rel) + 0.25 * min(1.0, eff_size / 2.0)
        return CLASS_DIRECT, float(np.clip(conf_score, 0.60, 0.95)), "; ".join(reasons)
        
    # Otherwise: Light-responsive / indirect or uncertain
    indirect_reasons = []
    if not passes_latency:
        indirect_reasons.append(f"Delayed latency ({med_lat:.2f}ms >= {lat_thresh_ms}ms)" if not np.isnan(med_lat) else "No evoked spikes for latency")
    if not passes_reliability:
        indirect_reasons.append(f"Sub-threshold reliability ({rel:.2f} < {rel_thresh})")
    if not passes_modulation:
        indirect_reasons.append(f"Sub-threshold modulation ({mod_ratio:.2f} <= {mod_thresh})")
    if not np.isnan(lat_sd) and lat_sd > 3.0:
        indirect_reasons.append(f"High latency jitter (SD={lat_sd:.2f}ms)")
        
    return CLASS_INDIRECT, 0.65, "; ".join(indirect_reasons)


def label_unit_features_table(
    df: pd.DataFrame,
    lat_thresh_ms: float = 8.0,
    rel_thresh: float = 0.30,
    mod_thresh: float = 2.0,
    p_thresh: float = 0.05
) -> pd.DataFrame:
    """Apply operational labeling across all units in a features DataFrame."""
    labeled_df = df.copy()
    classes = []
    confidences = []
    reasons = []
    
    for _, row in labeled_df.iterrows():
        c, conf, r = assign_reference_label(
            row,
            lat_thresh_ms=lat_thresh_ms,
            rel_thresh=rel_thresh,
            mod_thresh=mod_thresh,
            p_thresh=p_thresh
        )
        classes.append(c)
        confidences.append(np.round(conf, 3))
        reasons.append(r)
        
    labeled_df["reference_class"] = classes
    labeled_df["label_confidence"] = confidences
    labeled_df["label_reasons"] = reasons
    return labeled_df


def run_threshold_sensitivity_analysis(
    df: pd.DataFrame,
    latency_sweep: List[float] = [6.0, 8.0, 10.0],
    reliability_sweep: List[float] = [0.20, 0.30, 0.50],
    modulation_sweep: List[float] = [1.5, 2.0, 3.0],
    p_thresh: float = 0.05
) -> pd.DataFrame:
    """
    Perform exhaustive sensitivity analysis across threshold permutations.
    Returns sensitivity summary table showing unit distribution across classes.
    """
    records = []
    total_units = len(df)
    
    for lat_t, rel_t, mod_t in product(latency_sweep, reliability_sweep, modulation_sweep):
        labeled = label_unit_features_table(
            df,
            lat_thresh_ms=lat_t,
            rel_thresh=rel_t,
            mod_thresh=mod_t,
            p_thresh=p_thresh
        )
        counts = labeled["reference_class"].value_counts().to_dict()
        n_direct = counts.get(CLASS_DIRECT, 0)
        n_indirect = counts.get(CLASS_INDIRECT, 0)
        n_non_resp = counts.get(CLASS_NON_RESPONSIVE, 0)
        n_insuff = counts.get(CLASS_INSUFFICIENT, 0)
        
        records.append({
            "latency_threshold_ms": lat_t,
            "reliability_threshold": rel_t,
            "modulation_threshold": mod_t,
            "n_total": total_units,
            "n_putatively_direct": n_direct,
            "pct_putatively_direct": np.round((n_direct / max(1, total_units)) * 100.0, 2),
            "n_indirect_uncertain": n_indirect,
            "pct_indirect_uncertain": np.round((n_indirect / max(1, total_units)) * 100.0, 2),
            "n_not_responsive": n_non_resp,
            "pct_not_responsive": np.round((n_non_resp / max(1, total_units)) * 100.0, 2),
            "n_insufficient": n_insuff
        })
        
    sens_df = pd.DataFrame(records)
    logger.info("Generated sensitivity analysis table with %d parameter combinations", len(sens_df))
    return sens_df

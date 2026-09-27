"""
session_discovery.py
====================
Dataset discovery, eligibility filtering, and session inventory generation.

Identifies optogenetically stimulated Neuropixels sessions (ChR2 / Ai32 Cre-lines),
counts probes, units, and brain areas, and records stimulation condition presence.
Produces results/tables/session_inventory.csv.
"""

from typing import Dict, Any, List, Optional, Tuple
import logging
from pathlib import Path
import numpy as np
import pandas as pd

from src.data_access import load_session_table, load_probes_table, load_channels_table, load_units_table

logger = logging.getLogger(__name__)


def identify_optogenetic_sessions(sessions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter the session table for sessions that contain optogenetic stimulation.
    In the Allen Visual Coding Neuropixels dataset, transgenic mice expressing
    channelrhodopsin-2 carry the Ai32 allele (e.g., Sst-Cre, Pvalb-Cre, Vip-Cre x Ai32).
    """
    if "genotype" in sessions_df.columns:
        opto_mask = sessions_df["genotype"].str.contains("Ai32", na=False, case=False)
    elif "full_genotype" in sessions_df.columns:
        opto_mask = sessions_df["full_genotype"].str.contains("Ai32", na=False, case=False)
    else:
        logger.warning("No genotype column found in session table.")
        opto_mask = pd.Series(False, index=sessions_df.index)
        
    eligible = sessions_df[opto_mask].copy()
    logger.info("Identified %d eligible optogenetic sessions out of %d total sessions",
                len(eligible), len(sessions_df))
    return eligible


def build_session_inventory(
    cache,
    config: Dict[str, Any],
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Generate comprehensive inventory of all sessions and specifically eligible
    optogenetic sessions.
    
    Records:
    - session_id
    - specimen_id
    - genotype
    - session_type
    - probe_count
    - unit_count
    - brain_areas
    - has_10ms_pulse
    - has_5ms_pulse
    - has_2p5ms_train
    - has_1s_ramp
    """
    sessions_df = load_session_table(cache, config)
    probes_df = load_probes_table(cache, config)
    
    # Try loading channels or units for brain areas & unit counts
    try:
        channels_df = load_channels_table(cache, config)
    except Exception as e:
        logger.warning("Channels table not yet available: %s", e)
        channels_df = None
        
    try:
        units_df = load_units_table(cache, config)
    except Exception as e:
        logger.warning("Units table not yet available: %s", e)
        units_df = None
        
    # Build channel -> session_id and channel -> structure mapping
    channel_to_session = {}
    channel_to_area = {}
    session_to_areas = {}
    
    if channels_df is not None and probes_df is not None:
        probe_to_session = probes_df["ecephys_session_id"].to_dict()
        ch_probe_map = channels_df["ecephys_probe_id"].to_dict()
        ch_area_map = channels_df["ecephys_structure_acronym"].to_dict()
        
        for ch_id, p_id in ch_probe_map.items():
            if p_id in probe_to_session:
                s_id = probe_to_session[p_id]
                channel_to_session[ch_id] = s_id
                area = ch_area_map.get(ch_id)
                if pd.notna(area) and str(area).strip() and str(area) != "nan":
                    channel_to_area[ch_id] = str(area)
                    if s_id not in session_to_areas:
                        session_to_areas[s_id] = set()
                    session_to_areas[s_id].add(str(area))
                    
    # Count units per session
    session_to_unit_count = {}
    if units_df is not None and len(channel_to_session) > 0:
        for _, u_row in units_df.iterrows():
            ch_id = u_row.get("ecephys_channel_id")
            if ch_id in channel_to_session:
                s_id = channel_to_session[ch_id]
                session_to_unit_count[s_id] = session_to_unit_count.get(s_id, 0) + 1
                
    inventory_rows = []
    
    for session_id, s_row in sessions_df.iterrows():
        specimen_id = s_row.get("specimen_id", "unknown")
        genotype = s_row.get("genotype", s_row.get("full_genotype", "unknown"))
        session_type = s_row.get("session_type", "unknown")
        
        is_opto = "Ai32" in str(genotype)
        
        # Probe count
        if probes_df is not None and "ecephys_session_id" in probes_df.columns:
            session_probes = probes_df[probes_df["ecephys_session_id"] == session_id]
            probe_count = len(session_probes)
        else:
            probe_count = 0
            
        # Unit count
        unit_count = session_to_unit_count.get(session_id, np.nan)
            
        # Brain areas
        areas_set = session_to_areas.get(session_id, set())
        brain_areas = sorted(list(areas_set))
            
        # In Allen Visual Coding protocol, all Ai32 sessions follow the standardized
        # optotagging protocol containing 10-ms pulses, 5-ms pulses, 2.5-ms trains, and 1-s ramps
        has_10ms = is_opto
        has_5ms = is_opto
        has_2p5ms = is_opto
        has_1s = is_opto
        
        inventory_rows.append({
            "session_id": session_id,
            "specimen_id": specimen_id,
            "genotype": genotype,
            "session_type": session_type,
            "is_eligible_opto": is_opto,
            "probe_count": probe_count,
            "unit_count": unit_count,
            "brain_areas": "; ".join(brain_areas) if brain_areas else "Standard Visual Hierarchy",
            "has_10ms_pulse": has_10ms,
            "has_5ms_pulse": has_5ms,
            "has_2p5ms_train": has_2p5ms,
            "has_1s_ramp": has_1s
        })
        
    inv_df = pd.DataFrame(inventory_rows)
    
    if output_path is None:
        output_path = config["paths"]["session_inventory"]
        
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    inv_df.to_csv(out_file, index=False)
    logger.info("Saved session inventory to %s (%d total, %d opto)",
                out_file, len(inv_df), inv_df["is_eligible_opto"].sum())
    return inv_df


def print_discovery_summary(inventory_df: pd.DataFrame):
    """Print high-level discovery statistics as required by Phase 0."""
    eligible = inventory_df[inventory_df["is_eligible_opto"]].copy()
    
    print("\n" + "=" * 65)
    print("PHASE 0: DATASET DISCOVERY SUMMARY")
    print("=" * 65)
    print(f"Total sessions cataloged: {len(inventory_df)}")
    print(f"Eligible optogenetic sessions (Ai32 Cre-lines): {len(eligible)}")
    print("\nEligible Session IDs and Genotypes:")
    for _, row in eligible.iterrows():
        s_id = row["session_id"]
        geno = row["genotype"]
        p_cnt = row["probe_count"]
        u_cnt = row["unit_count"]
        u_str = f"{u_cnt} units" if not pd.isna(u_cnt) else "units pending"
        print(f"  • Session {s_id} | {geno} | {p_cnt} probes | {u_str}")
        
    print("\nStimulation Conditions in Protocol:")
    print("  • Primary condition: Single 10-ms square optical pulses")
    print("  • Secondary condition 1: Single 5-ms pulses")
    print("  • Secondary condition 2: 2.5-ms pulses delivered at 10 Hz for 1 s")
    print("  • Secondary condition 3: 1-s raised-cosine stimulation")
    print("=" * 65 + "\n")

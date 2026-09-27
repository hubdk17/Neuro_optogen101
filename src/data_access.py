"""
data_access.py
==============
Data access and caching layer for Allen Institute Neuropixels Visual Coding
electrophysiology and optogenetics dataset.

Uses AllenSDK EcephysProjectCache with local caching of metadata tables
and controlled incremental downloading of session NWB files.
"""

from pathlib import Path
import os
import sys
import yaml
import logging
import pandas as pd
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load research pipeline configuration from YAML file."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def get_cache_manifest_path(config: Dict[str, Any]) -> str:
    """Resolve absolute manifest path from configuration."""
    manifest_rel = config["paths"]["manifest_path"]
    return str(Path(manifest_rel).resolve())


def initialize_cache(config: Dict[str, Any]):
    """
    Initialize AllenSDK EcephysProjectCache with the configured manifest path.
    Defers import of allensdk to allow modular testing.
    """
    from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
    
    manifest_path = get_cache_manifest_path(config)
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    logger.info("Initializing EcephysProjectCache with manifest: %s", manifest_path)
    cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
    return cache


def load_session_table(cache, config: Dict[str, Any], force_refresh: bool = False) -> pd.DataFrame:
    """Retrieve session metadata table with local caching."""
    metadata_dir = Path(config["paths"]["metadata_dir"])
    metadata_dir.mkdir(parents=True, exist_ok=True)
    local_csv = metadata_dir / "sessions.csv"
    raw_csv = Path(config["paths"]["raw_dir"]) / "sessions.csv"
    
    if not force_refresh:
        if local_csv.exists():
            return pd.read_csv(local_csv, index_col="id")
        if raw_csv.exists():
            df = pd.read_csv(raw_csv, index_col="id")
            df.to_csv(local_csv)
            return df
        
    logger.info("Querying session table from cache/warehouse...")
    sessions_df = cache.get_session_table()
    sessions_df.to_csv(local_csv)
    return sessions_df


def load_probes_table(cache, config: Dict[str, Any], force_refresh: bool = False) -> pd.DataFrame:
    """Retrieve probes metadata table with local caching."""
    metadata_dir = Path(config["paths"]["metadata_dir"])
    metadata_dir.mkdir(parents=True, exist_ok=True)
    local_csv = metadata_dir / "probes.csv"
    raw_csv = Path(config["paths"]["raw_dir"]) / "probes.csv"
    
    if not force_refresh:
        if local_csv.exists():
            return pd.read_csv(local_csv, index_col="id")
        if raw_csv.exists():
            df = pd.read_csv(raw_csv, index_col="id")
            df.to_csv(local_csv)
            return df
            
    probes_df = cache.get_probes()
    probes_df.to_csv(local_csv)
    return probes_df


def load_channels_table(cache, config: Dict[str, Any], force_refresh: bool = False) -> Optional[pd.DataFrame]:
    """Retrieve channels metadata table if locally cached, without blocking on remote warehouse."""
    metadata_dir = Path(config["paths"]["metadata_dir"])
    local_csv = metadata_dir / "channels.csv"
    raw_csv = Path(config["paths"]["raw_dir"]) / "channels.csv"
    
    if not force_refresh:
        if local_csv.exists():
            return pd.read_csv(local_csv, index_col="id")
        if raw_csv.exists():
            df = pd.read_csv(raw_csv, index_col="id")
            df.to_csv(local_csv)
            return df
    return None


def load_units_table(cache, config: Dict[str, Any], force_refresh: bool = False) -> Optional[pd.DataFrame]:
    """Retrieve units metadata table if locally cached, without blocking on remote warehouse."""
    metadata_dir = Path(config["paths"]["metadata_dir"])
    local_csv = metadata_dir / "units.csv"
    raw_csv = Path(config["paths"]["raw_dir"]) / "units.csv"
    
    if not force_refresh:
        if local_csv.exists():
            return pd.read_csv(local_csv, index_col="id")
        if raw_csv.exists():
            df = pd.read_csv(raw_csv, index_col="id")
            df.to_csv(local_csv)
            return df
    return None


def estimate_session_storage(session_id: int) -> Dict[str, Any]:
    """
    Provide estimated storage and memory requirements for loading a single session.
    Allen Neuropixels Visual Coding NWB sessions are approximately 1.8 to 2.8 GB on disk.
    In-memory expansion for spike times and unit structures is approximately 1.5 to 3.0 GB RAM.
    """
    return {
        "session_id": session_id,
        "estimated_disk_gb": 2.5,
        "estimated_ram_gb": 3.0,
        "safe_cpu_cores": 1
    }


def download_session_from_s3(session_id: int, dest_path: str):
    """
    Download session NWB file directly from Allen Institute AWS S3 bucket at high speed
    using multi-threaded parallel chunk transfers.
    """
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    from boto3.s3.transfer import TransferConfig
    from tqdm import tqdm
    
    bucket = "allen-brain-observatory"
    s3_key = f"visual-coding-neuropixels/ecephys-cache/session_{session_id}/session_{session_id}.nwb"
    
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    if dest.exists() and dest.stat().st_size > 100 * 1024 * 1024:
        logger.info("Session %s already downloaded at %s (%.2f MB)",
                    session_id, dest, dest.stat().st_size / (1024 * 1024))
        return dest
        
    logger.info("Downloading session %s from s3://%s/%s ...", session_id, bucket, s3_key)
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    
    head = s3.head_object(Bucket=bucket, Key=s3_key)
    total_size = head["ContentLength"]
    
    transfer_config = TransferConfig(
        multipart_threshold=16 * 1024 * 1024,
        max_concurrency=25,
        multipart_chunksize=16 * 1024 * 1024,
        use_threads=True
    )
    
    part_path = str(dest) + ".download"
    if os.path.exists(part_path):
        try:
            os.unlink(part_path)
        except OSError:
            pass
            
    with tqdm(total=total_size, unit="B", unit_scale=True, desc=f"Session {session_id} NWB") as pbar:
        def update_bar(chunk_bytes):
            pbar.update(chunk_bytes)
            
        s3.download_file(bucket, s3_key, part_path, Config=transfer_config, Callback=update_bar)
        
    os.replace(part_path, dest)
    logger.info("Successfully downloaded session %s to %s (%.2f MB)",
                session_id, dest, dest.stat().st_size / (1024 * 1024))
    return dest


def get_session_data(cache, session_id: int):
    """
    Download/load a single session's NWB object using fast S3 download
    and EcephysSession.from_nwb_path direct loader.
    """
    from allensdk.brain_observatory.ecephys.ecephys_session import EcephysSession
    
    est = estimate_session_storage(session_id)
    logger.info("Requesting session %s (Est disk: ~%.1f GB, Est RAM: ~%.1f GB)",
                session_id, est["estimated_disk_gb"], est["estimated_ram_gb"])
                
    expected_path = cache.get_cache_path(None, cache.SESSION_NWB_KEY, session_id, session_id)
    if not os.path.exists(expected_path) or os.path.getsize(expected_path) < 100 * 1024 * 1024:
        download_session_from_s3(session_id, expected_path)
        
    logger.info("Loading EcephysSession from %s ...", expected_path)
    session = EcephysSession.from_nwb_path(expected_path)
    logger.info("Successfully loaded session %s (units: %d)", session_id, len(session.units))
    return session

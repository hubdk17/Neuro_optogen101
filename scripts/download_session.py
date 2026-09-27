"""
download_session.py
===================
Downloads a session NWB file directly from Allen Institute AWS S3 bucket
with integrity checks and progress reporting.
"""

import sys
import os
import time
from pathlib import Path
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from boto3.s3.transfer import TransferConfig
from tqdm import tqdm
import h5py

def download_session(session_id: int):
    bucket = "allen-brain-observatory"
    s3_key = f"visual-coding-neuropixels/ecephys-cache/session_{session_id}/session_{session_id}.nwb"
    
    dest_dir = Path(f"data/raw/session_{session_id}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_file = dest_dir / f"session_{session_id}.nwb"
    part_file = dest_dir / f"session_{session_id}.nwb.part"
    
    if dest_file.exists() and dest_file.stat().st_size > 100 * 1024 * 1024:
        print(f"Session {session_id} already exists at {dest_file} ({dest_file.stat().st_size / (1024**3):.2f} GB)")
        try:
            with h5py.File(dest_file, "r") as f:
                print(f"Verified HDF5 integrity: {list(f.keys())}")
            return True, dest_file.stat().st_size, "Cached & Verified"
        except Exception as e:
            print(f"Corrupt existing file, will re-download: {e}")
            
    print(f"Connecting to AWS S3 (s3://{bucket}/{s3_key}) ...")
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED, max_pool_connections=20))
    
    try:
        head = s3.head_object(Bucket=bucket, Key=s3_key)
        total_size = head["ContentLength"]
        print(f"Remote file size: {total_size} bytes ({total_size / (1024**3):.2f} GB)")
    except Exception as e:
        print(f"Failed to query S3 object: {e}")
        return False, 0, str(e)
        
    transfer_config = TransferConfig(
        multipart_threshold=8 * 1024 * 1024,
        max_concurrency=10,
        multipart_chunksize=8 * 1024 * 1024,
        use_threads=True
    )
    
    t0 = time.time()
    try:
        with tqdm(total=total_size, unit="B", unit_scale=True, desc=f"Session {session_id}") as pbar:
            def update_bar(chunk_bytes):
                pbar.update(chunk_bytes)
            s3.download_file(bucket, s3_key, str(part_file), Config=transfer_config, Callback=update_bar)
            
        dt = time.time() - t0
        print(f"\nDownload completed in {dt:.1f}s ({total_size / (1024*1024) / dt:.2f} MB/s)")
        
        # Verify HDF5 integrity
        print("Verifying HDF5 file integrity...")
        with h5py.File(str(part_file), "r") as f:
            keys = list(f.keys())
            print(f"HDF5 verification successful. Root keys: {keys}")
            
        # Atomically rename
        if dest_file.exists():
            dest_file.unlink()
        os.replace(str(part_file), str(dest_file))
        print(f"Successfully verified and saved to {dest_file}")
        return True, total_size, f"Downloaded in {dt:.1f}s"
    except Exception as e:
        print(f"Download/verification failed: {e}")
        return False, 0, str(e)

if __name__ == "__main__":
    sid = int(sys.argv[1]) if len(sys.argv) > 1 else 746083955
    download_session(sid)

import os
import sys
import json
import h5py
import pandas as pd
import numpy as np

def probe():
    df = pd.read_parquet('results/ml_final/master_ml_dataset_28spec.parquet')
    print(f"Master units: {len(df)}")
    print(f"Sessions: {df['session_id'].nunique()}")
    print(f"Sample response_at_each_light_level:\n{df['response_at_each_light_level'].iloc[0]}")
    
    # Check non-null values
    print("\nNon-null counts:")
    for c in ['baseline_rate', 'evoked_rate', 'modulation_ratio', 'median_latency_ms',
              'trial_reliability', 'intensity_slope', 'adaptation_index', 'operational_label', 'evidence_score']:
        print(f"  {c}: {df[c].notnull().sum()} / {len(df)}")
        
    # Check NWB files
    raw_dir = 'data/raw'
    sessions = sorted([s for s in os.listdir(raw_dir) if s.startswith('session_')])
    print(f"\nRaw sessions on disk: {len(sessions)}")
    
    # Check sample NWB
    sample_nwb = os.path.join(raw_dir, sessions[0], f"{sessions[0]}.nwb")
    print(f"Inspecting sample NWB: {sample_nwb} (size: {os.path.getsize(sample_nwb)/(1024**3):.2f} GB)")
    
    with h5py.File(sample_nwb, 'r') as f:
        print("Root keys:", list(f.keys()))
        if 'intervals' in f:
            print("Intervals keys:", list(f['intervals'].keys()))
        if 'processing' in f:
            print("Processing keys:", list(f['processing'].keys()))
        if 'units' in f:
            print("Units keys:", list(f['units'].keys()))

if __name__ == '__main__':
    probe()

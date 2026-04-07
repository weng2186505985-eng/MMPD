import pandas as pd
import os
import pickle
import numpy as np

def verify_preprocessing(data_root, metadata_file):
    print(f"--- Verification for {metadata_file} ---")
    if not os.path.exists(metadata_file):
        print("Error: Metadata file not found.")
        return
        
    with open(metadata_file, 'rb') as f:
        meta = pickle.load(f)
        
    tm_path = os.path.join(os.path.dirname(metadata_file), 'mission1_tm.mmap')
    tc_path = os.path.join(os.path.dirname(metadata_file), 'mission1_tc.mmap')
    label_path = os.path.join(os.path.dirname(metadata_file), 'mission1_labels.mmap')
    
    tm_shape = meta['tm_shape']
    tc_shape = meta['tc_shape']
    tm_features = meta['tm_features']
    tc_features = meta['tc_features']
    
    print(f"Time steps: {tm_shape[0]}")
    print(f"TM features: {len(tm_features)} (Shape: {tm_shape})")
    print(f"TC features: {len(tc_features)} (Shape: {tc_shape})")
    
    # Check TM memmap
    tm_mmap = np.memmap(tm_path, dtype='float32', mode='r', shape=tm_shape)
    print(f"TM Sample (first row): {tm_mmap[0, :5]}")
    print(f"TM NaN count: {np.isnan(tm_mmap).sum()}")
    
    # Check TC memmap
    if tc_shape[1] > 0:
        tc_mmap = np.memmap(tc_path, dtype='int8', mode='r', shape=tc_shape)
        print(f"TC Sample (first row): {tc_mmap[0, :5]}")
    
    # Check labels
    label_mmap = np.memmap(label_path, dtype='int8', mode='r', shape=(tm_shape[0],))
    print(f"Anomaly ratio: {label_mmap.mean():.4f}")
    
    print("\n--- Split Check ---")
    time_index = meta['time_index']
    test_mask = time_index >= meta['test_start_date']
    print(f"Test split date: {meta['test_start_date']}")
    print(f"Train/Val steps: {time_index[~test_mask].shape[0]}")
    print(f"Test steps:      {time_index[test_mask].shape[0]}")

    print("\nVerification complete.")

if __name__ == "__main__":
    verify_preprocessing(
        data_root='D:/bishe/ESA/data/ESA-Mission1/',
        metadata_file='D:/bishe/ESA/processed_data/mission1_metadata.pkl'
    )

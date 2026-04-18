import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import pickle

def preprocess_mission1_compact(data_root, output_dir, freq='1min'):
    channels_csv = os.path.join(data_root, 'channels.csv')
    labels_csv = os.path.join(data_root, 'labels.csv')
    telecommands_csv = os.path.join(data_root, 'telecommands.csv')
    
    channels_df = pd.read_csv(channels_csv)
    tcs_meta_df = pd.read_csv(telecommands_csv)
    
    os.makedirs(output_dir, exist_ok=True)
    
    start_time = pd.Timestamp('2000-01-01 00:00:00')
    end_time = pd.Timestamp('2013-12-31 23:59:00')
    global_index = pd.date_range(start=start_time, end=end_time, freq=freq)
    num_timestamps = len(global_index)
    
    # 1. Screen Channels
    valid_tm = []
    for c in tqdm(channels_df['Channel'], desc="Screening TM"):
        path = os.path.join(data_root, 'channels', c, c)
        if os.path.exists(path): valid_tm.append(c)
            
    # 2. Allocate TM Memmap (float32)
    tm_mmap_path = os.path.join(output_dir, 'mission1_tm.mmap')
    tm_mmap = np.memmap(tm_mmap_path, dtype='float32', mode='w+', shape=(num_timestamps, len(valid_tm)))
    tm_mmap[:] = np.nan
    
    used_tm = []
    tm_indices = []
    print("Processing TM...")
    for i, c in enumerate(tqdm(valid_tm)):
        path = os.path.join(data_root, 'channels', c, c)
        try:
            df = pd.read_pickle(path)
            resampled = df.resample(freq).mean().reindex(global_index).interpolate(method='linear', limit_direction='both').fillna(0)
            if resampled.isnull().mean().values[0] < 0.3:
                tm_mmap[:, i] = resampled.iloc[:, 0].values.astype(np.float32)
                used_tm.append(c)
                tm_indices.append(i)
        except: pass
    
    # Fill gaps for valid TM
    for i in tqdm(tm_indices, desc="Filling TM gaps"):
        col = pd.Series(tm_mmap[:, i]).interpolate(method='linear', limit_direction='both').fillna(0)
        tm_mmap[:, i] = col.values
    
    # 3. Screen and Process TC (int8)
    valid_tc = []
    tc_events = []
    print("Screening and Processing TC...")
    for tc in tqdm(tcs_meta_df['Telecommand']):
        path = os.path.join(data_root, 'telecommands', tc, tc)
        if os.path.exists(path):
            try:
                df_tc = pd.read_pickle(path)
                if len(df_tc) > 0:
                    resampled = df_tc.resample(freq).count().reindex(global_index).fillna(0)
                    binary = (resampled.iloc[:, 0] > 0).astype(np.int8)
                    if binary.sum() > 0:
                        valid_tc.append(tc)
                        tc_events.append(binary.values)
            except: pass
            
    tc_mmap_path = os.path.join(output_dir, 'mission1_tc.mmap')
    if valid_tc:
        tc_mmap = np.memmap(tc_mmap_path, dtype='int8', mode='w+', shape=(num_timestamps, len(valid_tc)))
        for i, events in enumerate(tc_events):
            tc_mmap[:, i] = events
    else:
        tc_mmap = None

    # 4. Labels
    labels_mmap_path = os.path.join(output_dir, 'mission1_labels.mmap')
    labels_mmap = np.memmap(labels_mmap_path, dtype='int8', mode='w+', shape=(num_timestamps,))
    labels_mmap[:] = 0
    labels_df = pd.read_csv(labels_csv)
    for _, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc="Marking labels"):
        start = pd.to_datetime(row['StartTime']).tz_localize(None).floor(freq)
        end = pd.to_datetime(row['EndTime']).tz_localize(None).ceil(freq)
        s_idx = global_index.get_indexer([start], method='nearest')[0]
        e_idx = global_index.get_indexer([end], method='nearest')[0]
        labels_mmap[s_idx : e_idx + 1] = 1

    # 5. Metadata
    metadata = {
        'time_index': global_index,
        'tm_features': used_tm,
        'tc_features': valid_tc,
        'features': used_tm + valid_tc,
        'tm_shape': (num_timestamps, len(used_tm)),
        'tc_shape': (num_timestamps, len(valid_tc)) if valid_tc else (num_timestamps, 0),
        'tm_indices': tm_indices,
        'test_start_date': pd.Timestamp('2012-01-01')
    }
    with open(os.path.join(output_dir, 'mission1_metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)
        
    tm_mmap.flush()
    if tc_mmap is not None: tc_mmap.flush()
    labels_mmap.flush()
    print("Done. Storage optimized.")

if __name__ == "__main__":
    preprocess_mission1_compact('D:/bishe/ESA/data/ESA-Mission1/', 'D:/bishe/ESA/processed_data/')

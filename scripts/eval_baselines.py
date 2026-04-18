import os
import sys
import argparse
import json
import torch
import torch.nn as nn
import numpy as np
import pickle
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.baselines import PatchTSTBaseline, USADBaseline, LSTMBaseline
from scripts.inference import dynamic_ewma_threshold, evaluate

def eval_baseline(config, model_type):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with open(config['data_path'], 'rb') as f:
        meta = pickle.load(f)
        
    config['enc_in'] = len(meta['features'])
    config['num_tm'] = len(meta['tm_features'])
    config['num_tc'] = len(meta['tc_features'])
    
    if model_type == 'PatchTST':
        model = PatchTSTBaseline(config).to(device)
    elif model_type == 'USAD':
        model = USADBaseline(config).to(device)
    elif model_type == 'LSTM':
        model = LSTMBaseline(config).to(device)
        
    model.load_state_dict(torch.load(f'checkpoints/{model_type}_best.pth', map_location=device))
    model.eval()
    
    base = os.path.dirname(config['data_path'])
    tm_mmap = np.memmap(os.path.join(base, 'mission1_tm.mmap'), dtype='float32', mode='r', shape=meta['tm_shape'])
    tc_mmap = np.memmap(os.path.join(base, 'mission1_tc.mmap'), dtype='int8', mode='r', shape=meta['tc_shape'])
    label_mmap = np.memmap(os.path.join(base, 'mission1_labels.mmap'), dtype='int8', mode='r', shape=(meta['tm_shape'][0],))
    
    time_index = meta['time_index']
    test_mask = time_index >= meta['test_start_date']
    test_indices = np.where(test_mask)[0]
    
    start_idx, end_idx = test_indices[0], test_indices[-1] + 1
    seq_len = config['seq_len']
    pred_len = config['pred_len']
    stride = config.get('infer_stride', pred_len)
    
    num_timestamps = end_idx - start_idx
    anomaly_scores = np.zeros(num_timestamps)
    starts = list(range(0, num_timestamps - seq_len - pred_len + 1, stride))
    
    batch_size = config.get('infer_batch_size', 64)
    criterion = nn.MSELoss(reduction='none')
    
    with torch.no_grad():
        for i in tqdm(range(0, len(starts), batch_size), desc=f"Evaluating {model_type}"):
            batch_starts = starts[i:i+batch_size]
            x_cond_list, x_0_list = [], []
            for start in batch_starts:
                abs_start = start_idx + start
                
                tm_cond = tm_mmap[abs_start : abs_start + seq_len, meta['tm_indices']]
                tm_0 = tm_mmap[abs_start + seq_len : abs_start + seq_len + pred_len, meta['tm_indices']]
                tc_cond = tc_mmap[abs_start : abs_start + seq_len, :].astype(np.float32)
                tc_0 = tc_mmap[abs_start + seq_len : abs_start + seq_len + pred_len, :].astype(np.float32)
                
                x_cond_list.append(np.concatenate([tm_cond, tc_cond], axis=-1))
                x_0_list.append(np.concatenate([tm_0, tc_0], axis=-1))
                
            x_cond = torch.from_numpy(np.array(x_cond_list)).to(device)
            x_0 = torch.from_numpy(np.array(x_0_list)).to(device)
            
            if model_type in ['PatchTST', 'LSTM']:
                pred = model(x_cond)
                x_0_tm = x_0[..., :config['num_tm']]
                mse = criterion(pred, x_0_tm).mean(dim=-1) # [B, pred_len]
                score = mse.mean(dim=-1).cpu().numpy() # [B]
                
            elif model_type == 'USAD':
                w1, w2, w3 = model(x_cond)
                mse = 0.5 * criterion(x_cond, w1) + 0.5 * criterion(x_cond, w2)
                score = mse.mean(dim=(1, 2)).cpu().numpy() # [B]
                
            for b, start in enumerate(batch_starts):
                anomaly_scores[start + seq_len : start + seq_len + pred_len] = np.maximum(
                    anomaly_scores[start + seq_len : start + seq_len + pred_len], score[b]
                )
                
    anomaly_scores = gaussian_filter1d(anomaly_scores, sigma=3)
    test_labels = label_mmap[start_idx : end_idx]
    
    print(f"\n--- {model_type} Results ---")
    evaluate(anomaly_scores, test_labels, time_index[start_idx:end_idx], output_dir=f'results_{model_type}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['PatchTST', 'USAD', 'LSTM'])
    args = parser.parse_args()
    
    with open('config.json', 'r') as f:
        config = json.load(f)
        
    eval_baseline(config, args.model)

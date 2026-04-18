import os
import sys

# Add the project root to sys.path to allow importing from 'scripts'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import pandas as pd
import pickle
import json
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
from sklearn.mixture import GaussianMixture
from scripts.mmpd_model import MMPD

def run_inference(config, model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Metadata and Memmaps
    with open(config['data_path'], 'rb') as f:
        meta = pickle.load(f)
    
    base = os.path.dirname(config['data_path'])
    tm_mmap = np.memmap(os.path.join(base, 'mission1_tm.mmap'), dtype='float32', mode='r', shape=meta['tm_shape'])
    tc_mmap = np.memmap(os.path.join(base, 'mission1_tc.mmap'), dtype='int8', mode='r', shape=meta['tc_shape'])
    label_mmap = np.memmap(os.path.join(base, 'mission1_labels.mmap'), dtype='int8', mode='r', shape=(meta['tm_shape'][0],))
    
    time_index = meta['time_index']
    test_mask = time_index >= meta['test_start_date']
    test_indices = np.where(test_mask)[0]
    
    # Slice only test data (Keep it as mmap view)
    # We need a continuous block for windowing. test_indices are continuous due to split.
    start_idx, end_idx = test_indices[0], test_indices[-1] + 1
    
    # 2. Load Model
    config['enc_in'] = len(meta['features'])
    config['num_tm'] = len(meta['tm_features'])
    config['num_tc'] = len(meta['tc_features'])
    model = MMPD(config).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    seq_len = config['seq_len']
    pred_len = config['pred_len']
    stride = config.get('infer_stride', max(1, pred_len // 2))
    
    num_timestamps = end_idx - start_idx
    anomaly_scores_sum = np.zeros(num_timestamps)
    anomaly_scores_cnt = np.zeros(num_timestamps)
    
    # Window starts relative to test set start
    starts = list(range(0, num_timestamps - seq_len - pred_len + 1, stride))
    
    # Stage 1: Rough Scoring
    print(f"Stage 1: Rough Screening (MSE) on {len(starts)} windows...")
    stage1_scores = []
    
    batch_size = config.get('infer_batch_size', 64)
    with torch.no_grad():
        for i in tqdm(range(0, len(starts), batch_size)):
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
            
            mse = model.get_mse_recon(x_cond, x_0)
            B = x_cond.shape[0]
            mse_mean = mse.reshape(B, -1).mean(dim=1).cpu().numpy()
            stage1_scores.extend(mse_mean)
            
    stage1_scores = np.array(stage1_scores)
    mean_s1 = np.mean(stage1_scores)
    std_s1 = np.std(stage1_scores)
    threshold_val = mean_s1 + 3 * std_s1
    to_refine_indices = np.where(stage1_scores >= threshold_val)[0]
    
    # Write Base scores
    print("Writing Stage 1 base scores...")
    for idx, start in enumerate(starts):
        anomaly_scores_sum[start + seq_len : start + seq_len + pred_len] += stage1_scores[idx]
        anomaly_scores_cnt[start + seq_len : start + seq_len + pred_len] += 1

    # Stage 2: Fine Scoring
    print(f"Stage 2: Refining {len(to_refine_indices)} windows...")
    def gmm_log_prob(x_obs, mu, pi, sigma):
        B, K, pred_len, C = mu.shape
        log_probs = []
        for k in range(K):
            mu_k = mu[:, k, :, :]
            diff_sq = ((x_obs - mu_k)**2).sum(dim=1)
            lp_k = -0.5 * pred_len * np.log(2 * np.pi) - pred_len * torch.log(sigma[:, :, k] + 1e-6) - 0.5 * diff_sq / (sigma[:, :, k]**2 + 1e-6)
            log_probs.append(lp_k + torch.log(pi[:, :, k] + 1e-6))
        log_probs = torch.stack(log_probs, dim=2)
        return torch.logsumexp(log_probs, dim=2)

    diff_batch_size = config.get('diff_batch_size', 8)
    for i in tqdm(range(0, len(to_refine_indices), diff_batch_size)):
        batch_indices = to_refine_indices[i:i+diff_batch_size]
        
        x_cond_list, x_obs_list = [], []
        for idx in batch_indices:
            start = starts[idx]
            abs_start = start_idx + start
            
            tm_cond = tm_mmap[abs_start : abs_start + seq_len, meta['tm_indices']]
            tm_0 = tm_mmap[abs_start + seq_len : abs_start + seq_len + pred_len, meta['tm_indices']]
            tc_cond = tc_mmap[abs_start : abs_start + seq_len, :].astype(np.float32)
            tc_0 = tc_mmap[abs_start + seq_len : abs_start + seq_len + pred_len, :].astype(np.float32)
            
            x_cond_list.append(np.concatenate([tm_cond, tc_cond], axis=-1))
            x_obs_list.append(np.concatenate([tm_0, tc_0], axis=-1))
            
        x_cond = torch.from_numpy(np.array(x_cond_list)).to(device)
        x_obs = torch.from_numpy(np.array(x_obs_list)).to(device)
        
        mu_k, pi_k, sigma_k = model.sample(x_cond, num_samples=config.get('num_samples', 50), ddim_steps=20, gmm_K=config.get('gmm_K', 5))
        
        x_obs_tm = x_obs[..., :config['num_tm']]
        lp = gmm_log_prob(x_obs_tm, mu_k, pi_k, sigma_k)
        nll = -lp.mean(dim=1).cpu().numpy()
            
        for b, idx in enumerate(batch_indices):
            start = starts[idx]
            # Replace stage 1 score with stage 2 nll
            anomaly_scores_sum[start + seq_len : start + seq_len + pred_len] -= stage1_scores[idx]
            anomaly_scores_sum[start + seq_len : start + seq_len + pred_len] += nll[b]
            
    anomaly_scores = np.divide(anomaly_scores_sum, anomaly_scores_cnt, out=np.zeros_like(anomaly_scores_sum), where=anomaly_scores_cnt!=0)
    anomaly_scores = gaussian_filter1d(anomaly_scores, sigma=3)
    test_labels = label_mmap[start_idx : end_idx]
    
    return anomaly_scores, test_labels, time_index[start_idx : end_idx]

def dynamic_ewma_threshold(scores, alpha=0.1, z=3.0):
    ewma = np.zeros_like(scores)
    ewma_std = np.zeros_like(scores)
    ewma[0] = scores[0]
    ewma_std[0] = 0
    for i in range(1, len(scores)):
        ewma[i] = alpha * scores[i] + (1 - alpha) * ewma[i-1]
        var = alpha * (scores[i] - ewma[i])**2 + (1 - alpha) * ewma_std[i-1]**2
        ewma_std[i] = np.sqrt(var)
    return ewma + z * ewma_std

def evaluate(scores, labels, times, threshold_pct=99, output_dir='results', pa_k_ratio=0.1):
    from sklearn.metrics import precision_recall_fscore_support
    os.makedirs(output_dir, exist_ok=True)
    
    thresh_dynamic = dynamic_ewma_threshold(scores, alpha=0.1, z=3.0)
    pred = (scores > thresh_dynamic).astype(int)
    
    # Event-wise Correction calculation (Rigorous PA%K)
    def get_events(labels):
        events = []
        if len(labels) == 0: return events
        is_event = False
        start = 0
        for i in range(len(labels)):
            if labels[i] == 1 and not is_event:
                is_event = True
                start = i
            elif labels[i] == 0 and is_event:
                is_event = False
                events.append((start, i))
        if is_event: events.append((start, len(labels)))
        return events

    events = get_events(labels)
    corrected_pred = pred.copy()
    for s, e in events:
        if pred[s:e].mean() >= pa_k_ratio:
            corrected_pred[s:e] = 1
            
    # Calculate Metrics
    p, r, f, _ = precision_recall_fscore_support(labels, corrected_pred, average='binary', beta=0.5, zero_division=0)
    raw_p, raw_r, raw_f, _ = precision_recall_fscore_support(labels, pred, average='binary', beta=0.5, zero_division=0)
    
    summary = (
        f"Dynamic Thresholding (EWMA)\n"
        f"Raw Point-wise Precision: {raw_p:.4f}\n"
        f"Raw Point-wise Recall: {raw_r:.4f}\n"
        f"Raw Point-wise F0.5: {raw_f:.4f}\n"
        f"Rigorous PA (>{pa_k_ratio*100}%) Precision: {p:.4f}\n"
        f"Rigorous PA (>{pa_k_ratio*100}%) Recall: {r:.4f}\n"
        f"Rigorous PA (>{pa_k_ratio*100}%) F0.5: {f:.4f}\n"
    )
    print(summary)
    
    # 1. Save Summary
    with open(os.path.join(output_dir, 'summary.txt'), 'w') as f_out:
        f_out.write(summary)
    
    # 2. Save detailed scores (CSV)
    df = pd.DataFrame({
        'timestamp': times,
        'anomaly_score': scores,
        'prediction': pred,
        'label': labels
    })
    df.to_csv(os.path.join(output_dir, 'anomaly_detection_results.csv'), index=False)
    print(f"Results saved to {output_dir}/ folder.")
    return f

if __name__ == "__main__":
    with open('config.json', 'r') as f:
        config = json.load(f)
    model_file = 'checkpoints/best_mmpd.pth'
    if os.path.exists(model_file):
        scores, labels, times = run_inference(config, model_file)
        evaluate(scores, labels, times)
    else:
        print(f"Model file {model_file} not found. Please train the model first.")

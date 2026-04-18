import os
import sys

# Add the project root to sys.path to allow importing from 'scripts'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import pandas as pd
import pickle
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from scripts.mmpd_model import MMPD

def explain_top_anomaly(config_path='config.json', model_path='checkpoints/best_mmpd.pth', results_csv='results/anomaly_detection_results.csv'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Check dependencies
    missing = []
    if not os.path.exists(config_path): missing.append(config_path)
    if not os.path.exists(model_path): missing.append(model_path)
    if not os.path.exists(results_csv): missing.append(results_csv)
    
    if missing:
        print(f"Missing required files for explainability script: {missing}")
        return

    with open(config_path, 'r') as f:
        config = json.load(f)

    with open(config['data_path'], 'rb') as f:
        meta = pickle.load(f)

    # 2. Find the peak anomaly in the true anomaly regions
    df = pd.read_csv(results_csv)
    true_anomalies = df[df['label'] == 1]
    if len(true_anomalies) == 0:
        print("No true anomalies found in the results.")
        return
        
    peak_idx = true_anomalies['anomaly_score'].idxmax()
    start_idx = max(0, peak_idx - config['seq_len'])
    
    # 3. Load Memmaps
    base = os.path.dirname(config['data_path'])
    tm_mmap = np.memmap(os.path.join(base, 'mission1_tm.mmap'), dtype='float32', mode='r', shape=meta['tm_shape'])
    tc_mmap = np.memmap(os.path.join(base, 'mission1_tc.mmap'), dtype='int8', mode='r', shape=meta['tc_shape'])
    
    time_index = meta['time_index']
    test_mask = time_index >= meta['test_start_date']
    test_indices = np.where(test_mask)[0]
    test_start_abs = test_indices[0]
    
    abs_start = test_start_abs + start_idx
    seq_len = config['seq_len']
    pred_len = config['pred_len']

    # 4. Extract snippet
    tm_cond = tm_mmap[abs_start : abs_start + seq_len, meta['tm_indices']]
    tm_0 = tm_mmap[abs_start + seq_len : abs_start + seq_len + pred_len, meta['tm_indices']]
    tc_cond = tc_mmap[abs_start : abs_start + seq_len, :].astype(np.float32)
    tc_0 = tc_mmap[abs_start + seq_len : abs_start + seq_len + pred_len, :].astype(np.float32)

    x_cond = torch.from_numpy(np.concatenate([tm_cond, tc_cond], axis=-1)).unsqueeze(0).to(device)
    x_obs = torch.from_numpy(np.concatenate([tm_0, tc_0], axis=-1)).to(device)

    # 5. Load Model
    config['enc_in'] = len(meta['features'])
    model = MMPD(config).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # 6. Run Inference
    with torch.no_grad():
        mu_k, pi_k, sigma_k = model.sample(x_cond, num_samples=10, ddim_steps=10, gmm_K=config.get('gmm_K', 5))
        
    # mu_k shape: (K, pred_len, C)
    # We take the most likely GMM component (or the expected value)
    # For simplicity, let's take the mean weighted by pi_k if possible, or just the component with highest pi_k
    pi_k_mean = pi_k.mean(dim=0) # Shape: (K,)
    best_k = torch.argmax(pi_k_mean).item()
    mu_best = mu_k[best_k].cpu().numpy()
    sigma_best = sigma_k[:, best_k].cpu().numpy()
    x_obs_np = x_obs.cpu().numpy()

    # 7. Calculate Feature-wise Errors (MSE)
    mse_per_feature = ((x_obs_np - mu_best)**2).mean(axis=0)
    
    # Get Top 5 contributing features
    feature_names = meta['features']
    top_indices = np.argsort(mse_per_feature)[::-1][:5]
    top_feature_names = [feature_names[i] for i in top_indices]
    top_errors = mse_per_feature[top_indices]

    # --- PLOT 1: Feature Contribution Bar Chart ---
    plt.figure(figsize=(10, 6))
    plt.barh(top_feature_names[::-1], top_errors[::-1], color='coral')
    plt.xlabel('Mean Squared Error')
    plt.title('Top 5 Features Contributing to Peak Anomaly')
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/feature_contribution.png', dpi=300)
    plt.close()
    print("Saved results/feature_contribution.png")

    # --- PLOT 2: Feature Reconstruction Comparison ---
    fig, axes = plt.subplots(len(top_indices), 1, figsize=(12, 2.5 * len(top_indices)), sharex=True)
    if len(top_indices) == 1:
        axes = [axes]
    
    x_cond_np = x_cond.squeeze(0).cpu().numpy()
    time_axis_cond = np.arange(seq_len)
    time_axis_pred = np.arange(seq_len, seq_len + pred_len)
    
    for ax, f_idx, f_name in zip(axes, top_indices, top_feature_names):
        # Plot History (Cond)
        ax.plot(time_axis_cond, x_cond_np[:, f_idx], color='black', label='History (Input)')
        # Plot True Future
        ax.plot(time_axis_pred, x_obs_np[:, f_idx], color='red', label='True Ground Truth', linestyle='--')
        # Plot Predicted Future
        ax.plot(time_axis_pred, mu_best[:, f_idx], color='blue', label='Model Prediction')
        
        # Uncertainty bounds (3 sigma)
        lower_bound = mu_best[:, f_idx] - 3 * sigma_best[f_idx]
        upper_bound = mu_best[:, f_idx] + 3 * sigma_best[f_idx]
        ax.fill_between(time_axis_pred, lower_bound, upper_bound, color='blue', alpha=0.2, label=r'3$\sigma$ Confidence')
        
        ax.set_ylabel('Value')
        ax.set_title(f"Feature: {f_name}")
        ax.grid(True, alpha=0.3)
        if ax == axes[0]:
            ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
            
    axes[-1].set_xlabel('Time Steps')
    plt.tight_layout()
    plt.savefig('results/feature_reconstruction.png', dpi=300)
    plt.close()
    print("Saved results/feature_reconstruction.png")

if __name__ == '__main__':
    explain_top_anomaly()

import torch
import numpy as np
import json
from scripts.mmpd_model import MMPD

def check_fixes():
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    config['enc_in'] = 757
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MMPD(config).to(device)
    model.eval()
    
    # 1. Numerical Consistency Check
    print("--- 1. Numerical Consistency Check ---")
    x_cond = torch.randn(1, config['seq_len'], 757).to(device) * 10.0 + 5.0 # Large scale
    
    # Trigger RevIN stats
    _ = model.revin(x_cond, 'norm')
    
    mu_k, pi_k, sigma_k = model.sample(x_cond, num_samples=10, ddim_steps=5, gmm_K=5)
    
    print(f"Input mean: {x_cond.mean().item():.4f}, std: {x_cond.std().item():.4f}")
    print(f"Sample mu mean: {mu_k.mean().item():.4f}, std: {mu_k.std().item():.4f}")
    
    # 2. Loss Weighting Check
    print("\n--- 2. Loss Weighting Check ---")
    x_0 = torch.randn(1, config['pred_len'], 757).to(device)
    loss_diff, loss_recon = model(x_cond, x_0)
    print(f"loss_diff: {loss_diff.item():.6f}")
    print(f"loss_recon: {loss_recon.item():.6f}")
    print(f"Weighted loss: {(loss_diff + config['lambda_recon'] * loss_recon).item():.6f}")

    # 3. Batch Robustness Check
    print("\n--- 3. Batch Robustness Check ---")
    try:
        x_cond_large = torch.randn(2, config['seq_len'], 757).to(device)
        model.sample(x_cond_large)
    except AssertionError as e:
        print(f"Caught expected assertion: {e}")

if __name__ == "__main__":
    check_fixes()

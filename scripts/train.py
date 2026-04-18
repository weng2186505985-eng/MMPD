import os
import sys

# Add the project root to sys.path to allow importing from 'scripts'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import json
import time
import numpy as np
import random
from tqdm import tqdm
from scripts.mmpd_model import MMPD
from scripts.dataset import get_dataloaders


def set_seed(seed=42):
    """固定随机种子以确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model):
    """统计模型参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def train(config):
    # 固定随机种子
    seed = config.get('seed', 42)
    set_seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 1. Load Data
    train_loader, val_loader, data_meta = get_dataloaders(
        config['data_path'], config
    )
    config['enc_in'] = data_meta['enc_in']
    config['num_tm'] = data_meta['num_tm']
    config['num_tc'] = data_meta['num_tc']
    print(f"Features: TM={config['num_tm']}, TC={config['num_tc']}, Total={config['enc_in']}")
    print(f"Training samples: {len(train_loader.dataset)}, Validation samples: {len(val_loader.dataset)}")
    
    # 2. Model
    model = MMPD(config).to(device)
    config['lr'] = min(config.get('lr', 1e-4), 1e-4) # Clamp to max 1e-4
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    
    # 模型参数统计
    total_params, trainable_params = count_parameters(model)
    print(f"Model Parameters: Total={total_params:,}, Trainable={trainable_params:,}")
    print(f"Model Size: ~{total_params * 4 / 1024 / 1024:.1f} MB (float32)")
    
    # 2.1 Scheduler: Cosine Annealing
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=1e-6)
    
    # 2.2 Mixed Precision
    use_amp = False # Force disable AMP to avoid FP16 overflow
    scaler = GradScaler(enabled=use_amp)
    if use_amp:
        print("Mixed Precision (AMP) enabled.")
    
    # 3. Training History
    history = {
        'config': config,
        'device': str(device),
        'total_params': total_params,
        'trainable_params': trainable_params,
        'seed': seed,
        'epochs': []
    }
    
    # 4. Training Loop
    best_val_metric = float('inf')
    best_metric_name = ""
    patience = config.get('patience', 3)
    patience_counter = 0
    checkpoint_interval = config.get('checkpoint_interval', 5)  # 每N个epoch保存一次checkpoint
    
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"  Training Configuration")
    print(f"{'='*60}")
    print(f"  Epochs:          {config['epochs']}")
    print(f"  Batch Size:      {config['batch_size']}")
    print(f"  Learning Rate:   {config['lr']}")
    print(f"  Patience:        {patience}")
    print(f"  Lambda Recon:    {config.get('lambda_recon', 0.1)}")
    print(f"  Seq Len:         {config['seq_len']}")
    print(f"  Pred Len:        {config['pred_len']}")
    print(f"  Diff Steps:      {config.get('num_diff_steps', 100)}")
    print(f"{'='*60}\n")
    
    total_start_time = time.time()
    
    for epoch in range(config['epochs']):
        epoch_start_time = time.time()
        
        # ==================== Training ====================
        model.train()
        train_total_loss = 0
        train_diff_loss = 0
        train_recon_loss = 0
        train_grad_norms = []
        num_batches = 0
        
        curr_lr = optimizer.param_groups[0]['lr']
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Train]")
        
        for x_cond, x_0, _ in pbar:
            x_cond, x_0 = x_cond.to(device), x_0.to(device)
            
            optimizer.zero_grad()
            
            # Mixed Precision Forward
            with torch.amp.autocast('cuda', enabled=use_amp):
                total_loss, loss_diff, loss_recon = model(x_cond, x_0)
                
            if torch.isnan(loss_diff) or torch.isnan(loss_recon):
                print("\nFatal Error: NaN detected in loss computation!")
                break
            
            # Mixed Precision Backward
            scaler.scale(total_loss).backward()
            
            # Gradient Clipping (unscale first for proper clipping)
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            train_grad_norms.append(grad_norm.item())
            
            scaler.step(optimizer)
            scaler.update()
            
            train_total_loss += total_loss.item()
            train_diff_loss += loss_diff.item()
            train_recon_loss += loss_recon.item()
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'diff': f"{loss_diff.item():.4f}",
                'recon': f"{loss_recon.item():.4f}",
                'grad': f"{grad_norm.item():.3f}"
            })
        
        # Update Scheduler
        scheduler.step()
        
        # ==================== Validation ====================
        model.eval()
        val_total_loss = 0
        val_diff_loss = 0
        val_recon_loss = 0
        val_batches = 0
        
        val_scores = []
        val_labels = []
        
        with torch.no_grad():
            for x_cond, x_0, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Val]", leave=False):
                x_cond, x_0 = x_cond.to(device), x_0.to(device)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    total_loss, ld, lr = model(x_cond, x_0)
                    mse_scores = model.get_mse_recon(x_cond, x_0)
                val_total_loss += total_loss.item()
                val_diff_loss += ld.item()
                val_recon_loss += lr.item()
                val_batches += 1
                
                # mse_scores is [B*num_tm]
                B = x_cond.shape[0]
                mse_scores = mse_scores.reshape(B, -1).max(dim=1)[0]
                val_scores.append(mse_scores.cpu().numpy())
                val_labels.append(labels.max(dim=1)[0].numpy())
        
        # ==================== Epoch Statistics ====================
        epoch_time = time.time() - epoch_start_time
        
        avg_train_total = train_total_loss / num_batches
        avg_train_diff = train_diff_loss / num_batches
        avg_train_recon = train_recon_loss / num_batches
        avg_val_total = val_total_loss / val_batches
        avg_val_diff = val_diff_loss / val_batches
        avg_val_recon = val_recon_loss / val_batches
        avg_grad_norm = np.mean(train_grad_norms)
        max_grad_norm = np.max(train_grad_norms)
        
        # GPU Memory
        gpu_mem_mb = torch.cuda.max_memory_allocated() / 1024**2 if device.type == 'cuda' else 0
        
        # Epoch record
        epoch_record = {
            'epoch': epoch + 1,
            'lr': curr_lr,
            'train_total_loss': avg_train_total,
            'train_diff_loss': avg_train_diff,
            'train_recon_loss': avg_train_recon,
            'val_total_loss': avg_val_total,
            'val_diff_loss': avg_val_diff,
            'val_recon_loss': avg_val_recon,
            'avg_grad_norm': avg_grad_norm,
            'max_grad_norm': max_grad_norm,
            'epoch_time_sec': epoch_time,
            'gpu_mem_mb': gpu_mem_mb,
            'is_best': False
        }
        
        # ==================== Print Summary ====================
        print(f"\n{'─'*60}")
        print(f"  Epoch {epoch+1}/{config['epochs']} Summary  (Time: {epoch_time:.1f}s)")
        print(f"{'─'*60}")
        print(f"  Train Loss:   Total={avg_train_total:.6f}  Diff={avg_train_diff:.6f}  Recon={avg_train_recon:.6f}")
        print(f"  Val Loss:     Total={avg_val_total:.6f}  Diff={avg_val_diff:.6f}  Recon={avg_val_recon:.6f}")
        print(f"  LR={curr_lr:.2e}  AvgGrad={avg_grad_norm:.4f}  MaxGrad={max_grad_norm:.4f}")
        if gpu_mem_mb > 0:
            print(f"  GPU Memory: {gpu_mem_mb:.0f} MB")
        
        val_scores = np.concatenate(val_scores)
        val_labels = np.concatenate(val_labels)
        
        if val_labels.sum() > 0:
            from sklearn.metrics import precision_recall_fscore_support
            thresh = np.percentile(val_scores, 95)
            preds = (val_scores > thresh).astype(int)
            p, r, f05, _ = precision_recall_fscore_support(val_labels, preds, average='binary', beta=0.5, zero_division=0)
            val_metric = -f05 # minimize
            metric_str = f"F0.5={f05:.4f}"
            print(f"  Val Anomalies: {val_labels.sum()} windows. P={p:.4f}, R={r:.4f}, F0.5={f05:.4f}")
        else:
            val_metric = avg_val_total
            metric_str = f"Loss={avg_val_total:.6f}"
            print(f"  Val Anomalies: 0 windows. Using Val Loss for Early Stopping.")
            
        # ==================== Checkpointing ====================
        if val_metric < best_val_metric:
            best_val_metric = val_metric
            best_metric_name = metric_str
            patience_counter = 0
            epoch_record['is_best'] = True
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_metric': best_val_metric,
                'config': config,
            }, 'checkpoints/best_mmpd.pth')
            print(f"  ★ New Best Model Saved! (Val Metric: {metric_str})")
        else:
            patience_counter += 1
            print(f"  EarlyStopping: {patience_counter}/{patience} (Best: {best_metric_name})")
        
        # 定期保存checkpoint
        if (epoch + 1) % checkpoint_interval == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_metric': best_val_metric,
                'config': config,
            }, f'checkpoints/mmpd_epoch_{epoch+1}.pth')
            print(f"  Checkpoint saved: checkpoints/mmpd_epoch_{epoch+1}.pth")
        
        print(f"{'─'*60}\n")
        
        # Log to text file
        with open('train_log.txt', 'a', encoding='utf-8') as f:
            best_tag = " [BEST]" if epoch_record['is_best'] else ""
            f.write(
                f"Epoch {epoch+1}: "
                f"Train={avg_train_total:.6f} (Diff={avg_train_diff:.6f}, Recon={avg_train_recon:.6f}), "
                f"Val={avg_val_total:.6f} (Diff={avg_val_diff:.6f}, Recon={avg_val_recon:.6f}), "
                f"LR={curr_lr:.2e}, GradNorm={avg_grad_norm:.4f}, "
                f"Time={epoch_time:.1f}s{best_tag}\n"
            )
        
        history['epochs'].append(epoch_record)
        
        # 每个 epoch 保存训练历史 JSON（便于可视化脚本实时读取）
        with open('results/train_history.json', 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        
        if patience_counter >= patience:
            print("Early stopping triggered. Training terminated.")
            break
    
    # ==================== Training Complete ====================
    total_time = time.time() - total_start_time
    history['total_training_time_sec'] = total_time
    history['best_val_metric'] = best_val_metric
    
    with open('results/train_history.json', 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"  Training Complete!")
    print(f"{'='*60}")
    print(f"  Total Time:      {total_time/60:.1f} minutes")
    print(f"  Best Val Metric: {best_metric_name}")
    print(f"  Epochs Trained:  {len(history['epochs'])}")
    print(f"  History saved:   results/train_history.json")
    print(f"{'='*60}")


if __name__ == "__main__":
    with open('config.json', 'r') as f:
        config = json.load(f)
    train(config)

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import json
import time
from tqdm import tqdm
from scripts.mmpd_model import MMPD
from scripts.dataset import get_dataloaders

def train(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Data
    train_loader, val_loader, num_features = get_dataloaders(
        config['data_path'], config
    )
    config['enc_in'] = num_features
    
    # 2. Model
    model = MMPD(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    
    # 2.1 Scheduler: Cosine Annealing
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=1e-6)
    
    # 3. Training Loop
    best_val_loss = float('inf')
    patience = config.get('patience', 3)
    patience_counter = 0
    
    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0
        curr_lr = optimizer.param_groups[0]['lr']
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [LR={curr_lr:.2e}]")
        
        for x_cond, x_0 in pbar:
            x_cond, x_0 = x_cond.to(device), x_0.to(device)
            
            optimizer.zero_grad()
            # MMPD loss (DDPM objective + Stage 1 Recon)
            loss_diff, loss_recon = model(x_cond, x_0)
            
            # Bug 4: Weighted Loss
            total_loss = loss_diff + config.get('lambda_recon', 0.1) * loss_recon
            total_loss.backward()
            
            # Stability: Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += total_loss.item()
            pbar.set_postfix({'loss': f"{total_loss.item():.4f}"})
            
        # Update Scheduler
        scheduler.step()
            
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_cond, x_0 in val_loader:
                x_cond, x_0 = x_cond.to(device), x_0.to(device)
                ld, lr = model(x_cond, x_0)
                val_loss += (ld + config.get('lambda_recon', 0.1) * lr).item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} Summary: Train={avg_train_loss:.6f}, Val={avg_val_loss:.6f}, LR={curr_lr:.2e}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(), 'checkpoints/best_mmpd.pth')
            print("Saved best model.")
            # Log to file
            with open('train_log.txt', 'a', encoding='utf-8') as f:
                f.write(f"Epoch {epoch+1}: Train={avg_train_loss:.6f}, Val={avg_val_loss:.6f}, LR={curr_lr:.2e} [BEST]\n")
        else:
            patience_counter += 1
            print(f"EarlyStopping counter: {patience_counter} out of {patience}")
            with open('train_log.txt', 'a', encoding='utf-8') as f:
                f.write(f"Epoch {epoch+1}: Train={avg_train_loss:.6f}, Val={avg_val_loss:.6f}, LR={curr_lr:.2e}\n")
            if patience_counter >= patience:
                print("Early stopping triggered. Training terminated.")
                break

if __name__ == "__main__":
    with open('config.json', 'r') as f:
        config = json.load(f)
    train(config)

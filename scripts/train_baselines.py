import os
import sys
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.dataset import get_dataloaders
from scripts.baselines import PatchTSTBaseline, USADBaseline, LSTMBaseline
from scripts.train import set_seed, count_parameters

def train_baseline(config, model_type):
    set_seed(config.get('seed', 42))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training {model_type} on {device}")
    
    train_loader, val_loader, data_meta = get_dataloaders(config['data_path'], config)
    config['enc_in'] = data_meta['enc_in']
    config['num_tm'] = data_meta['num_tm']
    config['num_tc'] = data_meta['num_tc']
    
    if model_type == 'PatchTST':
        model = PatchTSTBaseline(config).to(device)
    elif model_type == 'USAD':
        model = USADBaseline(config).to(device)
    elif model_type == 'LSTM':
        model = LSTMBaseline(config).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    epochs = config['epochs']
    
    os.makedirs('checkpoints', exist_ok=True)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for x_cond, x_0, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            x_cond, x_0 = x_cond.to(device), x_0.to(device)
            optimizer.zero_grad()
            
            if model_type in ['PatchTST', 'LSTM']:
                pred = model(x_cond)
                x_0_tm = x_0[..., :config['num_tm']]
                loss = criterion(pred, x_0_tm)
            elif model_type == 'USAD':
                # USAD trains on reconstructing x_cond
                # Simplified USAD loss
                n = epoch + 1
                w1, w2, w3 = model(x_cond)
                loss1 = 1/n * criterion(x_cond, w1) + (1 - 1/n) * criterion(x_cond, w3)
                loss2 = 1/n * criterion(x_cond, w2) - (1 - 1/n) * criterion(x_cond, w3)
                loss = loss1 + loss2
                
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_cond, x_0, _ in val_loader:
                x_cond, x_0 = x_cond.to(device), x_0.to(device)
                if model_type in ['PatchTST', 'LSTM']:
                    pred = model(x_cond)
                    x_0_tm = x_0[..., :config['num_tm']]
                    loss = criterion(pred, x_0_tm)
                elif model_type == 'USAD':
                    w1, w2, w3 = model(x_cond)
                    loss = criterion(x_cond, w1) # validation uses reconstruction error
                val_loss += loss.item()
                
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.6f} | Val Loss: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'checkpoints/{model_type}_best.pth')
            print("  -> Saved new best model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['PatchTST', 'USAD', 'LSTM'])
    args = parser.parse_args()
    
    with open('config.json', 'r') as f:
        config = json.load(f)
        
    train_baseline(config, args.model)

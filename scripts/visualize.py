import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import re

def plot_loss(log_path='train_log.txt', output_dir='results'):
    if not os.path.exists(log_path):
        print(f"Log file {log_path} not found.")
        return
    
    epochs = []
    train_losses = []
    val_losses = []
    
    # Regex to parse the log lines
    # Format: Epoch 1: Train=0.086908, Val=0.017930, LR=1.00e-03
    pattern = re.compile(r'Epoch (\d+): Train=([\d\.]+), Val=([\d\.]+)')
    
    # Robust reading: Try utf-16 first (Windows default for some apps), then utf-8
    content = ""
    for enc in ['utf-16', 'utf-8', 'gbk']:
        try:
            with open(log_path, 'r', encoding=enc) as f:
                content = f.read()
            break
        except UnicodeDecodeError:
            continue
            
    if not content:
        print(f"Could not read {log_path} with common encodings.")
        return

    for line in content.splitlines():
        match = pattern.search(line)
        if match:
            epochs.append(int(match.group(1)))
            train_losses.append(float(match.group(2)))
            val_losses.append(float(match.group(3)))
    
    if not epochs:
        print("No valid log data found to plot.")
        return

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs, val_losses, label='Val Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('MMPD Training and Validation Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'loss_curve.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Loss curve saved to {save_path}")
    plt.close()

def plot_anomaly_results(results_csv='results/anomaly_detection_results.csv', output_dir='results', max_points=10000):
    if not os.path.exists(results_csv):
        print(f"Results file {results_csv} not found.")
        return
    
    df = pd.read_csv(results_csv)
    
    # If the data is too large, sub-sample for visualization
    if len(df) > max_points:
        print(f"Data too large ({len(df)} points), sampling every {len(df)//max_points} points for plotting.")
        df_plot = df.iloc[::len(df)//max_points].reset_index(drop=True)
    else:
        df_plot = df

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    # Plot 1: Anomaly Scores
    ax1.plot(df_plot.index, df_plot['anomaly_score'], color='blue', label='Anomaly Score', alpha=0.6)
    
    # Highlight Threshold (Approximate it based on evaluate logic if possible, or just plot)
    # Since threshold varies, we just plot the scores and highlights
    
    # Plot 2: Labels and Predictions
    ax2.fill_between(df_plot.index, 0, df_plot['label'], color='red', alpha=0.3, label='Ground Truth Anomaly')
    ax2.plot(df_plot.index, df_plot['prediction'], color='green', label='Model Prediction', linewidth=1)
    
    ax1.set_ylabel('Score')
    ax1.set_title('Anomaly Scores over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_ylabel('Detection (0/1)')
    ax2.set_xlabel('Sample Index (Sub-sampled)')
    ax2.set_title('Ground Truth vs Prediction')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'anomaly_detection_plot.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Anomaly plot saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    plot_loss()
    plot_anomaly_results()

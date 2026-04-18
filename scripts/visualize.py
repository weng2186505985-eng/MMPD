import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import re

def plot_loss(log_path='train_log.txt', output_dir='results'):
    import json as _json
    
    # Try JSON history first (new format)
    json_path = os.path.join(output_dir, 'train_history.json')
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                history = _json.load(f)
            if history.get('epochs'):
                epochs = [e['epoch'] for e in history['epochs']]
                train_losses = [e['train_total_loss'] for e in history['epochs']]
                val_losses = [e['val_total_loss'] for e in history['epochs']]
                
                plt.figure(figsize=(10, 5))
                plt.plot(epochs, train_losses, label='Train Loss', marker='o', color='#2196F3')
                plt.plot(epochs, val_losses, label='Val Loss', marker='s', color='#F44336')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('MMPD Training and Validation Loss')
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)
                
                os.makedirs(output_dir, exist_ok=True)
                save_path = os.path.join(output_dir, 'loss_curve.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Loss curve saved to {save_path} (from JSON history)")
                plt.close()
                return
        except Exception:
            pass
    
    # Fallback: parse text log
    if not os.path.exists(log_path):
        print(f"Log file {log_path} not found.")
        return
    
    epochs = []
    train_losses = []
    val_losses = []
    
    # Regex to parse the log lines
    # Format: Epoch 1: Train=0.086908, Val=0.017930, LR=1.00e-03
    pattern = re.compile(r'Epoch (\d+): Train=([\d\.]+)')
    val_pattern = re.compile(r'Val=([\d\.]+)')
    
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
        val_match = val_pattern.search(line)
        if match and val_match:
            epochs.append(int(match.group(1)))
            train_losses.append(float(match.group(2)))
            val_losses.append(float(val_match.group(1)))
    
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

def plot_roc_pr_curves(results_csv='results/anomaly_detection_results.csv', output_dir='results'):
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
    if not os.path.exists(results_csv): return
    df = pd.read_csv(results_csv)
    
    if len(df['label'].unique()) < 2:
        print("Not enough classes for ROC/PR curves.")
        return

    # ROC
    fpr, tpr, _ = roc_curve(df['label'], df['anomaly_score'])
    roc_auc = auc(fpr, tpr)
    
    # PR
    precision, recall, _ = precision_recall_curve(df['label'], df['anomaly_score'])
    pr_auc = average_precision_score(df['label'], df['anomaly_score'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Receiver Operating Characteristic')
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    ax2.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.3f})')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'roc_pr_curves.png')
    plt.savefig(save_path, dpi=300)
    print(f"ROC/PR curves saved to {save_path}")
    plt.close()

def plot_score_distribution(results_csv='results/anomaly_detection_results.csv', output_dir='results'):
    if not os.path.exists(results_csv): return
    df = pd.read_csv(results_csv)
    
    plt.figure(figsize=(10, 5))
    normal_scores = df[df['label'] == 0]['anomaly_score']
    anomaly_scores = df[df['label'] == 1]['anomaly_score']
    
    plt.hist(normal_scores, bins=50, alpha=0.5, color='blue', label='Normal', density=True)
    if len(anomaly_scores) > 0:
        plt.hist(anomaly_scores, bins=50, alpha=0.5, color='red', label='Anomaly', density=True)
        
    plt.xlabel('Anomaly Score')
    plt.ylabel('Density')
    plt.title('Anomaly Score Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(output_dir, 'score_distribution.png')
    plt.savefig(save_path, dpi=300)
    print(f"Score distribution saved to {save_path}")
    plt.close()

def plot_confusion_matrix_heatmap(results_csv='results/anomaly_detection_results.csv', output_dir='results'):
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    if not os.path.exists(results_csv): return
    df = pd.read_csv(results_csv)
    
    cm = confusion_matrix(df['label'], df['prediction'])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    
    save_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(save_path, dpi=300)
    print(f"Confusion matrix saved to {save_path}")
    plt.close()

def plot_zoomed_anomalies(results_csv='results/anomaly_detection_results.csv', output_dir='results', window=500):
    if not os.path.exists(results_csv): return
    df = pd.read_csv(results_csv)
    
    anomaly_indices = df[df['label'] == 1].index
    if len(anomaly_indices) == 0: return
    
    # Group anomalies into events
    events = []
    current_event = [anomaly_indices[0]]
    for idx in anomaly_indices[1:]:
        if idx == current_event[-1] + 1:
            current_event.append(idx)
        else:
            events.append(current_event)
            current_event = [idx]
    events.append(current_event)
    
    # Pick the longest event
    events.sort(key=len, reverse=True)
    top_event = events[0]
    center_idx = top_event[len(top_event)//2]
    
    start_idx = max(0, center_idx - window)
    end_idx = min(len(df), center_idx + window)
    df_zoom = df.iloc[start_idx:end_idx]
    
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(df_zoom.index, df_zoom['anomaly_score'], color='blue', label='Anomaly Score')
    ax1.set_ylabel('Score')
    ax1.set_xlabel('Sample Index')
    
    ax2 = ax1.twinx()
    ax2.fill_between(df_zoom.index, 0, df_zoom['label'], color='red', alpha=0.3, label='Ground Truth')
    ax2.plot(df_zoom.index, df_zoom['prediction'], color='green', label='Prediction')
    ax2.set_ylabel('Label / Prediction')
    ax2.set_ylim(-0.1, 1.1)
    
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')
    
    plt.title('Zoomed-in Anomaly Event')
    save_path = os.path.join(output_dir, 'zoomed_anomaly.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Zoomed anomaly plot saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    plot_loss()
    plot_anomaly_results()
    plot_roc_pr_curves()
    plot_score_distribution()
    plot_confusion_matrix_heatmap()
    plot_zoomed_anomalies()

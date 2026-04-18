import os
import sys

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score

def plot_metrics_vs_threshold(results_csv='results/anomaly_detection_results.csv', output_dir='results'):
    """
    Plots Precision, Recall, and F1-score as the anomaly threshold varies.
    This helps in visually identifying the optimal threshold.
    """
    if not os.path.exists(results_csv):
        print(f"File not found: {results_csv}")
        return

    df = pd.read_csv(results_csv)
    
    # Check if we have both normal and anomaly classes
    if len(df['label'].unique()) < 2:
        print("Not enough classes for threshold metrics.")
        return

    # Sample percentiles to test
    percentiles = np.linspace(80, 99.9, 30)
    precisions = []
    recalls = []
    f1_scores = []

    labels = df['label'].values
    scores = df['anomaly_score'].values

    print("Calculating metrics across varying thresholds...")
    for p in percentiles:
        thresh = np.percentile(scores, p)
        preds = (scores > thresh).astype(int)
        
        # Calculate metrics (standard point-wise for speed, though event-wise is better for time-series)
        prec = precision_score(labels, preds, zero_division=0)
        rec = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)
        
        precisions.append(prec)
        recalls.append(rec)
        f1_scores.append(f1)

    plt.figure(figsize=(10, 6))
    plt.plot(percentiles, precisions, label='Precision', color='blue', linewidth=2)
    plt.plot(percentiles, recalls, label='Recall', color='red', linewidth=2)
    plt.plot(percentiles, f1_scores, label='F1-Score', color='green', linewidth=2, linestyle='--')
    
    # Mark the max F1 score
    max_f1_idx = np.argmax(f1_scores)
    max_f1_percentile = percentiles[max_f1_idx]
    plt.axvline(max_f1_percentile, color='black', linestyle=':', label=f'Max F1 at {max_f1_percentile:.1f}%')

    plt.xlabel('Anomaly Score Threshold (Percentile)')
    plt.ylabel('Score')
    plt.title('Performance Metrics vs. Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'metrics_vs_threshold.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved {save_path}")
    plt.close()

def plot_temporal_heatmap(results_csv='results/anomaly_detection_results.csv', output_dir='results'):
    """
    Creates a heatmap showing when anomalies most frequently occur 
    (Day of Week vs. Hour of Day).
    """
    if not os.path.exists(results_csv): return
    df = pd.read_csv(results_csv)
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Filter only true anomalies
    anomalies = df[df['label'] == 1].copy()
    if len(anomalies) == 0:
        print("No true anomalies to plot temporal heatmap.")
        return

    anomalies['hour'] = anomalies['timestamp'].dt.hour
    anomalies['dayofweek'] = anomalies['timestamp'].dt.dayofweek

    # Create pivot table for heatmap
    heatmap_data = pd.crosstab(anomalies['dayofweek'], anomalies['hour'])
    
    # Ensure all days and hours are present
    heatmap_data = heatmap_data.reindex(index=range(7), columns=range(24), fill_value=0)
    
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    plt.figure(figsize=(12, 5))
    sns.heatmap(heatmap_data, cmap='YlOrRd', yticklabels=days, cbar_kws={'label': 'Anomaly Count'})
    plt.title('Temporal Distribution of True Anomalies')
    plt.xlabel('Hour of Day')
    plt.ylabel('Day of Week')
    
    save_path = os.path.join(output_dir, 'temporal_heatmap.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved {save_path}")
    plt.close()

def plot_anomaly_duration_histogram(results_csv='results/anomaly_detection_results.csv', output_dir='results'):
    """
    Plots the distribution of the continuous duration of anomaly events.
    """
    if not os.path.exists(results_csv): return
    df = pd.read_csv(results_csv)
    
    # Find continuous segments of anomalies
    labels = df['label'].values
    events = []
    
    is_event = False
    duration = 0
    
    for l in labels:
        if l == 1:
            is_event = True
            duration += 1
        elif l == 0 and is_event:
            events.append(duration)
            duration = 0
            is_event = False
            
    if is_event:
        events.append(duration)
        
    if not events:
        print("No anomaly events found for duration histogram.")
        return

    plt.figure(figsize=(10, 5))
    # Using a log scale for duration if it varies widely
    sns.histplot(events, bins=50, kde=True, color='purple')
    plt.xlabel('Duration of Anomaly Event (time steps)')
    plt.ylabel('Frequency (Number of Events)')
    plt.title('Distribution of Anomaly Event Durations')
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(output_dir, 'anomaly_duration_hist.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved {save_path}")
    plt.close()

def plot_score_boxplot(results_csv='results/anomaly_detection_results.csv', output_dir='results'):
    """
    Plots a boxplot of anomaly scores for Normal vs Anomaly classes.
    Uses log scale for better visualization of outliers.
    """
    if not os.path.exists(results_csv): return
    df = pd.read_csv(results_csv)
    
    # Subsample if too large for seaborn boxplot
    if len(df) > 50000:
        df_plot = pd.concat([
            df[df['label'] == 0].sample(n=min(25000, len(df[df['label']==0])), random_state=42),
            df[df['label'] == 1].sample(n=min(25000, len(df[df['label']==1])), random_state=42)
        ])
    else:
        df_plot = df

    df_plot['Class'] = df_plot['label'].map({0: 'Normal', 1: 'Anomaly'})
    
    plt.figure(figsize=(8, 6))
    # Add small epsilon to avoid log(0)
    df_plot['log_score'] = np.log10(df_plot['anomaly_score'] + 1e-6)
    
    sns.boxplot(x='Class', y='log_score', data=df_plot, palette={'Normal': 'blue', 'Anomaly': 'red'})
    plt.title('Anomaly Score Distribution (Log Scale)')
    plt.ylabel('Log10(Anomaly Score)')
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(output_dir, 'score_boxplot.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved {save_path}")
    plt.close()

if __name__ == "__main__":
    plot_metrics_vs_threshold()
    plot_temporal_heatmap()
    plot_anomaly_duration_histogram()
    plot_score_boxplot()
    print("All advanced visualizations completed.")

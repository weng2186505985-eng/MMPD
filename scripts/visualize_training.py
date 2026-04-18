"""
训练过程可视化脚本
================
从 train_history.json 读取完整的训练历史，生成多种学术级可视化图表。

生成的图表:
1. 综合训练仪表盘 (4合1)
2. 分项损失对比曲线
3. 学习率变化曲线
4. 梯度范数监控图
5. 训练效率分析图
6. 损失景观分析图 (收敛率)
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ─── 全局风格配置 ───
plt.rcParams.update({
    'figure.facecolor': '#FAFAFA',
    'axes.facecolor': '#FFFFFF',
    'axes.edgecolor': '#CCCCCC',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'lines.linewidth': 2,
    'figure.dpi': 150,
})

# 调色板
COLORS = {
    'train_total': '#2196F3',     # Blue
    'val_total': '#F44336',       # Red
    'train_diff': '#1565C0',      # Dark Blue
    'val_diff': '#C62828',        # Dark Red
    'train_recon': '#42A5F5',     # Light Blue
    'val_recon': '#EF5350',       # Light Red
    'lr': '#FF9800',              # Orange
    'grad_avg': '#4CAF50',        # Green
    'grad_max': '#E91E63',        # Pink
    'time': '#9C27B0',            # Purple
    'best': '#FFD700',            # Gold
    'fill_train': '#BBDEFB',      # Light Blue Fill
    'fill_val': '#FFCDD2',        # Light Red Fill
}


def load_history(history_path='results/train_history.json'):
    """加载训练历史"""
    if not os.path.exists(history_path):
        print(f"训练历史文件未找到: {history_path}")
        print("请先运行 train.py 生成训练历史。")
        return None
    with open(history_path, 'r', encoding='utf-8') as f:
        history = json.load(f)
    if not history.get('epochs'):
        print("训练历史为空，没有 epoch 数据。")
        return None
    return history


def extract_metrics(history):
    """从历史记录中提取指标数组"""
    epochs_data = history['epochs']
    metrics = {
        'epochs': [e['epoch'] for e in epochs_data],
        'lr': [e['lr'] for e in epochs_data],
        'train_total': [e['train_total_loss'] for e in epochs_data],
        'train_diff': [e['train_diff_loss'] for e in epochs_data],
        'train_recon': [e['train_recon_loss'] for e in epochs_data],
        'val_total': [e['val_total_loss'] for e in epochs_data],
        'val_diff': [e['val_diff_loss'] for e in epochs_data],
        'val_recon': [e['val_recon_loss'] for e in epochs_data],
        'avg_grad': [e['avg_grad_norm'] for e in epochs_data],
        'max_grad': [e['max_grad_norm'] for e in epochs_data],
        'time': [e['epoch_time_sec'] for e in epochs_data],
        'gpu_mem': [e.get('gpu_mem_mb', 0) for e in epochs_data],
        'is_best': [e.get('is_best', False) for e in epochs_data],
    }
    return metrics


def plot_training_dashboard(history, output_dir='results'):
    """
    生成 4 合 1 的训练仪表盘:
      左上: 总损失曲线
      右上: 分项损失曲线
      左下: 学习率 + 梯度范数
      右下: 训练效率 (时间/epoch)
    """
    metrics = extract_metrics(history)
    epochs = metrics['epochs']
    
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.3)
    
    # ── 左上: 总损失曲线 ──
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, metrics['train_total'], color=COLORS['train_total'], label='Train Loss', marker='o', markersize=5)
    ax1.plot(epochs, metrics['val_total'], color=COLORS['val_total'], label='Val Loss', marker='s', markersize=5)
    
    # 标记最佳 epoch
    best_epochs = [e for e, is_best in zip(epochs, metrics['is_best']) if is_best]
    best_vals = [v for v, is_best in zip(metrics['val_total'], metrics['is_best']) if is_best]
    if best_epochs:
        ax1.scatter(best_epochs, best_vals, color=COLORS['best'], s=120, zorder=5,
                    marker='★', label=f'Best (Epoch {best_epochs[-1]})')
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Total Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.legend(loc='upper right')
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # ── 右上: 分项损失曲线 ──
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, metrics['train_diff'], color=COLORS['train_diff'], label='Train Diffusion', linestyle='-')
    ax2.plot(epochs, metrics['val_diff'], color=COLORS['val_diff'], label='Val Diffusion', linestyle='-')
    ax2.plot(epochs, metrics['train_recon'], color=COLORS['train_recon'], label='Train Recon', linestyle='--')
    ax2.plot(epochs, metrics['val_recon'], color=COLORS['val_recon'], label='Val Recon', linestyle='--')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Component Losses (Diffusion vs Reconstruction)')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # ── 左下: 学习率 + 梯度范数 ──
    ax3 = fig.add_subplot(gs[1, 0])
    ax3_lr = ax3
    ax3_lr.plot(epochs, metrics['lr'], color=COLORS['lr'], label='Learning Rate', linewidth=2.5)
    ax3_lr.set_xlabel('Epoch')
    ax3_lr.set_ylabel('Learning Rate', color=COLORS['lr'])
    ax3_lr.tick_params(axis='y', labelcolor=COLORS['lr'])
    ax3_lr.set_title('Learning Rate & Gradient Norms')
    
    ax3_grad = ax3.twinx()
    ax3_grad.plot(epochs, metrics['avg_grad'], color=COLORS['grad_avg'], label='Avg Grad Norm', alpha=0.8)
    ax3_grad.plot(epochs, metrics['max_grad'], color=COLORS['grad_max'], label='Max Grad Norm', alpha=0.6, linestyle=':')
    ax3_grad.set_ylabel('Gradient Norm', color=COLORS['grad_avg'])
    ax3_grad.tick_params(axis='y', labelcolor=COLORS['grad_avg'])
    
    # 合并图例
    lines1, labels1 = ax3_lr.get_legend_handles_labels()
    lines2, labels2 = ax3_grad.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)
    ax3.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # ── 右下: 训练效率 ──
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.bar(epochs, metrics['time'], color=COLORS['time'], alpha=0.7, label='Time per Epoch')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Time (seconds)')
    ax4.set_title('Training Efficiency')
    
    # 标注平均时间
    avg_time = np.mean(metrics['time'])
    ax4.axhline(y=avg_time, color='red', linestyle='--', alpha=0.7, label=f'Avg: {avg_time:.1f}s')
    ax4.legend(loc='upper right')
    ax4.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # 总标题
    total_time = history.get('total_training_time_sec', sum(metrics['time']))
    best_loss = history.get('best_val_loss', min(metrics['val_total']))
    fig.suptitle(
        f'MMPD Training Dashboard  |  Best Val Loss: {best_loss:.6f}  |  Total Time: {total_time/60:.1f} min',
        fontsize=14, fontweight='bold', y=0.98
    )
    
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'training_dashboard.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"Saved: {save_path}")
    plt.close()


def plot_loss_curves(history, output_dir='results'):
    """生成训练/验证损失曲线（带置信区间效果）"""
    metrics = extract_metrics(history)
    epochs = metrics['epochs']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # ── 总损失 ──
    ax1.plot(epochs, metrics['train_total'], color=COLORS['train_total'], label='Train', marker='o', markersize=4)
    ax1.plot(epochs, metrics['val_total'], color=COLORS['val_total'], label='Validation', marker='s', markersize=4)
    ax1.fill_between(epochs, metrics['train_total'], metrics['val_total'], 
                     alpha=0.1, color='gray', label='Train-Val Gap')
    
    # 标注最佳
    best_epochs = [e for e, b in zip(epochs, metrics['is_best']) if b]
    best_vals = [v for v, b in zip(metrics['val_total'], metrics['is_best']) if b]
    if best_epochs:
        ax1.axvline(x=best_epochs[-1], color=COLORS['best'], linestyle='--', alpha=0.5)
        ax1.annotate(f'Best: {best_vals[-1]:.4f}\n(Epoch {best_epochs[-1]})',
                     xy=(best_epochs[-1], best_vals[-1]),
                     xytext=(20, 20), textcoords='offset points',
                     arrowprops=dict(arrowstyle='->', color='gray'),
                     fontsize=9, bbox=dict(boxstyle='round', facecolor='lightyellow'))
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Total Loss')
    ax1.set_title('Total Loss Curve')
    ax1.legend()
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # ── 分项损失 ──
    ax2.plot(epochs, metrics['train_diff'], color=COLORS['train_diff'], label='Train Diffusion', marker='o', markersize=3)
    ax2.plot(epochs, metrics['val_diff'], color=COLORS['val_diff'], label='Val Diffusion', marker='s', markersize=3)
    ax2.plot(epochs, metrics['train_recon'], color=COLORS['train_recon'], label='Train Recon', marker='^', markersize=3, linestyle='--')
    ax2.plot(epochs, metrics['val_recon'], color=COLORS['val_recon'], label='Val Recon', marker='v', markersize=3, linestyle='--')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Component Loss Breakdown')
    ax2.legend(fontsize=9)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    fig.suptitle('MMPD Training & Validation Loss', fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'loss_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_lr_schedule(history, output_dir='results'):
    """学习率变化曲线"""
    metrics = extract_metrics(history)
    epochs = metrics['epochs']
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epochs, metrics['lr'], color=COLORS['lr'], marker='o', markersize=6, linewidth=2.5)
    ax.fill_between(epochs, 0, metrics['lr'], alpha=0.15, color=COLORS['lr'])
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule (Cosine Annealing)')
    ax.set_yscale('log')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # 标注起止LR
    ax.annotate(f'{metrics["lr"][0]:.1e}', xy=(epochs[0], metrics['lr'][0]),
                xytext=(10, 10), textcoords='offset points', fontsize=9)
    ax.annotate(f'{metrics["lr"][-1]:.1e}', xy=(epochs[-1], metrics['lr'][-1]),
                xytext=(-50, 10), textcoords='offset points', fontsize=9)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'lr_schedule.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_gradient_analysis(history, output_dir='results'):
    """梯度范数分析图"""
    metrics = extract_metrics(history)
    epochs = metrics['epochs']
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(epochs, metrics['avg_grad'], color=COLORS['grad_avg'], marker='o', markersize=5, label='Avg Gradient Norm')
    ax.fill_between(epochs, metrics['avg_grad'], metrics['max_grad'], alpha=0.2, color=COLORS['grad_max'])
    ax.plot(epochs, metrics['max_grad'], color=COLORS['grad_max'], marker='^', markersize=5, 
            linestyle=':', alpha=0.8, label='Max Gradient Norm')
    
    # 梯度裁剪线
    ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.4, label='Clip Threshold (1.0)')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Gradient Norm')
    ax.set_title('Gradient Norm Monitoring')
    ax.legend()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # 判断梯度稳定性
    grad_std = np.std(metrics['avg_grad'])
    grad_mean = np.mean(metrics['avg_grad'])
    stability = 'Stable ✓' if grad_std / (grad_mean + 1e-8) < 0.5 else 'Unstable ⚠'
    ax.text(0.02, 0.95, f'Stability: {stability}\nCV={grad_std/grad_mean:.2f}', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'gradient_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_convergence_analysis(history, output_dir='results'):
    """损失收敛分析（损失变化率 + 过拟合检测）"""
    metrics = extract_metrics(history)
    epochs = metrics['epochs']
    
    if len(epochs) < 2:
        print("Not enough epochs for convergence analysis.")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # ── 损失变化率 ──
    train_deltas = np.diff(metrics['train_total'])
    val_deltas = np.diff(metrics['val_total'])
    delta_epochs = epochs[1:]
    
    ax1.bar(np.array(delta_epochs) - 0.15, train_deltas, width=0.3, color=COLORS['train_total'], 
            alpha=0.7, label='Train Δ')
    ax1.bar(np.array(delta_epochs) + 0.15, val_deltas, width=0.3, color=COLORS['val_total'], 
            alpha=0.7, label='Val Δ')
    ax1.axhline(y=0, color='black', linewidth=0.8)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss Change (Δ)')
    ax1.set_title('Loss Change Rate per Epoch')
    ax1.legend()
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # ── 过拟合指标: Train-Val Gap ──
    gap = np.array(metrics['val_total']) - np.array(metrics['train_total'])
    gap_color = ['#4CAF50' if g < 0.01 else '#FF9800' if g < 0.05 else '#F44336' for g in gap]
    
    ax2.bar(epochs, gap, color=gap_color, alpha=0.7)
    ax2.axhline(y=0, color='black', linewidth=0.8)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Val Loss - Train Loss')
    ax2.set_title('Overfitting Monitor (Val-Train Gap)')
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # 颜色图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#4CAF50', alpha=0.7, label='Good (<0.01)'),
        Patch(facecolor='#FF9800', alpha=0.7, label='Warning (0.01-0.05)'),
        Patch(facecolor='#F44336', alpha=0.7, label='Overfitting (>0.05)'),
    ]
    ax2.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    fig.suptitle('Convergence & Overfitting Analysis', fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'convergence_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_training_summary_table(history, output_dir='results'):
    """生成训练统计摘要表格图"""
    metrics = extract_metrics(history)
    
    best_epoch_idx = np.argmin(metrics['val_total'])
    
    summary_data = [
        ['Total Epochs', str(len(metrics['epochs']))],
        ['Best Epoch', str(metrics['epochs'][best_epoch_idx])],
        ['Best Val Loss', f"{metrics['val_total'][best_epoch_idx]:.6f}"],
        ['Final Train Loss', f"{metrics['train_total'][-1]:.6f}"],
        ['Final Val Loss', f"{metrics['val_total'][-1]:.6f}"],
        ['Initial LR', f"{metrics['lr'][0]:.1e}"],
        ['Final LR', f"{metrics['lr'][-1]:.1e}"],
        ['Avg Grad Norm', f"{np.mean(metrics['avg_grad']):.4f}"],
        ['Max Grad Norm', f"{np.max(metrics['max_grad']):.4f}"],
        ['Avg Time/Epoch', f"{np.mean(metrics['time']):.1f}s"],
        ['Total Time', f"{sum(metrics['time'])/60:.1f} min"],
        ['Model Params', f"{history.get('trainable_params', 'N/A'):,}"],
    ]
    
    if max(metrics['gpu_mem']) > 0:
        summary_data.append(['Peak GPU Mem', f"{max(metrics['gpu_mem']):.0f} MB"])
    
    fig, ax = plt.subplots(figsize=(8, max(5, len(summary_data) * 0.45)))
    ax.axis('off')
    
    table = ax.table(cellText=summary_data, colLabels=['Metric', 'Value'],
                     cellLoc='left', loc='center',
                     colColours=['#E3F2FD', '#E3F2FD'])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.6)
    
    # Style header
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight='bold')
            cell.set_facecolor('#1565C0')
            cell.set_text_props(color='white', fontweight='bold')
        elif row % 2 == 0:
            cell.set_facecolor('#F5F5F5')
        cell.set_edgecolor('#DDDDDD')
    
    ax.set_title('Training Summary', fontsize=14, fontweight='bold', pad=20)
    
    save_path = os.path.join(output_dir, 'training_summary.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def generate_all_training_visualizations(history_path='results/train_history.json', output_dir='results'):
    """主入口: 生成所有训练可视化"""
    history = load_history(history_path)
    if history is None:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*50}")
    print(f"  Generating Training Visualizations")
    print(f"{'='*50}")
    print(f"  Epochs: {len(history['epochs'])}")
    print(f"  Output: {output_dir}/")
    print(f"{'='*50}\n")
    
    plot_training_dashboard(history, output_dir)
    plot_loss_curves(history, output_dir)
    plot_lr_schedule(history, output_dir)
    plot_gradient_analysis(history, output_dir)
    plot_convergence_analysis(history, output_dir)
    plot_training_summary_table(history, output_dir)
    
    print(f"\n{'='*50}")
    print(f"  All training visualizations complete!")
    print(f"  Generated 6 plots in {output_dir}/")
    print(f"{'='*50}")


if __name__ == "__main__":
    generate_all_training_visualizations()

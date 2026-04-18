# 异常检测模型可视化图表索引指南

本文档对 `results` 文件夹下生成的各类可视化图表进行了分类和说明。这些图表是评估模型性能和撰写学术论文/技术报告的重要素材。所有图表均由 `visualize.py` 和 `advanced_visualize.py` 基于推理结果 (`anomaly_detection_results.csv`) 自动生成。

---

## 1. 阈值决策与模型内在性能 (Threshold & Intrinsic Performance)

这部分图表主要用于论证**第一阶段评分算法的有效性**以及**第二阶段阈值选择的科学性**。

*   🖼️ **`metrics_vs_threshold.png` (性能指标随阈值变化曲线)**
    *   **描述**：展示 Precision、Recall 和 F1-Score 随判定阈值百分位（80% - 99.9%）变化的趋势。
    *   **论文用法**：用于证明您所选取的特定阈值（如 Percentile 99）是综合 F1 得分最高的最优解。

*   🖼️ **`roc_pr_curves.png` (ROC 曲线与 PR 曲线)**
    *   **描述**：展示与具体阈值无关的全局区分能力。
    *   **论文用法**：用于横向对比不同模型的基础性能。如果 PR 曲线面积大，说明在极度不平衡的数据集中模型依然稳健。

*   🖼️ **`score_distribution.png` & `score_boxplot.png` (得分分布与箱线图)**
    *   **描述**：展示正常样本与异常样本的异常得分分布。
    *   **论文用法**：用于直观说明模型是否能将正常和异常数据在评分空间中有效拉开距离。

## 2. 宏观时序与事件分析 (Macro Temporal & Event Analysis)

这部分图表从业务场景和全局视角分析异常的分布规律。

*   🖼️ **`anomaly_detection_plot.png` (全局时序检测对比图)**
    *   **描述**：在时间轴上叠加展示 Ground Truth、原始异常得分以及模型的最终二分类预测。
    *   **论文用法**：提供模型在整个生命周期中稳定运行的全局概览。

*   🖼️ **`temporal_heatmap.png` (异常时间分布热力图)**
    *   **描述**：展示真实异常在“星期几”和“一天中哪个小时”的高发频次。
    *   **论文用法**：增强论文的数据挖掘与业务洞察深度，揭示数据背后可能存在的周期性规律或工作环境影响。

*   🖼️ **`anomaly_duration_hist.png` (异常事件持续时间直方图)**
    *   **描述**：统计每一次连续异常事件的时间跨度。
    *   **论文用法**：用于论证数据集的异常特性（如：绝大多数异常是短脉冲型还是长尾持续型）。

## 3. 落地应用与微观细节 (Application & Micro Details)

这部分聚焦于最终的检测结果和局部响应速度。

*   🖼️ **`confusion_matrix.png` (混淆矩阵热力图)**
    *   **描述**：展示最终判定的 True Positives, False Positives, True Negatives, False Negatives 数量。
    *   **论文用法**：最直接的标准评估图，说明模型是倾向于高误报还是高漏报。

*   🖼️ **`zoomed_anomaly.png` (局部异常事件放大图)**
    *   **描述**：自动截取测试集中持续时间最长的一段异常事件并放大其前后波形。
    *   **论文用法**：非常适合作为 **Case Study (案例分析)**。用于证明模型在异常发生的一瞬间具有极快的响应速度（得分陡增）。

---

> **💡 进阶建议**
> 如果您的论文涉及到 Point-Adjustment（基于事件的点调整计算方法），强烈建议补充 **“误报漏报专属分布图”** 和 **“点调整机制对齐对比图”**，这将极大提升论文的深度和专业性。


## 4. 训练过程可视化 (Training Process Visualization)

此部分图表由 isualize_training.py 基于训练历史 (	rain_history.json) 自动生成。

*   	raining_dashboard.png — 4合1训练仪表盘（总损失/分项损失/学习率与梯度/训练效率）
*   loss_curves.png — 总损失与分项损失对比曲线
*   lr_schedule.png — 学习率调度曲线 (Cosine Annealing)
*   gradient_analysis.png — 梯度范数监控与稳定性分析
*   convergence_analysis.png — 收敛率与过拟合检测
*   	raining_summary.png — 训练统计摘要表

> 使用方法: python -m scripts.visualize_training

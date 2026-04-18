# 多阶段掩码补丁扩散模型 (MMPD) - 航天器异常检测

本仓库包含针对航天器多元时间序列（包含连续遥测 TM 和离散遥控 TC 特征）异常检测框架的官方 PyTorch 实现代码。

本模型采用两阶段检测方案：
1. **第一阶段 (Stage 1)**: 利用 Transformer 骨干网络进行序列重建，实现快速粗筛。
2. **第二阶段 (Stage 2)**: 结合动态变分高斯混合模型 (GMM) 和 DDIM 扩散模型进行细粒度异常精炼。

## 核心特性
- **通道混合 (Cross-Channel Mixing)**: 打破传统通道独立假设，有效捕捉不同传感器之间的复杂联合分布。
- **指令特征融合 (TC Embedding Fusion)**: 将离散的遥控指令事件映射为连续向量，并与隐藏层特征深度融合。
- **不确定性权重 (Uncertainty Weighting)**: 动态平衡扩散去噪损失与序列重建损失。
- **批处理推理 (Batched DDIM & GMM)**: 解锁全并行推理，极大提升测试集评估速度。
- **严谨评估协议 (PA%K)**: 采用严谨的 $PA_{>10\%}$ 协议，消除传统 Point-Adjust (PA) 带来的指标虚高问题。
- **集成多种基线模型 (Baselines)**: 内置并支持训练测试 PatchTST, USAD, LSTM-NDT 等最新/经典异常检测基线模型。

---

## 1. 环境配置 (Linux)

强烈建议使用 Conda 创建独立的虚拟环境。

```bash
# 创建新的 conda 环境
conda create -n mmpd python=3.9 -y
conda activate mmpd

# 安装 PyTorch (请根据您的服务器 CUDA 版本替换链接，例如 CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖包
pip install pandas numpy scipy scikit-learn tqdm matplotlib
```

---

## 2. 数据准备

请将 ESA 原始数据集放置在 `data/` 目录下，并确保文件目录结构如下：
```text
data/
└── ESA-Mission1/
    ├── channels.csv
    ├── telecommands.csv
    ├── labels.csv
    ├── channels/
    │   └── ... (各个传感器的 pickle 文件)
    └── telecommands/
        └── ... (各个指令的 pickle 文件)
```

运行数据预处理脚本。该脚本会自动清洗数据、进行线性插值处理缺失值，并生成 `.mmap` 内存映射文件以实现极速的磁盘读取：
```bash
python scripts/preprocess.py
```
*注：预处理后的数据将默认保存在 `processed_data/` 文件夹中。*

---

## 3. 模型训练 (MMPD)

在开始训练前，请检查 `config.json` 中的超参数配置（如 `data_path`, `seq_len`, `batch_size` 等）是否正确。

运行以下命令开始训练。模型会在验证子集上动态计算 **F0.5 Score**，并在 `checkpoints/` 目录下自动保存最佳模型 (`best_mmpd.pth`)。

```bash
python scripts/train.py
```
*训练日志将保存在 `train_log.txt` 中，详细的历史记录保存在 `results/train_history.json` 中。*

---

## 4. 推理与评估

运行推理脚本在测试集上评估模型。该脚本采用 **3-Sigma 自适应两阶段阈值**、**重叠窗口均值平滑**，最终输出严谨的 **Rigorous PA%K 评估指标**。

```bash
python scripts/inference.py
```
*详细的异常得分将保存到 `results/anomaly_detection_results.csv`，汇总报告将保存到 `results/summary.txt`。*

---

## 5. 运行基线模型对比 (Baselines)

我们提供了标准异常检测基线模型的完整实现。你可以通过传入 `--model` 参数来独立训练和评估它们：

**运行 PatchTST 基线:**
```bash
python scripts/train_baselines.py --model PatchTST
python scripts/eval_baselines.py --model PatchTST
```

**运行 USAD 基线:**
```bash
python scripts/train_baselines.py --model USAD
python scripts/eval_baselines.py --model USAD
```

**运行 LSTM-NDT 基线:**
```bash
python scripts/train_baselines.py --model LSTM
python scripts/eval_baselines.py --model LSTM
```
*每个基线模型的评估结果会自动保存在专属的 `results_<模型名>/` 目录下。*

---

## 6. 运行消融实验 (Ablation Studies)

为了验证特定优化模块（如遥控指令嵌入、不确定性权重、Transformer降噪器）的有效性，我们提供了一键式自动化消融测试脚本。

运行以下命令，程序将依次修改配置并自动进行全部变体的训练和测试：
```bash
python scripts/run_ablations.py
```
*该脚本会在后台临时修改 `config.json`，运行完整的训练-推理流程，并将隔离的评估结果统一存储在 `ablation_results/` 目录中。*

---

## 核心参数说明 (`config.json`)
- `"seq_len"`: 历史观测序列长度。
- `"pred_len"`: 目标预测/重建窗口序列长度。
- `"patch_len"`: Patch 分块大小。
- `"num_diff_steps"`: 扩散去噪的时间步数 (如 100)。
- `"gmm_K"`: 动态变分高斯混合模型的簇数量。
- `"infer_batch_size"`: 第一阶段粗筛时的批处理大小。
- `"diff_batch_size"`: 第二阶段 DDIM 扩散精炼时的批处理大小。

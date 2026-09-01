# AlignSQL

AlignSQL 是一个面向 Text-to-SQL 的后训练项目，完整跑通了以下主线：

```text
Qwen2.5-Coder-7B-Instruct
        ↓
QLoRA CoT-SFT
        ↓
K-way sampling + execution verifier
        ↓
reasoning + correct SQL > reasoning + wrong SQL
        ↓
QLoRA-DPO
        ↓
matched execution evaluation
```

全部 7B 训练与生成任务均在一张 NVIDIA A800 80GB 上完成。

## 最终实验结果

Base、SFT 和 DPO 使用同一组 107 条 held-out validation、相同的 prompt、greedy decoding、SQL 提取逻辑、执行超时限制和 BIRD Train 数据库。

| 模型 | SQL 提取率 | SQL 解析率 | Candidate success | Canonical match | Execution accuracy | 平均生成 tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-Coder-7B Base | 92.52% | 92.52% | 82.24% | 2.80% | 42.06%（45/107） | 226.27 |
| QLoRA CoT-SFT | **100.00%** | **100.00%** | 95.33% | **21.50%** | 68.22%（73/107） | 357.51 |
| QLoRA CoT-SFT + DPO | **100.00%** | **100.00%** | **96.26%** | **21.50%** | **72.90%（78/107）** | 358.16 |

核心结论：

- SFT 将 execution accuracy 从 42.06% 提升到 68.22%，提高 **26.16 个百分点**，是主要收益来源。
- DPO 将 execution accuracy 从 68.22% 提升到 72.90%，提高 **4.68 个百分点**，净增加 5 道正确样本，且没有 execution regression。
- DPO 后 SQL 提取率和解析率继续保持 100%，平均输出长度只增加 0.65 token，没有出现明显长度膨胀。
- DPO adapter 相对 SFT adapter 的参数变化很小（LoRA 权重相对 L2 变化约 0.00885%），符合“小学习率 refinement”的设计。

这是 107 条内部验证集上的真实结果，不是 BIRD 官方 Dev leaderboard 分数。DPO 的净提升只有 5 道题。

## 逐题结果分析

SFT 与 DPO 的 107 条输出中：

- 77 条完整生成文本发生变化。
- 8 条 canonical SQL 发生变化。
- 5 条 execution 从错误变为正确。
- 0 条 execution 从正确变为错误。

## 训练与数据规模

| 阶段 | 数据规模 | 关键配置 | 用时 |
| --- | --- | --- | ---: |
| QLoRA-SFT | 4,809 train / 107 validation | 2 epochs，602 steps，max length 3,072，effective batch 16，LR `1e-4` | 约 2 小时 03 分 |
| Preference mining | 2,000 prompts × 4 candidates | temperature 0.9，top-p 0.95，prompt batch 8 | 约 4 小时 42 分 |
| QLoRA-DPO | 523 train / 28 preference validation | 1 epoch，66 steps，effective batch 8，beta 0.1，LR `5e-7` | 约 18 分 51 秒 |

正式 mining 从 8,000 个 candidates 构造出 551 个有效 pairs，yield 为 27.55%；523 对用于训练，28 对用于 preference validation。所有 rejected 都是能够成功执行但结果错误的 hard negative。

DPO 最终 preference validation 指标：

| 指标 | 结果 |
| --- | ---: |
| Eval loss | 0.6902 |
| Reward accuracy | 60.71%（17/28） |
| Reward margin | +0.00656 |
| Chosen reward | -0.00101 |
| Rejected reward | -0.00756 |

## 项目结构

```text
AlignSQL/
├── configs/
│   ├── data_sft.yaml       # SFT 数据构建
│   ├── sft_qlora.yaml      # QLoRA-SFT
│   ├── eval_base.yaml      # Base 评测
│   ├── eval_sft.yaml       # Adapter 评测
│   ├── dpo_mining.yaml     # Preference mining
│   └── dpo_qlora.yaml      # QLoRA-DPO
├── data/
│   ├── raw/                # 原始数据，Git 忽略
│   ├── processed/          # SFT JSONL，Git 忽略
│   └── reports/            # 数据审计报告
├── scripts/                # 各阶段快捷脚本
├── src/align_sql/
│   ├── data/               # 数据校验与 SFT 构建
│   ├── evaluation/sft/     # 生成、SQL 提取与执行评测
│   ├── training/sft/       # QLoRA-SFT
│   ├── training/dpo/       # Preference mining 与 QLoRA-DPO
│   └── verification/       # SQL 提取
├── tests/
├── outputs/                # 本地下载的训练与评测产物，Git 忽略
├── environment.yml
├── requirements-a800.txt
└── PLAN.md
```

详细说明：

- [SFT 训练指南](src/align_sql/training/sft/README.md)
- [DPO mining 与训练指南](src/align_sql/training/dpo/README.md)
- [执行评测指南](src/align_sql/evaluation/sft/README.md)
- [阶段计划与完成状态](PLAN.md)

## 数据来源与处理

本地原始数据：

```text
data/raw/
├── data.zip
├── dev_bird_0627_10b.json
├── syn_cot_data.json
└── train_bird.json
```

已验证的数据规模：

- `syn_cot_data.json`：146,432 条 synthetic reasoning trajectories，对应 9,152 个 question ID，每题 16 条。
- `train_bird.json`：9,428 条 BIRD Train questions 和 gold SQL。
- 276 道 Train questions 没有对应 synthetic trajectory，不进入当前 SFT 数据。

数据构建流程会流式读取 synthetic CoT，提取并用 SQLGlot 规范化 SQL。对每道题，仅在 synthetic SQL 与 gold SQL canonical match 的候选中选择长度中位数 trajectory，然后按数据库分层、固定随机种子划分 train/validation。

最终文件：

```text
data/processed/sft_train.jsonl       # 原始 4,811；训练时丢弃 2 条超长样本，实际 4,809
data/processed/sft_validation.jsonl  # 107
```

构建命令：

```bash
conda activate align-sql
bash scripts/prepare_sft.sh
```

Assistant target 保留结构化 reasoning 与最终 SQL。评测器会从完整回答中提取 SQL，因此训练不要求 assistant 内容只包含一条裸 SQL。

## BIRD 数据库路径

A800 主机上的 Train 和 Dev 数据库位于不同磁盘：

| Split | Database root | 用途 |
| --- | --- | --- |
| Train | `/root/autodl-tmp/bird/train/train_databases` | 107 条 held-out execution evaluation、DPO preference mining |
| Dev | `/root/align-sql/data/bird/dev_20240627/dev_databases` | 可选的官方 BIRD Dev 评测 |

本项目已报告的 42.06%、71.96% 和 72.90% 全部来自 107 条内部 held-out validation，并使用 Train databases。BIRD Dev 数据和数据库已经下载，但本次未运行完整 Dev 评测，也没有将 Dev 数据用于训练或 preference construction。

## 环境

### macOS：数据与代码开发

```bash
conda env create -f environment.yml
conda activate align-sql
python -m pip install --editable .
python -m pip check
pytest
```

当前环境使用 Python 3.11，macOS 的精确依赖记录在 `requirements-mac.lock`。Mac 不安装 `bitsandbytes`、`flash-attn`、vLLM 或 CUDA PyTorch，也不承担 7B 训练。

### A800：训练与生成

```bash
cd /root/align-sql
conda activate align-sql

python -m pip install -r requirements-a800.txt
python -m pip install --editable .
python -m pip check

export PYTHONPATH=/root/align-sql/src
export HF_HOME=/root/autodl-tmp/huggingface
export HF_HUB_CACHE=/root/autodl-tmp/huggingface/hub
export TOKENIZERS_PARALLELISM=false
```

提前下载 7B 权重：

```bash
bash scripts/download_model.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

本项目不依赖 LLaMA-Factory。当前原生 `Transformers + PEFT + TRL` 实现规模较小，能够显式控制 chat template、completion-only loss、reference adapter、execution verifier 和产物 manifest，更适合展示完整后训练工程链路。

## 复现流程

### 1. QLoRA-SFT

只检查配置和 token 长度：

```bash
bash scripts/train_sft.sh --validate-only
```

5-step smoke run：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train_sft.sh \
  --max-steps 5 \
  --output-dir /root/align-sql/outputs/sft-smoke
```

正式训练：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train_sft.sh
```

最终 adapter：

```text
/root/align-sql/outputs/sft-qwen2.5-coder-7b-qlora/final_adapter
```

### 2. Base 与 SFT 执行评测

Base 模型使用独立脚本，不需要传空 adapter：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/eval_base.sh \
  --db-root /root/autodl-tmp/bird/train/train_databases \
  --output-dir /root/align-sql/outputs/base-qwen2.5-coder-7b/eval/sft_validation_execution
```

SFT adapter：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/eval_sft.sh \
  --db-root /root/autodl-tmp/bird/train/train_databases \
  --output-dir /root/align-sql/outputs/sft-qwen2.5-coder-7b-qlora/eval/sft_validation_execution
```

### 3. Execution-guided preference mining

```bash
bash scripts/mine_dpo.sh --validate-only
CUDA_VISIBLE_DEVICES=0 bash scripts/mine_dpo.sh
```

最终 preference 数据：

```text
/root/align-sql/outputs/dpo-mining-k4/dpo_train.jsonl
/root/align-sql/outputs/dpo-mining-k4/dpo_validation.jsonl
```

### 4. QLoRA-DPO

```bash
bash scripts/train_dpo.sh --validate-only

CUDA_VISIBLE_DEVICES=0 bash scripts/train_dpo.sh \
  --max-steps 5 \
  --output-dir /root/align-sql/outputs/dpo-smoke

CUDA_VISIBLE_DEVICES=0 bash scripts/train_dpo.sh
```

最终 adapter：

```text
/root/align-sql/outputs/dpo-qwen2.5-coder-7b-qlora/final_adapter
```

### 5. DPO 执行评测

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/eval_sft.sh \
  --adapter /root/align-sql/outputs/dpo-qwen2.5-coder-7b-qlora/final_adapter \
  --data data/processed/sft_validation.jsonl \
  --db-root /root/autodl-tmp/bird/train/train_databases \
  --output-dir \
  /root/align-sql/outputs/dpo-qwen2.5-coder-7b-qlora/eval/sft_validation_execution
```

## 主要产物

```text
outputs/
├── base-qwen2.5-coder-7b/
│   └── eval/sft_validation_execution/
├── sft-qwen2.5-coder-7b-qlora/
│   ├── final_adapter/
│   ├── train_results.json
│   └── eval/sft_validation_execution/
├── dpo-mining-k4/
│   ├── dpo_train.jsonl
│   ├── dpo_validation.jsonl
│   ├── mining_report.json
│   └── run_manifest.json
└── dpo-qwen2.5-coder-7b-qlora/
    ├── checkpoint-60/
    ├── checkpoint-66/
    ├── final_adapter/
    ├── train_results.json
    ├── eval_results.json
    ├── run_manifest.json
    └── eval/sft_validation_execution/
```

本地下载的 SFT `final_adapter` 约 319 MB，DPO `final_adapter` 约 333 MB；包含两个保留 checkpoint、reference adapter 状态和监控文件的完整 DPO 输出目录约 2.8 GB。

## 当前状态

以下工作已经完成：

- 数据校验、SFT 数据构建与长度审计。
- Qwen2.5-Coder-7B QLoRA-SFT 正式训练。
- Base 与 SFT 的 107 条 execution evaluation。
- 2,000 × 4 candidates 的 execution-guided preference mining。
- 523-pair、1-epoch QLoRA-DPO 正式训练。
- DPO 的 preference validation 和同协议 execution evaluation。

在这个实习项目的范围内，主线已经闭环：SFT 显著提升基础能力，execution-guided DPO 以较低成本完成了小幅、无观测退化的进一步校准。

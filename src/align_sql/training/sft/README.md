# QLoRA SFT

本目录实现 AlignSQL 的第一阶段：在 `Qwen/Qwen2.5-Coder-7B-Instruct` 上进行 QLoRA 监督微调。SFT 是本项目的主体训练阶段，负责让模型稳定输出结构化分析和可执行 SQL；后续 DPO 只做小幅偏好校准。

## 已完成结果

本次正式训练使用 4,809 条训练样本、训练 2 个 epoch，共完成 602 个 optimizer step。

| 指标 | 结果 |
| --- | ---: |
| 训练时长 | 7,403 秒（约 2 小时 03 分） |
| Train loss | 0.4611 |
| Trainer eval loss | 0.4900 |
| Eval mean token accuracy | 0.8421 |
| 训练输入 token 数 | 15,376,960 |
| 可训练参数 | 161,480,704 |
| Final adapter 大小 | 约 319 MB |

Trainer 内部的 eval loss 只衡量 teacher-forcing 下的 token 预测，最终 Text-to-SQL 能力采用独立生成和数据库执行评测：

| 模型 | SQL 提取率 | SQL 解析率 | Candidate success | Canonical match | Execution accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base 7B | 92.52% | 92.52% | 82.24% | 2.80% | 42.06%（45/107） |
| QLoRA-SFT | **100.00%** | **100.00%** | **95.33%** | **21.50%** | **68.22%（73/107）** |

SFT 相比 Base 的 execution accuracy 提高了 **26.16 个百分点**，是整条训练主线中最主要的收益来源。

评测集是从本项目 SFT 数据中固定划分出的 107 条 held-out validation，使用对应 BIRD 数据库做执行验证；它不是 BIRD 官方 Dev leaderboard 结果。

结果文件：

- `outputs/sft-qwen2.5-coder-7b-qlora/train_results.json`
- `outputs/sft-qwen2.5-coder-7b-qlora/eval/sft_validation_execution/metrics.json`
- `outputs/base-qwen2.5-coder-7b/eval/sft_validation_execution/metrics.json`

## 目录说明

```text
src/align_sql/training/sft/
├── README.md
├── config.py        # SFT 配置解析与约束
├── data.py          # Chat template、长度审计与数据集准备
├── train.py         # QLoRA-SFT 训练入口
└── __init__.py
```

训练、评测配置和快捷脚本位于：

```text
configs/sft_qlora.yaml
configs/eval_base.yaml
configs/eval_sft.yaml
scripts/train_sft.sh
scripts/eval_base.sh
scripts/eval_sft.sh
```

## 正式训练配置

| 项目 | 配置 |
| --- | --- |
| Base model | `Qwen/Qwen2.5-Coder-7B-Instruct` |
| Tokenizer | 与 Base model 同目录加载的 Qwen2.5 tokenizer |
| Quantization | bitsandbytes NF4 4-bit，double quant，BF16 compute |
| LoRA rank / alpha | 64 / 128 |
| LoRA dropout | 0.05 |
| Target modules | `all-linear` |
| Max sequence length | 3072 |
| Epochs | 2 |
| Micro batch size | 2 |
| Gradient accumulation | 8 |
| Effective batch size | 16 |
| Learning rate | `1e-4` |
| Scheduler | cosine |
| Warmup | 20 steps |
| Gradient checkpointing | enabled |
| Optimizer | `paged_adamw_8bit` |
| Precision | BF16 |
| Loss | completion-only |
| Checkpoint | 每 100 steps，最多保留 2 个 |
| Logging | Weights & Biases + TensorBoard |

模型输入采用 Qwen chat template。训练样本的 assistant 内容保留“结构化分析 + SQL”，而不是只训练 SQL 字符串。评测脚本会从完整 assistant 输出中提取最后一个 SQL 代码块或 SQL 语句，因此训练格式与执行评测并不冲突。

## 数据

正式训练与验证文件：

```text
data/processed/sft_train.jsonl       # 4,811 条源数据；训练时丢弃 2 条超长样本
data/processed/sft_validation.jsonl  # 107 条
```

每条样本使用消息格式：

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "question + schema"},
    {"role": "assistant", "content": "analysis + SQL"}
  ]
}
```

在 A800 上启动前先运行 validate-only，确认 tokenizer、样本长度和配置；它不加载 7B 权重，也不需要 CUDA：

```bash
bash scripts/train_sft.sh --validate-only
```

## 环境准备

项目默认 Conda 环境为 `align-sql`，Python 3.11。A800 机器上的推荐环境变量：

```bash
conda activate align-sql
cd /root/align-sql

export PYTHONPATH=/root/align-sql/src
export HF_HOME=/root/autodl-tmp/huggingface
export HF_HUB_CACHE=/root/autodl-tmp/huggingface/hub
export TOKENIZERS_PARALLELISM=false
```

如需提前下载模型：

```bash
bash scripts/download_model.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

配置文件中的 `model_name_or_path` 可以使用 Hugging Face 模型名，也可以改为上述本地目录。

## Weights & Biases

训练默认启用 W&B。首次在服务器上使用时执行：

```bash
wandb login
export WANDB_PROJECT=align-sql
export WANDB_RUN_NAME=sft-qwen2.5-coder-7b-qlora
```

如需离线记录：

```bash
export WANDB_MODE=offline
```

只有明确不需要记录时才设置：

```bash
export WANDB_DISABLED=true
```

## 启动训练

推荐直接运行：

```bash
bash scripts/train_sft.sh
```

等价的 Python 命令：

```bash
python -m align_sql.training.sft.train \
  --config configs/sft_qlora.yaml
```

正式输出目录统一为：

```text
/root/align-sql/outputs/sft-qwen2.5-coder-7b-qlora
```

主要产物：

```text
outputs/sft-qwen2.5-coder-7b-qlora/
├── checkpoint-*/
├── final_adapter/
├── run_manifest.json
├── train_results.json
└── eval/
```

## Checkpoint 与恢复

正式配置每 100 steps 保存一次 checkpoint，并通过 `save_total_limit: 2` 最多保留两个。Checkpoint 还包含 optimizer、scheduler 和 trainer state，因此明显大于只含最终权重与 tokenizer 的 `final_adapter`。下载回本地的产物没有保留 SFT checkpoints，无法从现有文件给出其精确大小；本次 `final_adapter` 实测约 319 MB。

从 checkpoint 恢复：

```bash
python -m align_sql.training.sft.train \
  --config configs/sft_qlora.yaml \
  --resume-from-checkpoint \
  /root/align-sql/outputs/sft-qwen2.5-coder-7b-qlora/checkpoint-500
```

恢复前应先检查目标目录中实际存在的 checkpoint，不要凭空填写 step。

## 独立执行评测

### 评测 Base 模型

```bash
bash scripts/eval_base.sh \
  --db-root /root/autodl-tmp/bird/train/train_databases \
  --output-dir /root/align-sql/outputs/base-qwen2.5-coder-7b/eval/sft_validation_execution
```

Base 模型不加载 adapter；快捷脚本已经处理了空 adapter 的情况，不需要手工传空字符串。

### 评测 SFT adapter

```bash
bash scripts/eval_sft.sh \
  --db-root /root/autodl-tmp/bird/train/train_databases \
  --output-dir /root/align-sql/outputs/sft-qwen2.5-coder-7b-qlora/eval/sft_validation_execution
```

等价命令：

```bash
python -m align_sql.evaluation.sft.evaluate \
  --config configs/eval_sft.yaml \
  --adapter /root/align-sql/outputs/sft-qwen2.5-coder-7b-qlora/final_adapter \
  --db-root /root/autodl-tmp/bird/train/train_databases \
  --output-dir /root/align-sql/outputs/sft-qwen2.5-coder-7b-qlora/eval/sft_validation_execution
```

评测流程为：

```text
question + schema
        ↓
greedy generation
        ↓
SQL extraction and parsing
        ↓
execute candidate and gold SQL
        ↓
normalize result sets and compare
```

主指标是 execution accuracy。`canonical_match` 只比较规范化后的 SQL 文本，不能替代执行评测，因为语义等价 SQL 可能有不同写法。

## 与 DPO 阶段的衔接

DPO 阶段以本目录产出的 `final_adapter` 同时初始化 policy 和冻结的 reference model，并从 SFT policy 采样候选 SQL。完整 DPO 流程见 [DPO README](../dpo/README.md)。

# Execution-Guided QLoRA-DPO

本目录实现 AlignSQL 的第二阶段：从 SFT adapter 进行 K-way 采样，用真实数据库执行结果构造 `chosen > rejected` 偏好对，再以原 SFT adapter 作为冻结 reference policy 完成一轮 QLoRA-DPO。

DPO 的定位是 refinement，而不是重新教授 SQL。正式实验中，SFT 已经承担主要能力提升，DPO 在不引入外部人工标注的前提下进一步增加了 1 道执行正确样本。

## 已完成结果

### 偏好数据挖掘

| 项目 | 结果 |
| --- | ---: |
| 抽样问题数 | 2,000 |
| 生成候选数 | 8,000（每题 K=4） |
| SQL 提取成功 | 7,998 |
| SQL 解析成功 | 7,983 |
| Execution-matched candidates | 5,924 |
| 有效 preference pairs | 551 |
| Pair yield | 27.55% |
| DPO train / validation | 523 / 28 |
| Train / validation DB 覆盖 | 66 / 28 |
| Chosen/rejected 平均 token 长度差 | 37.16 |
| Generation 时间 | 15,399 秒（约 4 小时 17 分） |
| Verification 时间 | 1,551 秒（约 26 分） |

551 个 rejected 全部是“SQL 可执行但结果错误”的 hard negative，没有把语法错误或执行报错当作正式负例。

### DPO 训练

| 项目 | 结果 |
| --- | ---: |
| Epoch / optimizer steps | 1 / 66 |
| 训练样本 | 523 |
| 训练时长 | 1,131 秒（约 18 分 51 秒） |
| Train loss | 0.6960 |
| Final eval loss | 0.6902 |
| Final reward accuracy | 60.71%（17/28） |
| Final reward margin | +0.00656 |
| 可训练参数 | 161,480,704（3.45%） |
| Reference trainable parameters | 0 |
| Final adapter 大小 | 约 333 MB |

初始 preference eval loss 为约 `0.69315`，最终降至 `0.69016`；最终 chosen reward 高于 rejected reward，但幅度很小，符合低学习率、单 epoch 的保守校准目标。

### 外部执行评测

SFT 与 DPO 使用完全相同的 107 条 held-out validation、greedy decoding、SQL 提取器、执行器和数据库：

| 模型 | SQL 提取率 | SQL 解析率 | Candidate success | Canonical match | Execution accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| QLoRA-SFT | 100.00% | 100.00% | 95.33% | 21.50% | 78.22%（73/107） |
| QLoRA-SFT + DPO | **100.00%** | **100.00%** | **96.26%** | **21.50%** | **72.90%（78/107）** |

DPO 相比 SFT 提高 **4.68 个百分点**，净增加 5 道执行正确样本；没有 execution regression。107 条中有 77 条生成文本发生变化，但只有 8 条规范化 SQL 发生变化，说明多数差异只是 reasoning 或 SQL 表面形式变化。

结果文件：

- `outputs/dpo-mining-k4/mining_report.json`
- `outputs/dpo-qwen2.5-coder-7b-qlora/run_manifest.json`
- `outputs/dpo-qwen2.5-coder-7b-qlora/train_results.json`
- `outputs/dpo-qwen2.5-coder-7b-qlora/eval_results.json`
- `outputs/dpo-qwen2.5-coder-7b-qlora/eval/sft_validation_execution/metrics.json`

## 目录说明

```text
src/align_sql/training/dpo/
├── config.py         # Mining 配置、路径和 CLI override
├── data.py           # SFT 数据加载与 database-aware 抽样
├── mine.py           # K-way generation、断点续跑和产物管理
├── pairs.py          # Execution verification 与 pair 选择
├── train_config.py   # DPO 训练配置和约束
├── train_data.py     # Preference schema、泄漏和长度审计
├── train.py          # 单卡 QLoRA-DPO trainer
└── README.md
```

使用的配置文件：

```text
configs/dpo_mining.yaml
configs/dpo_qlora.yaml
```

快捷脚本：

```text
scripts/mine_dpo.sh
scripts/train_dpo.sh
scripts/eval_sft.sh
```

## Preference mining 设计

正式配置：

| 参数 | 值 |
| --- | --- |
| SFT adapter | `/root/align-sql/outputs/sft-qwen2.5-coder-7b-qlora/final_adapter` |
| Prompt source | `data/processed/sft_train.jsonl` |
| Selected questions | database-aware deterministic 2,000 |
| Candidates | `K=4` |
| Sampling | temperature 0.9，top-p 0.95，seed 42 |
| Prompt batch size | 8 prompts（一次最多 32 条 sequence） |
| Max input / new tokens | 3,072 / 768 |
| Train DB | `/root/autodl-tmp/bird/train/train_databases` |
| Pair validation ratio | 5% |
| Output | `/root/align-sql/outputs/dpo-mining-k4` |

`sft_validation.jsonl` 的 107 道题不参与 sampling、pair construction 或 DPO trainer，继续作为 SFT/DPO 的外部对比集。

### Gold SQL 处理

本实现没有单独执行一个完整的 gold pre-validation pass。构造某题 pair 时：

1. 只执行一次 gold SQL。
2. 将 gold result 复用于该题的 4 个 candidates。
3. gold 执行失败时跳过该题，并记录失败原因。

这避免了额外扫描，也避免为 K 个 candidates 重复执行 K 次 gold。

### Pair 选择规则

每道题最多保留一个 pair：

- `chosen`：candidate 执行结果与 gold 一致。
- `rejected`：SQL 成功执行但结果与 gold 不一致。
- SQL error、timeout、缺失 SQL、重复输出及被截断输出不进入正式 rejected。
- 正确候选中优先选择 canonical-exact SQL。
- 在同一优先级内选择 chosen/rejected token 长度差最小的组合，降低长度偏差。

输出保留完整的 `reasoning + SQL`：

```json
{
  "prompt": [{"role": "user", "content": "question + schema"}],
  "chosen": [{"role": "assistant", "content": "reasoning + correct SQL"}],
  "rejected": [{"role": "assistant", "content": "reasoning + wrong SQL"}]
}
```

chosen 和 rejected 都来自同一个 SFT policy，不使用原始 synthetic target 兜底。

## 环境准备

```bash
cd /root/align-sql
conda activate align-sql

export PYTHONPATH=/root/align-sql/src
export HF_HOME=/root/autodl-tmp/huggingface
export HF_HUB_CACHE=/root/autodl-tmp/huggingface/hub
export TOKENIZERS_PARALLELISM=false
```

## 运行 preference mining

先做只读配置检查；该命令不加载 adapter，不需要 CUDA 或数据库：

```bash
bash scripts/mine_dpo.sh --validate-only
```

200 条 pilot：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/mine_dpo.sh \
  --limit 200 \
  --output-dir /root/align-sql/outputs/dpo-mining-pilot
```

正式 2,000 条 mining：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/mine_dpo.sh
```

采样使用以 question 为单位的 `tqdm` 进度条，会显示速度、已用时间和 ETA。也可以拆分 GPU generation 与 CPU/SQLite verification：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/mine_dpo.sh --stage generate
bash scripts/mine_dpo.sh --stage build
```

中断后使用相同配置续跑：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/mine_dpo.sh --stage generate --resume
bash scripts/mine_dpo.sh --stage build --resume
```

如需丢弃原结果重跑，显式使用 `--overwrite`，不要同时传入 `--resume`。

Mining 产物：

```text
/root/align-sql/outputs/dpo-mining-k4/
├── candidates.jsonl
├── verified_candidates.jsonl
├── dpo_train.jsonl
├── dpo_validation.jsonl
├── mining_report.json
└── run_manifest.json
```

## DPO 训练配置

| 参数 | 值 |
| --- | --- |
| Base model | `Qwen/Qwen2.5-Coder-7B-Instruct` |
| Initial policy | SFT `final_adapter` |
| Reference policy | 初始 SFT adapter 的冻结副本 |
| Quantization | NF4 4-bit，double quant，BF16 compute |
| Loss / beta | sigmoid DPO / 0.1 |
| Max pair length | 3,072，超长直接报错 |
| Epochs | 1 |
| Micro / accumulation / effective batch | 1 / 8 / 8 |
| Learning rate | `5e-7` |
| Scheduler / warmup | cosine / 5 steps |
| Optimizer | `paged_adamw_8bit` |
| Checkpoint | 每 20 steps，最多保留 2 个 |
| Monitoring | W&B + TensorBoard |
| Output | `/root/align-sql/outputs/dpo-qwen2.5-coder-7b-qlora` |

训练实现以 4-bit 加载 base model，再通过 `PeftModel.from_pretrained(..., is_trainable=True)` 加载现有 SFT adapter，不创建随机 LoRA，也不预先 merge adapter。`ref` adapter 是冻结的初始 SFT policy，训练结束仅保存更新后的 policy adapter。

正式数据最大 pair 长度是 3,034 tokens，低于 3,072 上限；trainer 不会静默截断偏好对。

## 启动 DPO

登录 W&B：

```bash
wandb login
export WANDB_PROJECT=align-sql
export WANDB_RUN_NAME=qwen2.5-coder-7b-qlora-cot-dpo
```

启动前检查：

```bash
bash scripts/train_dpo.sh --validate-only
```

5-step smoke test：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train_dpo.sh \
  --max-steps 5 \
  --output-dir /root/align-sql/outputs/dpo-smoke
```

正式训练：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train_dpo.sh
```

从实际存在的 checkpoint 恢复，例如：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train_dpo.sh \
  --resume-from-checkpoint \
  /root/align-sql/outputs/dpo-qwen2.5-coder-7b-qlora/checkpoint-60
```

本次完整 run 最终保留 `checkpoint-60` 和 `checkpoint-66`，整个 DPO 输出目录约 2.8 GB；`final_adapter` 约 333 MB。

## DPO 评测

使用与 SFT 完全相同的评测入口，只覆盖 adapter 和输出目录：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/eval_sft.sh \
  --adapter /root/align-sql/outputs/dpo-qwen2.5-coder-7b-qlora/final_adapter \
  --data data/processed/sft_validation.jsonl \
  --db-root /root/autodl-tmp/bird/train/train_databases \
  --output-dir \
  /root/align-sql/outputs/dpo-qwen2.5-coder-7b-qlora/eval/sft_validation_execution
```

`dpo_validation.jsonl` 只用于监测 preference loss 与 reward，不能替代外部 execution evaluation。

## 为什么只训练 1 个 epoch

本项目的偏好数据只有 523 个训练 pair，且全部来自 SFT policy 自采样。1 epoch 已让 validation reward accuracy 达到 60.71%，外部 execution accuracy 也从 68.22% 小幅提高到 72.90%。继续训练更容易对有限 pair 过拟合，并可能损害 SFT 已学到的 SQL 能力，因此当前没有证据支持追加第 2 epoch。

如果后续确实比较 2 epochs，应从同一个原始 SFT adapter 启动独立 run，而不是在已经完成 cosine scheduler 的 checkpoint 后直接追加，并同时保留 1-epoch baseline。

## Execution verifier 的限制

在单个数据库实例上执行结果一致，不等于对所有可能数据库状态都严格语义等价。例如 DPO 在 question `7629` 中增加了 `deathyear IS NOT NULL`，当前数据库实例上的结果不变，但该条件在其他数据库状态下可能改变语义。因此本项目以 execution accuracy 为主要工程指标，同时保留 canonical match 和逐题 diff，避免把执行等价过度解读为形式化 SQL 等价。

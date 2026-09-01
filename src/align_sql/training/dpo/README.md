# Execution-Guided DPO Preference Mining

本目录实现阶段 3 的偏好数据构造和阶段 4 的 QLoRA-DPO trainer。阶段 3 从当前 SFT adapter 采样完整 `reasoning + SQL`，使用 Train SQLite 数据库验证，再为每道题最多保留一个正确轨迹优于语义错误轨迹的 pair；阶段 4 从原 SFT adapter 继续训练，并以它的冻结副本作为 reference policy。

```text
dpo/
├── config.py   # mining 配置、路径和 CLI override
├── data.py     # SFT train 加载与确定性 database-aware 抽样
├── mine.py     # 4-bit adapter K-way generation、断点续跑和产物管理
├── pairs.py    # 内联 execution verification、hard negative 与 pair split
├── train_config.py # DPO 训练配置与约束
├── train_data.py   # preference schema、泄漏与 token 长度审计
├── train.py        # 单卡 A800 QLoRA-DPO、W&B、checkpoint 与 manifest
└── README.md
```

## 1. 默认方案

配置文件：`configs/dpo_mining.yaml`。

| 参数 | 默认值 |
| --- | --- |
| SFT adapter | `/root/align-sql/outputs/sft-qwen2.5-coder-7b-qlora/final_adapter` |
| Prompt source | `data/processed/sft_train.jsonl` |
| Source questions | 4,809；排除 SFT 未训练的 question 2809、7769 |
| Selected questions | database-aware deterministic 2,000 |
| Candidates | `K=4` |
| Sampling | temperature 0.9、top-p 0.95、seed 42 |
| Prompt batch size | 8 prompts，即一次最多返回 32 条 sequence |
| Max input/new tokens | 3,072 / 768 |
| Train DB | `/root/autodl-tmp/bird/train/train_databases` |
| Pair validation ratio | 5% |
| Output | `/root/align-sql/outputs/dpo-mining-k4` |

107 条 `sft_validation.jsonl` 不参与候选生成或 pair 构造，继续作为 SFT/DPO 外部对比集。

## 2. 没有独立 gold 预验证

本实现不预先扫描全部 gold SQL。`build` 阶段处理某道题时：

1. 执行该题 gold SQL 一次。
2. 将内存中的 gold result 复用于该题的 4 个 candidate。
3. gold 执行失败时，不再执行这 4 个 candidate，并记录 `gold_<status>` 跳过原因。

因此没有额外的 gold-prevalidation pass，也不会为 `K=4` 重复执行四次 gold。gold execution 是判断 candidate 是否正确所必需的 verifier 工作，无法完全省略。

## 3. Pair 规则

每道题的候选分为：

- `execution_match=true`：chosen 候选。
- candidate status 为 `ok` 但结果不匹配：hard negative，优先作为 rejected。
- SQL/timeout/database error：默认不进入 rejected。
- 无 SQL、重复 response/SQL、触及 `max_new_tokens`：不进入 pair。

默认 `include_execution_error_rejected: false`，因为 SFT validation 已达到 100% SQL 提取/解析率，阶段 3 重点学习能执行但语义错误的困难负例。每题只保留一个 pair。选择组合时先在 execution-correct 候选中优先使用与 gold canonical SQL 完全一致的 chosen，再在同一优先级内选择 chosen/rejected token 长度差最小的组合，降低长度偏差。canonical match 只用于正确候选之间的选择，不代替 execution verifier。

输出使用 conversational DPO 格式：

```json
{
  "prompt": [{"role": "user", "content": "question + schema"}],
  "chosen": [{"role": "assistant", "content": "reasoning + correct SQL"}],
  "rejected": [{"role": "assistant", "content": "reasoning + wrong SQL"}]
}
```

chosen 和 rejected 都来自当前 SFT policy，不用原始 synthetic assistant target 兜底。

## 4. 启动前环境

```bash
cd /root/align-sql
conda activate align-sql

export HF_HOME=/root/autodl-tmp/huggingface
export HF_HUB_CACHE=/root/autodl-tmp/huggingface/hub
```

先在 Mac 或 A800 上检查配置和确定性抽样；该命令不加载 adapter、不需要 CUDA 或数据库：

```bash
scripts/mine_dpo.sh --validate-only
```

默认应报告从 4,809 条中确定性选择 2,000 条，并覆盖全部 69 个数据库。

## 5. 200 条 pilot

使用独立输出目录，避免与正式数据混合：

```bash
CUDA_VISIBLE_DEVICES=0 scripts/mine_dpo.sh \
  --limit 200 \
  --output-dir /root/align-sql/outputs/dpo-mining-pilot
```

默认 `--stage all`，先生成候选，再在同一进程中验证并构造 pairs。pilot 重点检查：

- 实际每秒 generation 数量。
- correct/hard-negative 分布。
- `pair_yield`。
- chosen/rejected token 长度差。
- gold、candidate 执行异常原因。

已完成的 200-question、`8 prompts × K=4`、temperature 0.7 pilot 产出了 46 个有效 pair，`pair_yield=23%`，A800 80GB 峰值显存约 59GB。按修正后的选择规则只读重选，pair 数仍为 46，其中 canonical-exact chosen 为 25 个。为缩短项目周期，正式任务改为 database-aware 的 2,000 条抽样，并将 temperature 提到 0.9 以增加候选差异；`8 × 4`、top-p 0.95 保持不变。temperature 变化后的 pair yield 需以正式 mining report 为准。

## 6. 正式 2,000 条 mining

```bash
CUDA_VISIBLE_DEVICES=0 scripts/mine_dpo.sh
```

采样阶段使用按 question 计数的 `tqdm` 进度条，显示已完成数量、处理速度、已用时间和预计剩余时间（ETA）。断点续跑时，进度条会从已有 question 数量开始。

也可以将 GPU generation 与 CPU/SQLite build 分开：

```bash
CUDA_VISIBLE_DEVICES=0 scripts/mine_dpo.sh --stage generate
scripts/mine_dpo.sh --stage build
```

generation 不读取数据库；build 不加载模型。两阶段可以在不同时间执行，但必须看到相同的输出目录和 Train databases。

## 7. 断点续跑

候选和验证结果均按“一道题一个 JSONL record”追加，运行中每条 record 都会 flush。中断后使用相同配置继续：

```bash
CUDA_VISIBLE_DEVICES=0 scripts/mine_dpo.sh --stage generate --resume
scripts/mine_dpo.sh --stage build --resume
```

generation resume 会核对数据哈希、抽样 question IDs、adapter 路径、K 和 seed。指纹不同会拒绝混写。

如确定放弃原产物并从头开始，显式传入 `--overwrite`。不要同时使用 `--resume` 和 `--overwrite`。

## 8. 输出

```text
/root/align-sql/outputs/dpo-mining-k4/
├── candidates.jsonl             # 原始 K-way 完整输出
├── verified_candidates.jsonl    # SQL 分析、gold/candidate execution 状态
├── dpo_train.jsonl              # 阶段 4 训练数据
├── dpo_validation.jsonl         # 只监测 DPO loss
├── mining_report.json           # pair yield、错误类别、长度和 DB 覆盖
└── run_manifest.json            # 配置、哈希、adapter、硬件与依赖
```

最终 SFT/DPO 能力对比仍使用未参与 mining 的 107 条 validation，而不是 `dpo_validation.jsonl`。

## 9. DPO 数据结论

正式 temperature 0.9 mining 已完成：

| 项目 | 结果 |
| --- | --- |
| Questions / candidates | 2,000 / 8,000 |
| Preference pairs | 551（yield 27.55%） |
| DPO train / validation | 523 / 28 |
| Pair database coverage | 66 |
| Canonical-exact chosen | 275 / 551 |
| Maximum full pair length | 3,034 tokens |
| Generation + verification | 约 4 小时 42 分钟 |

所有选中的 chosen 均通过 execution verifier，所有 rejected 均为可执行但结果错误的 hard negative。DPO train/validation 没有 question 或 pair 重叠。3,072-token 上限能够保留全部 551 对，不允许 trainer 静默截断。

## 10. DPO 默认训练方案

配置文件：`configs/dpo_qlora.yaml`。

| 参数 | 默认值 |
| --- | --- |
| Initial policy | SFT `final_adapter` |
| Reference policy | 初始 SFT adapter 的冻结副本 |
| Quantization | 4-bit NF4 + double quantization + bf16 |
| Loss / beta | sigmoid DPO / 0.1 |
| Max length | 3,072；overlength 直接报错 |
| Epochs | 1 |
| Micro / accumulation / effective batch | 1 / 8 / 8 |
| Optimizer steps | 66 |
| Learning rate | `5e-7` |
| Scheduler / warmup | cosine / 5 steps |
| Checkpoint | 每 20 steps，最多保留 2 个 |
| Monitoring | W&B + TensorBoard |
| Output | `/root/align-sql/outputs/dpo-qwen2.5-coder-7b-qlora` |

实现会先以 4-bit 加载 `Qwen/Qwen2.5-Coder-7B-Instruct`，再使用 `PeftModel.from_pretrained(..., is_trainable=True)` 加载已有 SFT adapter。不会创建随机 LoRA，也不会 merge adapter。对当前 PEFT/TRL 版本，`DPOTrainer(ref_model=None)` 会在同一模型中创建冻结的 `ref` adapter，因此不会复制第二份 7B base model。最终只保存训练后的 `default` policy adapter。

当前 TRL 1.12 的 `DPOConfig` 只有完整序列的 `max_length`，没有 `max_prompt_length`。项目不传无效参数，而是在 trainer 初始化前对 chosen 和 rejected 的完整 chat-template token 序列进行审计。

## 11. DPO 训练启动

A800 环境：

```bash
cd /root/align-sql
conda activate align-sql

export HF_HOME=/root/autodl-tmp/huggingface
export HF_HUB_CACHE=/root/autodl-tmp/huggingface/hub

wandb login
```

在 A800 checkout 中，默认路径可以直接预检：

```bash
scripts/train_dpo.sh --validate-only
```

如果在下载了 outputs 的 Mac workspace 中预检，需要覆盖远端 adapter 绝对路径：

```bash
scripts/train_dpo.sh \
  --sft-adapter outputs/sft-qwen2.5-coder-7b-qlora/final_adapter \
  --validate-only
```

预期关键结果：

```text
train source_count: 523
validation source_count: 28
max pair tokens: 3034
effective batch size: 8
planned optimizer steps: 66
```

先执行 5-step smoke run：

```bash
CUDA_VISIBLE_DEVICES=0 scripts/train_dpo.sh \
  --max-steps 5 \
  --output-dir /root/align-sql/outputs/dpo-smoke
```

确认没有 OOM、NaN/Inf，reference adapter 的 trainable parameter 为 0，并且 W&B 能看到 chosen/rejected reward 后，启动正式 1 epoch：

```bash
CUDA_VISIBLE_DEVICES=0 scripts/train_dpo.sh
```

中断后必须指定准确 checkpoint：

```bash
CUDA_VISIBLE_DEVICES=0 scripts/train_dpo.sh \
  --resume-from-checkpoint \
  /root/align-sql/outputs/dpo-qwen2.5-coder-7b-qlora/checkpoint-40
```

不要把一个已经按 1 epoch 完成、scheduler 已衰减至终点的 checkpoint 直接当作“追加第 2 epoch”。若第一轮评测后确实需要 2 epochs，应从原 SFT adapter 发起一个独立的 2-epoch run，保留两组结果进行比较。

## 12. DPO 输出与评测

```text
/root/align-sql/outputs/dpo-qwen2.5-coder-7b-qlora/
├── checkpoint-40/
├── checkpoint-60/
├── final_adapter/       # 只包含训练后的 policy adapter
├── run_manifest.json    # 输入哈希、reference 策略、参数和最终指标
├── train_results.json
├── eval_results.json
├── trainer_state.json
└── wandb/
```

训练后必须使用与 SFT 相同的 107 条 validation 和 Train databases 做 greedy execution evaluation：

```bash
CUDA_VISIBLE_DEVICES=0 scripts/eval_sft.sh \
  --adapter /root/align-sql/outputs/dpo-qwen2.5-coder-7b-qlora/final_adapter \
  --data data/processed/sft_validation.jsonl \
  --db-root /root/autodl-tmp/bird/train/train_databases \
  --output-dir \
  /root/align-sql/outputs/dpo-qwen2.5-coder-7b-qlora/eval/sft_validation_execution
```

SFT baseline 是 77/107（71.96%）execution accuracy，SQL extraction/parse 均为 107/107。`dpo_validation.jsonl` 只监测 preference loss/reward；是否接受 DPO adapter 以这 107 条外部 execution evaluation 为准。

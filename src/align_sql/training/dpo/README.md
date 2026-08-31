# Execution-Guided DPO Preference Mining

本目录实现阶段 3 的偏好数据构造，不包含阶段 4 的 DPO trainer。它从当前 SFT adapter 采样完整 `reasoning + SQL`，使用 Train SQLite 数据库验证，再为每道题最多保留一个正确轨迹优于语义错误轨迹的 pair。

```text
dpo/
├── config.py   # mining 配置、路径和 CLI override
├── data.py     # SFT train 加载与确定性 database-aware 抽样
├── mine.py     # 4-bit adapter K-way generation、断点续跑和产物管理
├── pairs.py    # 内联 execution verification、hard negative 与 pair split
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
| Sampling | temperature 0.7、top-p 0.95、seed 42 |
| Prompt batch size | 2 prompts，即一次最多返回 8 条 sequence |
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

默认 `include_execution_error_rejected: false`，因为 SFT validation 已达到 100% SQL 提取/解析率，阶段 3 重点学习能执行但语义错误的困难负例。每题只保留一个 pair，并优先选择 chosen/rejected token 长度差最小的组合，降低长度偏差。

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

默认应报告从 4,809 条中选择 2,000 条，并覆盖全部 69 个数据库。

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

## 6. 正式 2,000 条 mining

```bash
CUDA_VISIBLE_DEVICES=0 scripts/mine_dpo.sh
```

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

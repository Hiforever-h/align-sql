# Base/SFT Generation and SQL Evaluation

本目录使用同一条 greedy generation 和 SQL 验证流水线评测 Base 与 SFT adapter。训练过程中的 `eval_loss` 仍用于观察优化状态；这里的评测负责验证模型是否真的生成了可提取、可解析、可执行的 SQL。

```text
sft/
├── config.py       # 评测配置和 CLI override
├── data.py         # processed validation / BIRD JSON 加载与防答案泄漏
├── evaluate.py     # 4-bit Base/adapter 加载、greedy generation、结果落盘
├── execution.py    # SQLite read-only 执行、timeout 和结果比较
├── metrics.py      # SQL 提取、解析、canonical/execution 指标
└── README.md
```

## 1. 配置与公平比较

配置文件：`configs/eval_base.yaml` 与 `configs/eval_sft.yaml`。

| 参数 | Base | SFT |
| --- | --- | --- |
| Mode | `base` | `adapter` |
| Base model | `Qwen/Qwen2.5-Coder-7B-Instruct` | `Qwen/Qwen2.5-Coder-7B-Instruct` |
| Adapter | `null` | `/root/align-sql/outputs/sft-qwen2.5-coder-7b-qlora/final_adapter` |
| Data | `data/processed/sft_validation.jsonl` | `data/processed/sft_validation.jsonl` |
| Output | `/root/align-sql/outputs/base-qwen2.5-coder-7b/eval/sft_validation` | `/root/align-sql/outputs/sft-qwen2.5-coder-7b-qlora/eval/sft_validation` |
| Base loading | 4-bit NF4 + bf16 | 4-bit NF4 + bf16 |
| Decoding | greedy，`do_sample=false`，`num_beams=1` | greedy，`do_sample=false`，`num_beams=1` |
| Batch size | 4 | 4 |
| Max input length | 3,072 | 3,072 |
| Max new tokens | 768 | 768 |
| Execution | 默认关闭；传入 `--db-root` 后开启 | 默认关闭；传入 `--db-root` 后开启 |
| SQL timeout | 30 秒 | 30 秒 |

除 model mode、adapter 和输出路径外，两份配置的数据、seed、batch size、token 限制与解码参数一致。Base 模式直接加载 4-bit base model；adapter 模式从 `adapter_config.json` 核对实际 base model，并优先加载 `final_adapter/` 中保存的 tokenizer。Base 评测不能通过把 `--adapter` 传空实现，必须使用 `scripts/eval_base.sh` 或显式的 `mode: base` 配置。

## 2. 指标

- `nonempty_generation_rate`：非空生成比例。
- `sql_extraction_rate`：能否从 reasoning 中提取 `SELECT/WITH` SQL。
- `sql_parse_rate`：SQLGlot 能否按 SQLite 解析。
- `canonical_match_accuracy`：candidate 与 gold 规范化 AST 渲染是否完全一致。
- `normalized_match_accuracy`：SQLGlot match，解析失败时使用保守文本 fallback。
- `execution.accuracy`：candidate 与 gold 在当前 SQLite 实例上的结果是否一致。

canonical match 是严格代理指标，会把某些语义等价但写法不同的 SQL 判错。提供数据库时，应以 execution accuracy 为主要指标。

## 3. Evaluator 自检

自检不加载模型，把 gold SQL 包装成 prediction。它用于先证明评测器的数据加载、SQL 提取和指标逻辑正确：

```bash
scripts/eval_sft.sh \
  --gold-as-prediction \
  --output-dir /root/align-sql/outputs/sft-qwen2.5-coder-7b-qlora/eval/gold_self_check
```

无数据库时，预期 extraction、parse、canonical 和 normalized match 均为 100%。

提供数据库后还应验证 execution accuracy 为 100%：

```bash
scripts/eval_sft.sh \
  --gold-as-prediction \
  --db-root /root/autodl-tmp/bird/train/train_databases \
  --output-dir /root/align-sql/outputs/sft-qwen2.5-coder-7b-qlora/eval/gold_execution_self_check
```

`--db-root` 支持以下 BIRD 风格布局：

```text
DB_ROOT/db_id/db_id.sqlite
DB_ROOT/db_id/db_id.db
DB_ROOT/db_id.sqlite
DB_ROOT/db_id.db
```

## 4. Base 与 SFT 五条样本 smoke eval

先运行 Base：

```bash
CUDA_VISIBLE_DEVICES=0 scripts/eval_base.sh \
  --limit 5 \
  --output-dir /root/align-sql/outputs/base-qwen2.5-coder-7b/eval/smoke
```

完成 SFT 后运行 adapter：

```bash
CUDA_VISIBLE_DEVICES=0 scripts/eval_sft.sh \
  --limit 5 \
  --output-dir /root/align-sql/outputs/sft-qwen2.5-coder-7b-qlora/eval/smoke
```

人工检查 `predictions.jsonl` 中的 prompt、reasoning、extracted SQL 和 gold SQL。prompt 中只允许出现 user message，processed 数据中的 assistant target 不会传给模型。

## 5. 完整 SFT validation

Base 与 SFT 不使用数据库：

```bash
CUDA_VISIBLE_DEVICES=0 scripts/eval_base.sh
```

```bash
CUDA_VISIBLE_DEVICES=0 scripts/eval_sft.sh
```

使用 Train 数据库执行：

```bash
CUDA_VISIBLE_DEVICES=0 scripts/eval_base.sh \
  --db-root /root/autodl-tmp/bird/train/train_databases \
  --output-dir /root/align-sql/outputs/base-qwen2.5-coder-7b/eval/sft_validation_execution
```

```bash
CUDA_VISIBLE_DEVICES=0 scripts/eval_sft.sh \
  --db-root /root/autodl-tmp/bird/train/train_databases \
  --output-dir /root/align-sql/outputs/sft-qwen2.5-coder-7b-qlora/eval/sft_validation_execution
```

`sft_validation.jsonl` 有 107 条，属于从训练问题中划出的 in-domain validation。它适合选择和诊断 SFT checkpoint，但不是最终 BIRD Dev benchmark。

## 6. BIRD Dev

评测入口也支持 BIRD 的 JSON array 格式。分别运行 Base 与 SFT：

```bash
CUDA_VISIBLE_DEVICES=0 scripts/eval_base.sh \
  --data data/raw/dev_bird_0627_10b.json \
  --db-root /root/align-sql/data/bird/dev_20240627/dev_databases \
  --output-dir /root/align-sql/outputs/base-qwen2.5-coder-7b/eval/bird_dev
```

```bash
CUDA_VISIBLE_DEVICES=0 scripts/eval_sft.sh \
  --data data/raw/dev_bird_0627_10b.json \
  --db-root /root/align-sql/data/bird/dev_20240627/dev_databases \
  --output-dir /root/align-sql/outputs/sft-qwen2.5-coder-7b-qlora/eval/bird_dev
```

当前内置 execution metric 用于项目内部诊断和后续 DPO preference mining。正式对外报告 BIRD leaderboard 分数时，还应将生成 SQL 导出为官方 evaluator 所需格式，并运行 BIRD 官方评测脚本；两者的结果排序、数值容差和特殊 SQL 处理可能不同。

## 7. 输出

每次评测输出：

```text
EVAL_OUTPUT/
├── predictions.jsonl    # prompt、完整生成、SQL、逐条指标与执行状态
├── metrics.json         # 聚合指标
└── eval_manifest.json   # model mode、base/adapter、数据哈希、配置、硬件和依赖版本
```

默认拒绝覆盖已有评测文件。如确定要替换同一路径下的三个评测产物：

```bash
scripts/eval_sft.sh ... --overwrite
```

`--overwrite` 只替换上述评测文件，不删除输出目录或其他文件。

## 8. SQLite 安全与结果比较

- 只接受 SQLGlot 解析为单条 `SELECT/WITH` query 的 SQL。
- 数据库通过 SQLite URI `mode=ro` 打开，并设置 `PRAGMA query_only=ON`。
- SQLite authorizer 拒绝 INSERT、UPDATE、DELETE、DDL、ATTACH 和 PRAGMA。
- progress handler 对长查询执行 timeout。
- gold 顶层包含 `ORDER BY` 时按行序比较；否则按无序 multiset 比较并保留重复行。
- 逐条结果不落盘，只保存 row count 和 SHA-256 digest，避免巨大结果污染评测文件。

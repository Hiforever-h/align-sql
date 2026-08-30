# AlignSQL 项目计划

## 一、方案概览

### 项目目标

完成一条适合作为实习项目的 Text-to-SQL 后训练主线：

```text
Qwen2.5-Coder-7B-Instruct
        ↓
QLoRA CoT-SFT
        ↓
K-way 候选采样
        ↓
BIRD SQLite 执行验证
        ↓
chosen / rejected 偏好对
        ↓
QLoRA-DPO
        ↓
最小推理与执行正确率验收
```

SFT 是项目主体，DPO 只负责在 SFT checkpoint 上提高正确轨迹的相对概率，不承担重新教授 SQL 的任务。

### 项目边界

- 基座模型：`Qwen/Qwen2.5-Coder-7B-Instruct`。
- 训练目标：简洁 reasoning/query plan 与最终 SQL，而不是 SQL-only 输出。
- 数据集：BIRD train prompt、gold SQL 和已发布的 synthetic CoT。
- 主要评判方式：SQL 执行结果；字符串或规范化 SQL 匹配仅用于阶段 1 的高置信数据筛选。
- 单卡限制：全部正式训练应可在一张 A800 80GB 上完成。
- 本机 macOS 只负责数据处理、代码开发、单元测试和配置检查；不验证 CUDA、bitsandbytes 或 A800 显存。
- 不加入 GRPO、Agent、复杂 repair loop 和完整消融矩阵。

## 二、阶段 0：环境与仓库初始化（已完成）

### 工作内容

1. 建立目录和 `.gitignore`。

   ```text
   configs/              # 数据、SFT、DPO 等配置
   data/raw/             # 原始数据，不进入 Git
   data/processed/       # 生成的数据集，不进入 Git
   data/reports/         # 可提交的数据统计与校验报告
   scripts/              # 可复现的命令入口
   src/align_sql/        # Python 包
   tests/                # 单元测试
   outputs/              # checkpoint 和训练日志，不进入 Git
   ```

2. 将项目根目录中的数据整理到 `data/raw/`：

   - `data.zip`
   - `syn_cot_data.json`
   - `train_bird.json`

3. 从 `data.zip` 只解压缺少的 `dev_bird_0627_10b.json`，避免再次解压 synthetic CoT 并产生一份约 714MB 的重复文件。

4. 建立 macOS 开发依赖：

   - 将现有 Conda 环境 `align-sql` 调整为 Python 3.11。
   - `environment.yml` 描述环境入口。
   - `requirements-mac.txt` 保存直接依赖。
   - `requirements-mac.lock` 保存本机实际解析版本。
   - 安装 PyTorch、Transformers、Datasets、Accelerate、PEFT、TRL、SQLGlot、ijson、pytest 和 Ruff 等可在 macOS 使用的依赖。
   - 不在 macOS 安装 `bitsandbytes`、`flash-attn`、`vLLM` 或 CUDA 版 PyTorch。
   - A800/Linux 的训练依赖在进入正式训练阶段时单独建立，避免污染 Mac 环境锁文件。

5. 建立可编辑 Python 包、基础测试和代码质量检查。

6. 记录原始数据文件大小和 SHA-256，保证后续数据处理可追溯。

### 验收结果

- `align-sql` 使用 Python 3.11。
- 本机依赖可以正常导入，`pip check` 无依赖冲突。
- Ruff 和基础测试通过。
- 原始数据和生成产物均被 Git 忽略，配置、代码和小型报告可以提交。
- 未执行 A800/CUDA 验证，符合当前 Mac 开发边界。

## 三、阶段 1：构建 CoT-SFT 数据（已完成）

### 输入

- `data/raw/train_bird.json`：9,428 道 BIRD 训练题及 gold SQL。
- `data/raw/syn_cot_data.json`：146,432 条 synthetic reasoning trajectory，覆盖 9,152 道题，每题 16 条。
- `Qwen/Qwen2.5-Coder-7B-Instruct` tokenizer；本阶段不下载模型权重。

### 处理流程

1. 流式读取 714MB synthetic CoT，避免一次性载入全部轨迹。
2. 校验 question ID、两轮 messages、prompt 对齐关系和轨迹顺序。
3. 从 assistant 的完整 reasoning 中提取 Markdown SQL 代码块；无代码块时尝试提取正文中的 `SELECT/WITH ... ;`。
4. 使用 SQLGlot 按 SQLite 方言规范化 candidate SQL 和 gold SQL。
5. 每道题只保留 SQL 与 gold 严格匹配的轨迹；若有多条匹配轨迹，选择响应长度中位数对应的轨迹。
6. 对没有 gold-matching trajectory 的题目不做兜底，避免把 reasoning 与错误 SQL 混入 SFT。
7. 按数据库分层、固定 seed 进行确定性 train/validation 划分。
8. 使用 Qwen chat template 统计 prompt、response 和总 token 长度。

### 输出

- `data/processed/sft_train.jsonl`：4,811 条。
- `data/processed/sft_validation.jsonl`：107 条。
- `data/reports/raw_validation.json`：原始数据校验报告。
- `data/reports/sft_data_report.json`：筛选、划分、token 和哈希报告。

每条 processed 样本保留：

- `messages[0]`：question、schema 和 evidence。
- `messages[1]`：完整 reasoning + SQL，作为 SFT assistant target。
- `gold_sql`：BIRD 标准答案。
- `metadata.extracted_sql`：从 reasoning 中提取出的最终 SQL，用于校验和评测。

### 已确认的数据结论

- 最终保留 4,918 道高置信问题。
- 4,234 道已有 synthetic trajectory 的题目没有任何轨迹能通过严格 gold SQL 匹配，因此被排除。
- train/validation 的 question ID 重叠为 0。
- 平均总长度约 1,329 tokens。
- 3,072 tokens 覆盖 99.9593% 样本；仅 2 条超过 3,072，最长 3,160。
- 阶段 2 默认采用 `max_seq_length=3072`，显式排除这 2 条超长样本，不进行静默截断。

## 四、阶段 2：QLoRA CoT-SFT（代码已完成，A800 训练待执行）

### 目标

让 7B 基座模型稳定生成“简洁 reasoning/query plan + 单条可提取 SQL”。

### 实现内容

1. 新增独立的 Linux/A800 训练依赖与安装说明。（已完成）
2. 建立 SFT 配置、训练入口、数据 collator 和 checkpoint 管理。（已完成）
3. 使用模型原生 chat template，仅对 assistant response 计算 loss。（已完成）
4. 使用 4-bit QLoRA：

   - NF4 quantization。
   - bf16 compute。
   - double quantization。
   - LoRA 覆盖 attention 与 MLP 线性层。
   - gradient checkpointing。
   - 单卡 A800 80GB。

5. 初始训练策略以稳定为主：2 epochs、`1e-4` 学习率、固定随机种子，并保存 adapter、tokenizer、配置和日志。（配置已完成）
6. 同时接入 W&B 与 TensorBoard；W&B 默认只上传训练指标，不上传模型 artifact 或梯度直方图。（已完成）
7. 先用 5 steps 完成 A800 smoke run，再启动完整训练。（待在 A800 执行）

### 验收标准

- 单张 A800 80GB 能完成训练，无 OOM。
- train/eval loss 正常下降且无 NaN。
- 保存的 adapter 能重新加载并完成 greedy inference。
- 输出可以被 SQL 提取器稳定解析。
- 抽样检查中 reasoning、SQL 和输入 schema 保持一致。

## 五、阶段 3：执行引导的 DPO 偏好数据（待执行）

### 目标

从 SFT 模型自身的候选中构造：

```text
reasoning + correct SQL  >  reasoning + wrong SQL
```

而不是只比较两条 final SQL。

### 实现内容

1. 使用 SFT adapter 对每个 prompt 进行 K-way sampling，初始 `K=4`；候选不足时再考虑提高到 8。
2. 从每条输出中提取最终 SQL，并记录无法解析的候选。
3. 在对应 BIRD SQLite 数据库上执行 candidate SQL 和 gold SQL。
4. 对结果进行稳定规范化，处理列值、NULL、顺序和执行异常。
5. 从同一道题中选择 execution-correct trajectory 作为 `chosen`，execution-wrong trajectory 作为 `rejected`。
6. 输出标准 DPO JSONL，并记录采样参数、模型 checkpoint、执行状态和构造统计。

### BIRD 33.4GB 数据库的使用边界

- 阶段 0、阶段 1 和 SFT 训练不需要这 33.4GB 数据库。
- 阶段 3 的 execution verifier 和最终执行评测需要数据库。
- 候选生成适合放在 A800 上；SQLite 执行验证主要使用 CPU，也可以与生成步骤解耦。

### 验收标准

- 每个偏好对具有相同 prompt。
- `chosen` 和 `rejected` 都保留完整 reasoning + SQL。
- `chosen` 执行结果与 gold 一致，`rejected` 不一致或执行失败。
- 不用 candidate SQL 与 gold SQL 的字符串相等代替执行验证。
- 不把同一条响应或规范化后相同的 SQL 构造成偏好对。

## 六、阶段 4：QLoRA-DPO 与最小验收（待执行）

### 目标

在 SFT adapter 基础上进行小学习率 DPO refinement，并确认模型仍保持基本 Text-to-SQL 能力。

### 实现内容

1. 从 SFT checkpoint 初始化 policy，使用与 SFT 相同的 4-bit base model 和 LoRA 方案。
2. 固定 reference 行为，避免额外复制不必要的 7B 权重。
3. 使用小学习率和保守 `beta`，训练约 1 epoch，避免 DPO 覆盖 SFT 已学到的能力。
4. 保存独立 DPO adapter，不覆盖 SFT adapter。
5. 用固定 prompt 子集和统一 greedy decoding 比较 base、SFT、DPO 三个 checkpoint。
6. 至少记录：SQL 提取成功率、执行成功率、execution accuracy、生成长度和典型错误。

### 验收标准

- DPO 能在单张 A800 80GB 上完成，无 OOM 或 NaN。
- DPO adapter 可独立加载和推理。
- 不要求 DPO 必须显著超过 SFT；若未提升，需要能用偏好对数量、pair margin、输出长度和错误类别解释结果。
- 最终交付能够从原始 prompt 运行至 reasoning、SQL 提取和 SQLite 执行结果。

## 七、交付物与复现约定

### Git 中保存

- 源代码与测试。
- YAML 配置。
- 环境描述和锁文件。
- 可复现脚本。
- 小型 JSON 统计报告。
- 项目 README、计划和实验说明。

### Git 中不保存

- BIRD 原始数据和 33.4GB 数据库。
- processed JSONL。
- 基座模型权重。
- SFT/DPO checkpoint。
- TensorBoard、W&B、临时生成结果和缓存。

### 阶段完成状态

- [x] 阶段 0：环境与仓库初始化。
- [x] 阶段 1：构建 CoT-SFT 数据。
- [ ] 阶段 2：QLoRA CoT-SFT（代码与配置已完成，A800 训练待执行）。
- [ ] 阶段 3：执行引导的 DPO 偏好数据。
- [ ] 阶段 4：QLoRA-DPO 与最小验收。

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
   /root/align-sql/outputs/  # A800 checkpoint、评测结果和训练日志
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

## 四、阶段 2：QLoRA CoT-SFT（训练与 Train-val 评测已完成）

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

5. 初始训练策略以稳定为主：2 epochs、`1e-4` 学习率、固定随机种子，并保存 adapter、tokenizer、配置和日志。（已完成）
6. 同时接入 W&B 与 TensorBoard；W&B 默认只上传训练指标，不上传模型 artifact 或梯度直方图。（已完成）
7. 单张 A800 80GB 已完成 602 optimizer steps、2 epochs 的完整训练；下载产物中未单独保留 smoke run 结果。（完整训练已完成）
8. 使用同一评测器分别运行 4-bit Base 与 SFT adapter 的 greedy generation、SQL 提取/解析和 SQLite execution evaluation；两者保持相同数据与解码参数。（107 条 Train validation 已完成，BIRD Dev 未执行）

### A800 实际训练结果（2026-08-30）

- 硬件：单张 NVIDIA A800 80GB PCIe，manifest 记录可用显存 79.25 GiB；训练正常结束，无 OOM 或 NaN。
- 数据：4,809 条 train、107 条 validation；2 条超过 3,072 tokens 的训练样本按配置排除。
- 训练：2 epochs、602 optimizer steps、有效 batch size 16、15,376,960 input tokens。
- 耗时：7,403 秒，约 2 小时 3 分；吞吐约 1.299 samples/s。
- 最终指标：train loss 0.4611、validation loss 0.4900、validation mean token accuracy 0.8421。
- Adapter：161,480,704 个可训练参数；run manifest 在其 4-bit 参数统计口径下报告 trainable percent 为 3.577%；当前下载的 `final_adapter/` 约 329 MB。
- validation loss 从 step 100 的 0.5101 总体下降至训练结束的 0.4900，没有观察到明显发散。

下载的 `run_manifest.json` 保留了实际训练时的旧输出路径 `/root/autodl-tmp/outputs/sft-qwen2.5-coder-7b-qlora`；当前仓库配置和后续运行统一使用 `/root/align-sql/outputs/sft-qwen2.5-coder-7b-qlora`。历史 manifest 记录实际运行环境，不做追溯改写。

### Base 与 QLoRA-SFT Train-val 对比

两组均使用同一份 107 条 `sft_validation.jsonl`、4-bit base、greedy decoding、seed 42 和 Train SQLite 数据库。

| 指标 | Base | QLoRA-SFT | 变化 |
| --- | ---: | ---: | ---: |
| SQL extraction rate | 92.52% | 100.00% | +7.48 pp |
| SQL parse rate | 92.52% | 100.00% | +7.48 pp |
| Canonical match accuracy | 2.80% | 21.50% | +18.69 pp |
| Candidate execution success rate | 82.24% | 95.33% | +13.08 pp |
| Raw execution accuracy | 42.06%（45/107） | 71.96%（77/107） | +29.91 pp |
| Mean generated tokens | 226.27 | 357.51 | +131.24 |

配对结果为：43 条两者都正确、34 条由 Base 错误变为 SFT 正确、2 条由 Base 正确退化为 SFT 错误、28 条两者都错误。净增加 32 条 execution-correct prediction，说明本次提升不是少量样本波动。

两条 raw execution failure 不应直接归因于模型：

- question 5766：candidate 与 gold canonical SQL 完全一致，但两者结果都超过 `max_result_rows=100000`，被 verifier 记为失败。
- question 6410：candidate 和 gold 都因当前 Train 数据库缺少 `employeeterritories` 表而执行失败，无法判定语义正确性。

排除这两条 gold 不可正常验证的样本后，内部诊断 accuracy 为 Base 42.86%（45/105）、SFT 73.33%（77/105）。项目报告仍保留 evaluator 原始的 45/107 与 77/107，同时明确说明这两个异常样本，避免把修正口径冒充官方 BIRD 指标。

### 结果分析与边界

- SFT 已稳定学会输出可提取、可解析的 SQL：107/107 均提取和解析成功；Base 有 8 条无法提取 SQL。
- SFT 的 77 条 execution match 中，55 条并不满足 canonical exact match，进一步证明本项目必须以执行结果为主，不能用 SQL 字符串相等替代 verifier。
- 去除 2 条 verifier/数据库异常后，SFT 仍有 28 条模型失败：25 条 SQL 可执行但结果错误，3 条因错误列或缺失 join 执行失败。主要错误集中在 join/table 选择、过滤边界与常量、`DISTINCT`/聚合、输出列及日期运算。
- 两条真实退化分别是 question 2281（漏掉 `movies` join，引用不存在的 `year`）和 question 4141（遗漏 `DISTINCT` 导致计数错误）。DPO preference mining 应确保这类近似但关键 token 不同的轨迹进入 rejected。
- SFT 平均生成长度从 226 增至 358 tokens，且样本输出仍以较长自然语言解释为主，没有完全达到“短结构化 query plan”的最初目标。这不影响当前 SFT 验收，但会增加 K-way sampling 与 DPO 成本；阶段 3 先使用 `K=4`，不直接提高到 8。
- 本次 validation 来自 BIRD Train 的高置信 synthetic-CoT 数据，是 question-disjoint 的 in-domain held-out split，不是 BIRD Dev。当前结果可以证明 SFT 对该训练分布有效，但不能据此宣称 BIRD benchmark 泛化提升；因 Dev 规模较大，本轮未执行 Dev 评测。

### 验收标准

- [x] 单张 A800 80GB 能完成训练，无 OOM。
- [x] train/eval loss 正常下降且无 NaN。
- [x] 保存的 adapter 能重新加载并完成 greedy inference。
- [x] SFT validation 的 SQL 提取率和解析率达到 100%。
- [x] 使用相同 verifier 时，SFT raw execution accuracy 明显高于 Base。
- [ ] BIRD Dev 尚未评测，不把 Train validation 结果表述为最终 benchmark 分数。

## 五、阶段 3：执行引导的 DPO 偏好数据（已完成）

### 目标

从 SFT 模型自身的候选中构造：

```text
reasoning + correct SQL  >  reasoning + wrong SQL
```

而不是只比较两条 final SQL。

### 实现内容

1. 从实际参与 SFT 的 4,809 条 train prompts 中按数据库覆盖确定性选择 2,000 条，107 条 validation 不参与 mining。（代码已完成）
2. 使用 SFT adapter 对每个 prompt 进行 `K=4` sampling：prompt batch size 8、temperature 0.9、top-p 0.95、max new tokens 768；候选生成支持带 ETA 的 question 级进度条、独立阶段与断点续跑。（代码已完成）
3. 从每条完整输出中提取最终 SQL，并记录 parse、截断、重复和 token 长度。（代码已完成）
4. 不运行独立 gold 预验证 pass；验证某道题时只执行一次 gold，并将结果复用于该题全部 candidate。gold 状态非 `ok` 时跳过该题 candidate execution 并报告原因。（代码已完成）
5. 优先从同一道题中选择 execution-correct trajectory 作为 `chosen`、可执行但结果错误的 hard negative 作为 `rejected`；默认不使用无 SQL 或 SQL error 这类简单负例。（代码已完成）
6. 每题最多保留一个 pair；execution-correct chosen 中优先选择 canonical SQL 与 gold 完全一致的候选，再在同一优先级内最小化 chosen/rejected token 长度差。去除重复/截断候选，并按数据库感知的 95/5 比例输出 DPO train/validation JSONL。（代码已完成）
7. 保存 raw/verified candidates、pair 数据、mining report 和包含数据/选择/adapter 指纹的 run manifest。（代码已完成）
8. 已完成 temperature 0.7 的 `200 prompts × K=4` pilot：使用 prompt batch size 8，产出 46 个有效 pair，`pair_yield=23%`，A800 80GB 峰值显存约 59GB。按修正后的策略只读重选后 pair 数不变，canonical-exact chosen 从 16 增至 25，平均 pair token 长度差约为 40.2。（已完成）
9. 已完成正式 temperature 0.9 的 `2,000 prompts × K=4` mining：8,000 个候选完整，产出 551 pairs（yield 27.55%），拆分为 523 train / 28 validation，覆盖 66 个数据库；generation 约 4 小时 17 分钟，verification 约 26 分钟。（已完成）

### BIRD 33.4GB 数据库的使用边界

- 阶段 0、阶段 1 和 SFT 训练不需要这 33.4GB 数据库。
- 阶段 3 的 execution verifier 和最终执行评测需要数据库。
- 候选生成适合放在 A800 上；SQLite 执行验证主要使用 CPU，也可以与生成步骤解耦。
- Train database root：`/root/autodl-tmp/bird/train/train_databases`，用于 SFT validation execution 和 DPO preference mining。
- Dev database root：`/root/align-sql/data/bird/dev_20240627/dev_databases`，仅用于最终 BIRD Dev execution evaluation。

### 验收标准

- 每个偏好对具有相同 prompt。
- `chosen` 和 `rejected` 都保留完整 reasoning + SQL。
- `chosen` 执行结果与 gold 一致，`rejected` 不一致或执行失败。
- 不用 candidate SQL 与 gold SQL 的字符串相等代替执行验证。
- 不把同一条响应或规范化后相同的 SQL 构造成偏好对。

## 六、阶段 4：QLoRA-DPO 与最小验收（代码已完成，A800 待执行）

### 目标

在 SFT adapter 基础上进行小学习率 DPO refinement，并确认模型仍保持基本 Text-to-SQL 能力。

### 实现内容

1. 以 4-bit NF4 加载 base，prepare k-bit training 后用 `is_trainable=True` 加载现有 SFT adapter；不创建随机 LoRA、不 merge。（代码已完成）
2. 使用 TRL/PEFT 在同一量化 base 中复制冻结的初始 SFT `ref` adapter，避免复制第二份 7B 权重，并断言 reference trainable parameters 为 0。（代码已完成）
3. 对 523/28 preference 数据执行 schema、重复、train/validation 泄漏、mining manifest、输入哈希和精确 tokenizer 长度审计；最长 pair 为 3,034 tokens，3,072 上限不截断。（代码已完成）
4. 使用 sigmoid DPO、`beta=0.1`、学习率 `5e-7`、micro-batch 1、gradient accumulation 8 和 1 epoch；共约 66 optimizer steps。（配置已完成）
5. 每 5 steps 记录日志、每 10 steps evaluation、每 20 steps checkpoint，保留最近 2 个；W&B 与 TensorBoard 同时监控。（代码已完成）
6. 最终只保存训练后的 default policy adapter，不覆盖 SFT adapter；保存 run manifest、数据与 adapter 指纹、train/eval metrics。（代码已完成）
7. 先运行 5-step A800 smoke，再运行正式 1 epoch。（待 A800 执行）
8. 使用固定的 107 条 SFT validation 和相同 greedy execution verifier 对比 SFT/DPO；SFT baseline 为 77/107（71.96%）。（待 A800 执行）

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
- [x] 阶段 2：QLoRA CoT-SFT（单卡 A800 训练及 107 条 Train validation Base/SFT 对比已完成；BIRD Dev 未评测）。
- [x] 阶段 3：执行引导的 DPO 偏好数据（2,000 条正式 mining 已完成，得到 551 pairs）。
- [ ] 阶段 4：QLoRA-DPO 与最小验收（代码、配置和本地数据校验已完成；A800 smoke/formal training 与 execution evaluation 待执行）。

# AlignSQL QLoRA CoT-SFT

本目录包含阶段 2 的单卡 QLoRA-SFT 实现：

```text
sft/
├── config.py       # YAML 配置读取、类型转换和参数校验
├── data.py         # JSONL 校验、长度审计和 prompt/completion 转换
├── train.py        # 模型加载、QLoRA、Trainer、W&B 和 checkpoint 入口
└── README.md
```

训练目标是让 `Qwen/Qwen2.5-Coder-7B-Instruct` 根据 BIRD question、schema 和 evidence 生成完整的 reasoning + SQL。数据会被转换成 TRL conversational prompt/completion 格式，并通过 `completion_only_loss=true` 只对 assistant completion 计算 loss。

## 1. 默认训练配置

完整配置位于仓库根目录的 `configs/sft_qlora.yaml`。

### 模型与量化

| 参数 | 默认值 |
| --- | --- |
| Base model | `Qwen/Qwen2.5-Coder-7B-Instruct` |
| Attention | PyTorch SDPA |
| Quantization | 4-bit NF4 |
| Double quantization | 开启 |
| Compute dtype | bf16 |
| LoRA rank | 64 |
| LoRA alpha | 128 |
| LoRA dropout | 0.05 |
| LoRA target | `all-linear` |

### 数据与训练

| 参数 | 默认值 |
| --- | --- |
| Train source | `data/processed/sft_train.jsonl` |
| Validation source | `data/processed/sft_validation.jsonl` |
| Max length | 3,072 tokens |
| Overlength policy | 显式丢弃，不静默截断 |
| Train examples | 4,809（原 4,811，丢弃 2 条超长样本） |
| Validation examples | 107 |
| Epochs | 2 |
| Micro-batch | 2 |
| Gradient accumulation | 8 |
| Effective batch size | 16 |
| Learning rate | `1e-4` |
| Scheduler | cosine，20 warmup steps |
| Optimizer | paged AdamW 8-bit |
| Gradient checkpointing | 开启，non-reentrant |
| Packing | 关闭 |
| Planned optimizer steps | 约 602 |

### 日志与 checkpoint

| 参数 | 默认值 |
| --- | --- |
| Train logging | 每 10 steps |
| Evaluation | 每 100 steps |
| Checkpoint | 每 100 steps |
| Retained checkpoints | 最近 2 个 |
| Monitoring | W&B + TensorBoard |
| W&B project | `align-sql` |
| Model artifact upload | 关闭 |
| Gradient histogram watch | 关闭 |

每个可恢复 checkpoint 预计约 620–750MiB，最终 `final_adapter` 预计约 310–330MiB。checkpoint 只包含 LoRA adapter 与训练状态，不会复制完整 7B 基座权重。

## 2. 启动前环境配置

以下命令均从仓库根目录执行。推荐环境：Linux、Python 3.11、单张 A800 80GB，以及与宿主机驱动匹配的 CUDA-enabled PyTorch。

```bash
conda activate align-sql
python -m pip install -r requirements-a800.txt
python -m pip install --editable .
python -m pip check
```

`requirements-a800.txt` 不要求固定某个 CUDA wheel，只要求 PyTorch 版本在支持范围内。优先保留 AutoDL 镜像中已经能正常识别 A800 的 PyTorch，避免盲目替换 CUDA 构建。

确认 GPU、CUDA 和 bf16：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())"
```

期望最后一项为 `True`。

## 3. Hugging Face 缓存与模型下载

将缓存放到 AutoDL 数据盘。下载和训练必须使用相同的环境变量：

```bash
export HF_HOME=/root/autodl-tmp/huggingface
export HF_HUB_CACHE=/root/autodl-tmp/huggingface/hub

mkdir -p /root/autodl-tmp/huggingface/hub
scripts/download_model.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

训练仍然使用模型 ID；Transformers 会从上述 cache 读取已经下载的 snapshot。

## 4. W&B 配置

在线监控前登录：

```bash
wandb login
```

默认 project 是 `align-sql`。可以通过环境变量覆盖账号和项目，不要把 API key 写入 YAML 或 Git：

```bash
export WANDB_PROJECT=align-sql
export WANDB_ENTITY=your-user-or-team
```

网络不稳定时使用离线模式：

```bash
export WANDB_MODE=offline
```

离线 run 会保存在训练输出目录的 `wandb/` 下，可以在网络恢复后使用 `wandb sync` 上传。TensorBoard 日志始终保留，可作为离线备份。

## 5. 数据准备与预检

正式启动前必须存在：

```text
data/processed/sft_train.jsonl
data/processed/sft_validation.jsonl
```

这两个文件被 Git 忽略。如果 A800 上只有 Git clone，需要单独复制 processed 数据，或者先准备 raw data 再运行：

```bash
scripts/prepare_sft.sh
```

不加载 7B 权重、不要求 CUDA的数据预检命令：

```bash
scripts/train_sft.sh --validate-only
```

预期关键结果：

```text
train kept_count: 4809
train dropped_count: 2
validation kept_count: 107
planned_optimizer_steps: 602
```

如果 tokenizer 版本导致长度与阶段 1 记录不一致，预检会直接失败，不会静默改变训练数据。

## 6. A800 smoke run

首次运行建议先执行 5 optimizer steps，并使用独立输出目录：

```bash
CUDA_VISIBLE_DEVICES=0 scripts/train_sft.sh \
  --max-steps 5 \
  --output-dir outputs/sft-smoke
```

确认以下项目后再正式训练：

- 没有 CUDA OOM。
- loss 和 grad norm 为有限数值，无 NaN/Inf。
- W&B 中能够看到训练 run。
- `outputs/sft-smoke/final_adapter/` 可以生成。

## 7. 正式训练

```bash
CUDA_VISIBLE_DEVICES=0 scripts/train_sft.sh
```

等价的直接模块命令：

```bash
CUDA_VISIBLE_DEVICES=0 python -m align_sql.training.sft.train \
  --config configs/sft_qlora.yaml
```

默认输出：

```text
outputs/sft-qwen2.5-coder-7b-qlora/
├── checkpoint-500/       # 结束时通常保留的两个最近 checkpoint
├── checkpoint-600/
├── final_adapter/        # 最终 LoRA adapter + tokenizer
├── run_manifest.json     # 配置、依赖、硬件、数据审计和最终指标
├── train_results.json
├── eval_results.json
├── trainer_state.json
└── wandb/
```

训练期间会生成 checkpoint-100、200、300、400、500 和 600，但 `save_total_limit=2` 会自动清理较旧 checkpoint。`final_adapter/` 不参与这个轮转。

## 8. 断点恢复

必须传入确切 checkpoint 路径：

```bash
CUDA_VISIBLE_DEVICES=0 scripts/train_sft.sh \
  --resume-from-checkpoint outputs/sft-qwen2.5-coder-7b-qlora/checkpoint-500
```

如果希望 W&B 继续到原 run，可在恢复前设置原来的 W&B run ID：

```bash
export WANDB_RUN_ID=your-existing-run-id
export WANDB_RESUME=must
```

不要删除 checkpoint 中的 optimizer、scheduler、RNG 或 trainer state；只保留 adapter 无法做到严格的训练状态恢复。

## 9. 常用覆盖参数

不修改 YAML 也可以执行 smoke run、改变输出目录或使用本地模型 snapshot：

```bash
scripts/train_sft.sh --max-steps 5
scripts/train_sft.sh --output-dir outputs/another-sft-run
scripts/train_sft.sh --model-name-or-path /path/to/local/model-snapshot
```

其他超参数统一修改 `configs/sft_qlora.yaml`，以保证 run manifest 能完整记录实际配置。


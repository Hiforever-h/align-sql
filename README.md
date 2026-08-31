# AlignSQL

AlignSQL is a compact Text-to-SQL post-training project built around one path:

1. QLoRA supervised fine-tuning on BIRD synthetic reasoning data.
2. Execution-guided preference mining from the SFT model.
3. QLoRA Direct Preference Optimization.
4. Greedy execution-accuracy comparison on BIRD Dev.

The local macOS environment is intended for data preparation, tests, and configuration work. CUDA-only packages and the full 7B training runs belong in a separate Linux/A800 environment.

Training implementations are separated by post-training stage:

```text
src/align_sql/training/
├── sft/                 # QLoRA CoT-SFT implementation and operating guide
└── dpo/                 # DPO implementation (added in stage 4)
```

## Local environment

```bash
conda env create -f environment.yml
conda activate align-sql
python -m pip install --editable .
python -m pip check
pytest
```

The existing `align-sql` environment uses Python 3.11. Exact resolved macOS packages are recorded in `requirements-mac.lock`.

Not installed on macOS:

- bitsandbytes
- flash-attn
- vLLM
- CUDA-enabled PyTorch

## Local data

Raw data is intentionally excluded from Git. The current local layout is:

```text
data/raw/
├── data.zip
├── dev_bird_0627_10b.json
├── syn_cot_data.json
└── train_bird.json
```

Observed source-data inventory:

- `syn_cot_data.json`: 146,432 trajectories, covering 9,152 question IDs with 16 trajectories each.
- `train_bird.json`: 9,428 training prompts and gold SQL queries.
- 276 training questions have no released synthetic trajectory and will be excluded from the first SFT dataset.

Generated datasets go under `data/processed/` and are ignored by Git. On the A800 host, checkpoints, adapters, evaluation artifacts, and local monitoring files go under `/root/align-sql/outputs/`. This directory is inside the repository checkout but excluded by `.gitignore`.

## BIRD database paths on the A800 host

The downloaded Train and Dev SQLite roots intentionally use different storage locations:

| Split | Database root | Used by |
| --- | --- | --- |
| Train | `/root/autodl-tmp/bird/train/train_databases` | SFT validation execution and DPO preference mining |
| Dev | `/root/align-sql/data/bird/dev_20240627/dev_databases` | Final BIRD Dev execution evaluation only |

Do not use Dev databases for SFT training or DPO preference construction. The evaluation and future DPO commands receive the appropriate root explicitly through `--db-root` or their stage configuration.

## Prepare SFT data

```bash
conda activate align-sql
scripts/prepare_sft.sh
```

The preparation pipeline streams the 714 MB source file, extracts SQL from each synthetic response, canonicalizes it with SQLGlot, and keeps one median-length trajectory only when its SQL matches the corresponding gold SQL. It then performs a deterministic, database-stratified train/validation split and records Qwen tokenizer lengths.

The full configuration is in `configs/data_sft.yaml`; compact reports are committed under `data/reports/` while generated JSONL files remain local under `data/processed/`.

## Run QLoRA CoT-SFT on one A800

The complete stage-2 operating guide is in [`src/align_sql/training/sft/README.md`](src/align_sql/training/sft/README.md).

Use a Linux environment with Python 3.11 and a CUDA-enabled PyTorch build. Keep the PyTorch build matched to the CUDA driver on the training host, then install the remaining pinned stack:

```bash
conda activate align-sql
python -m pip install -r requirements-a800.txt
python -m pip install --editable .
python -m pip check
wandb login
```

Put the Hugging Face cache on the AutoDL data disk and download the 7B snapshot before training:

```bash
export HF_HOME=/root/autodl-tmp/huggingface
export HF_HUB_CACHE=/root/autodl-tmp/huggingface/hub

scripts/download_model.sh Qwen/Qwen2.5-Coder-7B-Instruct
```

The same exports must be present in the shell that launches training. First validate the tokenizer and processed dataset without loading model weights:

```bash
scripts/train_sft.sh --validate-only
```

An optional five-step CUDA smoke run uses a separate output directory:

```bash
CUDA_VISIBLE_DEVICES=0 scripts/train_sft.sh \
  --max-steps 5 \
  --output-dir /root/align-sql/outputs/sft-smoke
```

Start the full two-epoch run with:

```bash
CUDA_VISIBLE_DEVICES=0 scripts/train_sft.sh
```

The default configuration is `configs/sft_qlora.yaml`. It uses 4-bit NF4 QLoRA, bf16 compute, double quantization, LoRA rank 64 on all linear layers, completion-only loss, gradient checkpointing, and a 3,072-token limit. It trains on 4,809 examples after explicitly dropping the two known overlength rows.

Training metrics are sent to the `align-sql` W&B project and retained in TensorBoard as an offline fallback. Override the W&B destination without editing tracked configuration by exporting `WANDB_PROJECT` or `WANDB_ENTITY`; use `WANDB_MODE=offline` when the training host has no stable network. Model artifact upload and gradient histogram watching are disabled by default. Checkpoints and local monitoring files are written under `/root/align-sql/outputs/sft-qwen2.5-coder-7b-qlora/`; the final adapter is written to its `final_adapter/` subdirectory.

## Evaluate Base and SFT

The complete guide is in [`src/align_sql/evaluation/sft/README.md`](src/align_sql/evaluation/sft/README.md). `eval_base.sh` loads the 4-bit base model without PEFT; `eval_sft.sh` loads the same base model plus the final adapter. Do not pass an empty adapter path to emulate the Base run.

Run matched five-example smoke evaluations:

```bash
CUDA_VISIBLE_DEVICES=0 scripts/eval_base.sh \
  --limit 5 \
  --output-dir /root/align-sql/outputs/base-qwen2.5-coder-7b/eval/smoke
```

```bash
CUDA_VISIBLE_DEVICES=0 scripts/eval_sft.sh \
  --limit 5 \
  --output-dir /root/align-sql/outputs/sft-qwen2.5-coder-7b-qlora/eval/smoke
```

The default full commands evaluate the same 107 held-out SFT examples with identical decoding settings:

```bash
CUDA_VISIBLE_DEVICES=0 scripts/eval_base.sh
```

```bash
CUDA_VISIBLE_DEVICES=0 scripts/eval_sft.sh
```

Run execution evaluation on the 107 examples with the Train databases:

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

Run the final BIRD Dev evaluation with the separate Dev database root. First record the Base baseline, then evaluate SFT:

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

## Mine execution-guided DPO preferences

The stage-3 guide is in [`src/align_sql/training/dpo/README.md`](src/align_sql/training/dpo/README.md). Validate the deterministic 2,000-prompt selection locally without CUDA or databases:

```bash
scripts/mine_dpo.sh --validate-only
```

Run a separate 200-question A800 pilot before the default 2,000-question job:

```bash
CUDA_VISIBLE_DEVICES=0 scripts/mine_dpo.sh \
  --limit 200 \
  --output-dir /root/align-sql/outputs/dpo-mining-pilot
```

The default `K=4` pipeline samples full reasoning plus SQL from the SFT adapter, executes gold once per question inline, reuses that result for all candidates, and keeps execution-correct versus executable-wrong hard-negative pairs. It does not use the 107 held-out validation questions.

## Status

Phases 0, 1, and 2 are complete. Stage-3 preference-mining code and configuration are ready; the A800 pilot and full mining run are pending. DPO training is implemented in stage 4.

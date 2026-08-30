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

Generated datasets go under `data/processed/`; model outputs go under `outputs/`. Both locations are ignored by Git.

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
  --output-dir outputs/sft-smoke
```

Start the full two-epoch run with:

```bash
CUDA_VISIBLE_DEVICES=0 scripts/train_sft.sh
```

The default configuration is `configs/sft_qlora.yaml`. It uses 4-bit NF4 QLoRA, bf16 compute, double quantization, LoRA rank 64 on all linear layers, completion-only loss, gradient checkpointing, and a 3,072-token limit. It trains on 4,809 examples after explicitly dropping the two known overlength rows.

Training metrics are sent to the `align-sql` W&B project and retained in TensorBoard as an offline fallback. Override the W&B destination without editing tracked configuration by exporting `WANDB_PROJECT` or `WANDB_ENTITY`; use `WANDB_MODE=offline` when the training host has no stable network. Model artifact upload and gradient histogram watching are disabled by default. Checkpoints and local monitoring files are written under `outputs/sft-qwen2.5-coder-7b-qlora/`; the final adapter is written to its `final_adapter/` subdirectory.

## Status

Phases 0 and 1 are complete. The stage-2 QLoRA-SFT implementation and A800 launch configuration are ready; the actual A800 training run is pending. Preference mining, DPO, and BIRD evaluation are implemented in later phases.

# AlignSQL

AlignSQL is a compact Text-to-SQL post-training project built around one path:

1. QLoRA supervised fine-tuning on BIRD synthetic reasoning data.
2. Execution-guided preference mining from the SFT model.
3. QLoRA Direct Preference Optimization.
4. Greedy execution-accuracy comparison on BIRD Dev.

The local macOS environment is intended for data preparation, tests, and configuration work. CUDA-only packages and the full 7B training runs belong in a separate Linux/A800 environment.

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

## Status

Phases 0 and 1 (repository setup and SFT data preparation) are complete. Training, preference mining, and BIRD evaluation are implemented in later phases.

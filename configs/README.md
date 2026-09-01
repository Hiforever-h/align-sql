# Configurations

`data_sft.yaml` controls validation, deterministic trajectory selection, tokenizer auditing, and the SFT train/validation split.

`sft_qlora.yaml` is the single-A800 QLoRA CoT-SFT configuration. Its default effective batch size is 16 (micro-batch 2, gradient accumulation 8), and its 3,072-token limit explicitly drops rather than truncates overlength examples. Metrics are reported to both W&B and TensorBoard; W&B secrets and account-specific values stay in environment variables.

`eval_base.yaml` and `eval_sft.yaml` define the matched Base/SFT comparison. They use the same base model, validation data, greedy decoding, seed, token limits, and optional SQLite execution settings. The Base config sets `mode: base` with `adapter_path: null`; the SFT config sets `mode: adapter` with the final adapter path. Evaluation artifacts stay under `/root/align-sql/outputs/`.

`dpo_mining.yaml` controls the stage-3 database-aware prompt subset, K-way SFT sampling, inline execution verification, hard-negative pairing, and resumable artifacts. It does not run a separate gold SQL prevalidation pass.

`dpo_qlora.yaml` controls the stage-4 single-A800 DPO refinement. It starts from the trained SFT adapter, keeps a frozen reference copy inside the same PEFT model, uses one conservative epoch at `5e-7`, and rejects rather than truncates pairs over 3,072 tokens. Machine-specific secrets must stay out of these files.

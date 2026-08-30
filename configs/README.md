# Configurations

`data_sft.yaml` controls validation, deterministic trajectory selection, tokenizer auditing, and the SFT train/validation split.

`sft_qlora.yaml` is the single-A800 QLoRA CoT-SFT configuration. Its default effective batch size is 16 (micro-batch 2, gradient accumulation 8), and its 3,072-token limit explicitly drops rather than truncates overlength examples. Metrics are reported to both W&B and TensorBoard; W&B secrets and account-specific values stay in environment variables.

`eval_base.yaml` and `eval_sft.yaml` define the matched Base/SFT comparison. They use the same base model, validation data, greedy decoding, seed, token limits, and optional SQLite execution settings. The Base config sets `mode: base` with `adapter_path: null`; the SFT config sets `mode: adapter` with the final adapter path. Evaluation artifacts stay under `/root/align-sql/outputs/`.

Sampling, DPO, and evaluation configurations will be added in their corresponding implementation phases. Machine-specific paths and secrets must stay out of these files.

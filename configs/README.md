# Configurations

`data_sft.yaml` controls validation, deterministic trajectory selection, tokenizer auditing, and the SFT train/validation split.

`sft_qlora.yaml` is the single-A800 QLoRA CoT-SFT configuration. Its default effective batch size is 16 (micro-batch 2, gradient accumulation 8), and its 3,072-token limit explicitly drops rather than truncates overlength examples. Metrics are reported to both W&B and TensorBoard; W&B secrets and account-specific values stay in environment variables.

Sampling, DPO, and evaluation configurations will be added in their corresponding implementation phases. Machine-specific paths and secrets must stay out of these files.

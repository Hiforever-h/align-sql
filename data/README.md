# Data layout

Raw and generated datasets are excluded from Git. Keep source files in `data/raw/` and derived artifacts in `data/processed/`.

## Current raw files

| File | Purpose | SHA-256 |
| --- | --- | --- |
| `syn_cot_data.json` | Released synthetic reasoning trajectories for SFT | `f5a1a8c3117f28c38c9543b2048443aed7f9d64055ef5e58eda7438e52ea71fd` |
| `train_bird.json` | Preprocessed BIRD training prompts and gold SQL | `d0490dba38380b1d93ea8438f0cfa185d050cc61191e3dc2a3fc881f331fcc11` |
| `dev_bird_0627_10b.json` | Preprocessed BIRD development prompts | `7fea02bc1e416014859b49fab5d1d6ae078c085fed5258325163f4295686d048` |
| `data.zip` | Original downloaded archive | `ea39cb2df97f3190eff840fdb0dd9e8f314712c054845713cd101947e2c85ef3` |

The copies of `syn_cot_data.json` and `train_bird.json` inside `data.zip` have been verified byte-for-byte by SHA-256 against the extracted files.

The BIRD SQLite train/dev databases are not present yet. They are needed only for execution-guided preference mining and final execution-accuracy evaluation.

## Phase 1 output

Strict SQL-to-gold filtering selected 4,918 high-confidence questions across all 69 BIRD training databases:

- Train: 4,811 examples.
- Validation: 107 examples.
- 57,310 of 146,432 source trajectories contained SQL matching their gold query after canonicalization.
- 4,234 questions had released trajectories but no strict gold-matching SQL and were excluded to avoid contradictory SFT and later DPO labels.
- Qwen total length: mean 1,329.5 tokens, p95 2,348, p99 2,729, maximum 3,160.
- A 3,072-token cutoff covers 99.9593% of selected samples; 4,096 is lossless. The two samples above 3,072 must be filtered or trained with the lossless cutoff rather than silently truncated.

Detailed selection, token, split, and artifact checksums are recorded in `reports/sft_data_report.json`.

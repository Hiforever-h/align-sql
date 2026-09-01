# Scripts

- `prepare_sft.sh`: validate the raw BIRD/CoT sources and regenerate the processed SFT splits.
- `download_model.sh`: download a model snapshot into the configured Hugging Face cache.
- `train_sft.sh`: validate or run the stage-2 single-GPU QLoRA CoT-SFT job.
- `eval_base.sh`: run the 4-bit Base baseline without loading a PEFT adapter.
- `eval_sft.sh`: run gold self-check, greedy SFT generation, and optional SQLite execution evaluation.
- `mine_dpo.sh`: generate resumable K-way SFT candidates, verify them on Train SQLite databases, and build hard-negative DPO pairs.
- `train_dpo.sh`: validate or run the stage-4 single-GPU QLoRA-DPO refinement from the final SFT adapter.

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from align_sql.training.dpo.config import DpoMiningConfig


def test_default_dpo_mining_config() -> None:
    config = DpoMiningConfig.from_yaml("configs/dpo_mining.yaml")

    assert config.model.adapter_path == Path(
        "/root/align-sql/outputs/sft-qwen2.5-coder-7b-qlora/final_adapter"
    )
    assert config.data.path == Path("data/processed/sft_train.jsonl")
    assert config.data.sample_size == 2000
    assert config.data.exclude_question_ids == (2809, 7769)
    assert config.sampling.num_candidates == 4
    assert config.sampling.prompt_batch_size == 2
    assert config.execution.db_root == Path(
        "/root/autodl-tmp/bird/train/train_databases"
    )
    assert not config.pairing.include_execution_error_rejected
    assert config.output.candidates_file == Path(
        "/root/align-sql/outputs/dpo-mining-k4/candidates.jsonl"
    )


def test_dpo_config_rejects_invalid_candidate_count_and_relative_output() -> None:
    config = DpoMiningConfig.from_yaml("configs/dpo_mining.yaml")

    with pytest.raises(ValueError, match="at least 2"):
        replace(
            config,
            sampling=replace(config.sampling, num_candidates=1),
        ).validate()
    with pytest.raises(ValueError, match="absolute path"):
        replace(
            config,
            output=replace(config.output, directory=Path("outputs/dpo")),
        ).validate()

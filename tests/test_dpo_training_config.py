from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from align_sql.training.dpo.train_config import DpoRunConfig


def test_default_dpo_training_config() -> None:
    config = DpoRunConfig.from_yaml("configs/dpo_qlora.yaml")

    assert config.model.base_model_name_or_path == "Qwen/Qwen2.5-Coder-7B-Instruct"
    assert config.model.sft_adapter_path == Path(
        "/root/align-sql/outputs/sft-qwen2.5-coder-7b-qlora/final_adapter"
    )
    assert config.data.expected_train_pairs == 523
    assert config.data.expected_validation_pairs == 28
    assert config.data.max_length == 3072
    assert config.dpo.loss_type == "sigmoid"
    assert config.dpo.beta == 0.1
    assert config.training.num_train_epochs == 1.0
    assert config.training.effective_batch_size == 8
    assert config.training.learning_rate == 5e-7
    assert config.training.report_to == ("tensorboard", "wandb")

def test_dpo_training_overrides_do_not_mutate_source() -> None:
    config = DpoRunConfig.from_yaml("configs/dpo_qlora.yaml")
    updated = config.with_overrides(
        base_model_name_or_path="/models/qwen",
        sft_adapter_path="outputs/local-adapter",
        output_dir="/root/align-sql/outputs/dpo-smoke",
        max_steps=5,
        num_train_epochs=2.0,
    )

    assert config.training.max_steps == -1
    assert config.training.num_train_epochs == 1.0
    assert updated.model.base_model_name_or_path == "/models/qwen"
    assert updated.model.sft_adapter_path == Path("outputs/local-adapter")
    assert updated.training.output_dir == Path("/root/align-sql/outputs/dpo-smoke")
    assert updated.training.max_steps == 5
    assert updated.training.num_train_epochs == 2.0


def test_dpo_training_rejects_truncation_and_non_sigmoid_loss() -> None:
    config = DpoRunConfig.from_yaml("configs/dpo_qlora.yaml")

    with pytest.raises(ValueError, match="truncation is forbidden"):
        replace(config, data=replace(config.data, overlength_policy="drop")).validate()
    with pytest.raises(ValueError, match="standard sigmoid"):
        replace(config, dpo=replace(config.dpo, loss_type="hinge")).validate()

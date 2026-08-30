from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from align_sql.training.config import SftRunConfig


def test_stage2_config_has_expected_single_gpu_batch() -> None:
    config = SftRunConfig.from_yaml(Path("configs/sft_qlora.yaml"))

    assert config.model.name_or_path == "Qwen/Qwen2.5-Coder-7B-Instruct"
    assert config.data.max_length == 3072
    assert config.training.effective_batch_size == 16
    assert config.training.completion_only_loss is True
    assert config.training.report_to == ("tensorboard", "wandb")
    assert config.monitoring.wandb_project == "align-sql"
    assert config.monitoring.wandb_mode == "online"
    assert config.lora.target_modules == "all-linear"


def test_stage2_config_overrides_do_not_mutate_source() -> None:
    config = SftRunConfig.from_yaml(Path("configs/sft_qlora.yaml"))
    updated = config.with_overrides(
        model_name_or_path="/models/qwen",
        output_dir="outputs/smoke",
        max_steps=5,
    )

    assert config.training.max_steps == -1
    assert updated.model.name_or_path == "/models/qwen"
    assert updated.training.output_dir == Path("outputs/smoke")
    assert updated.training.max_steps == 5


def test_stage2_rejects_unaligned_max_length() -> None:
    config = SftRunConfig.from_yaml(Path("configs/sft_qlora.yaml"))
    invalid = replace(config, data=replace(config.data, max_length=3084))

    with pytest.raises(ValueError, match="multiple of 8"):
        invalid.validate()

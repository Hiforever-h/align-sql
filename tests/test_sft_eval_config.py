from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from align_sql.evaluation.sft.config import SftEvalConfig


def test_default_sft_eval_config_uses_autodl_output() -> None:
    config = SftEvalConfig.from_yaml("configs/eval_sft.yaml")

    assert config.model.mode == "adapter"
    assert config.model.base_model_name_or_path == "Qwen/Qwen2.5-Coder-7B-Instruct"
    assert config.model.adapter_path == Path(
        "/root/align-sql/outputs/sft-qwen2.5-coder-7b-qlora/final_adapter"
    )
    assert config.output.directory == Path(
        "/root/align-sql/outputs/sft-qwen2.5-coder-7b-qlora/eval/sft_validation"
    )
    assert config.generation.batch_size == 4
    assert config.generation.max_new_tokens == 768


def test_base_and_sft_configs_use_identical_eval_settings() -> None:
    base = SftEvalConfig.from_yaml("configs/eval_base.yaml")
    sft = SftEvalConfig.from_yaml("configs/eval_sft.yaml")

    assert base.model.mode == "base"
    assert base.model.base_model_name_or_path == sft.model.base_model_name_or_path
    assert base.model.adapter_path is None
    assert base.data == sft.data
    assert base.generation == sft.generation
    assert base.execution == sft.execution
    assert base.output.directory == Path(
        "/root/align-sql/outputs/base-qwen2.5-coder-7b/eval/sft_validation"
    )


def test_model_modes_reject_inconsistent_adapter_paths() -> None:
    base = SftEvalConfig.from_yaml("configs/eval_base.yaml")
    sft = SftEvalConfig.from_yaml("configs/eval_sft.yaml")

    with pytest.raises(ValueError, match="base mode requires adapter_path: null"):
        replace(base, model=replace(base.model, adapter_path=Path("adapter"))).validate()
    with pytest.raises(ValueError, match="adapter mode requires adapter_path"):
        replace(sft, model=replace(sft.model, adapter_path=None)).validate()
    with pytest.raises(ValueError, match="adapter_path override must not be empty"):
        sft.with_overrides(adapter_path="")


def test_sft_eval_rejects_relative_output_path() -> None:
    config = SftEvalConfig.from_yaml("configs/eval_sft.yaml")
    invalid = replace(config, output=replace(config.output, directory=Path("outputs/eval")))

    with pytest.raises(ValueError, match="absolute path"):
        invalid.validate()

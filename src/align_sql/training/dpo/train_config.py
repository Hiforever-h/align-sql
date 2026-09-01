"""Configuration schema for QLoRA-DPO refinement."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DpoModelConfig:
    base_model_name_or_path: str
    sft_adapter_path: Path
    trust_remote_code: bool
    attn_implementation: str


@dataclass(frozen=True)
class DpoQuantizationConfig:
    load_in_4bit: bool
    quant_type: str
    use_double_quant: bool
    compute_dtype: str


@dataclass(frozen=True)
class DpoDataConfig:
    train_file: Path
    validation_file: Path
    mining_manifest_file: Path
    max_length: int
    overlength_policy: str
    expected_train_pairs: int | None
    expected_validation_pairs: int | None


@dataclass(frozen=True)
class DpoObjectiveConfig:
    loss_type: str
    beta: float
    precompute_ref_log_probs: bool


@dataclass(frozen=True)
class DpoTrainingConfig:
    output_dir: Path
    run_name: str
    num_train_epochs: float
    max_steps: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    warmup_steps: int
    lr_scheduler_type: str
    weight_decay: float
    max_grad_norm: float
    optim: str
    bf16: bool
    tf32: bool
    gradient_checkpointing: bool
    gradient_checkpointing_use_reentrant: bool
    pad_to_multiple_of: int | None
    logging_steps: int
    eval_steps: int
    save_steps: int
    save_total_limit: int
    dataset_num_proc: int | None
    dataloader_num_workers: int
    seed: int
    report_to: tuple[str, ...]

    @property
    def effective_batch_size(self) -> int:
        return self.per_device_train_batch_size * self.gradient_accumulation_steps


@dataclass(frozen=True)
class DpoMonitoringConfig:
    wandb_project: str
    wandb_mode: str
    wandb_log_model: bool
    wandb_watch: bool


@dataclass(frozen=True)
class DpoRunConfig:
    model: DpoModelConfig
    quantization: DpoQuantizationConfig
    data: DpoDataConfig
    dpo: DpoObjectiveConfig
    training: DpoTrainingConfig
    monitoring: DpoMonitoringConfig
    source_path: Path

    @classmethod
    def from_yaml(cls, path: str | Path) -> DpoRunConfig:
        config_path = Path(path)
        with config_path.open(encoding="utf-8") as handle:
            raw: dict[str, Any] = yaml.safe_load(handle)

        model = raw["model"]
        quantization = raw["quantization"]
        data = raw["data"]
        dpo = raw["dpo"]
        training = raw["training"]
        monitoring = raw["monitoring"]
        config = cls(
            model=DpoModelConfig(
                base_model_name_or_path=str(model["base_model_name_or_path"]),
                sft_adapter_path=Path(model["sft_adapter_path"]),
                trust_remote_code=bool(model["trust_remote_code"]),
                attn_implementation=str(model["attn_implementation"]),
            ),
            quantization=DpoQuantizationConfig(
                load_in_4bit=bool(quantization["load_in_4bit"]),
                quant_type=str(quantization["bnb_4bit_quant_type"]),
                use_double_quant=bool(quantization["bnb_4bit_use_double_quant"]),
                compute_dtype=str(quantization["bnb_4bit_compute_dtype"]),
            ),
            data=DpoDataConfig(
                train_file=Path(data["train_file"]),
                validation_file=Path(data["validation_file"]),
                mining_manifest_file=Path(data["mining_manifest_file"]),
                max_length=int(data["max_length"]),
                overlength_policy=str(data["overlength_policy"]),
                expected_train_pairs=_optional_int(data.get("expected_train_pairs")),
                expected_validation_pairs=_optional_int(
                    data.get("expected_validation_pairs")
                ),
            ),
            dpo=DpoObjectiveConfig(
                loss_type=str(dpo["loss_type"]),
                beta=float(dpo["beta"]),
                precompute_ref_log_probs=bool(dpo["precompute_ref_log_probs"]),
            ),
            training=DpoTrainingConfig(
                output_dir=Path(training["output_dir"]),
                run_name=str(training["run_name"]),
                num_train_epochs=float(training["num_train_epochs"]),
                max_steps=int(training["max_steps"]),
                per_device_train_batch_size=int(training["per_device_train_batch_size"]),
                per_device_eval_batch_size=int(training["per_device_eval_batch_size"]),
                gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
                learning_rate=float(training["learning_rate"]),
                warmup_steps=int(training["warmup_steps"]),
                lr_scheduler_type=str(training["lr_scheduler_type"]),
                weight_decay=float(training["weight_decay"]),
                max_grad_norm=float(training["max_grad_norm"]),
                optim=str(training["optim"]),
                bf16=bool(training["bf16"]),
                tf32=bool(training["tf32"]),
                gradient_checkpointing=bool(training["gradient_checkpointing"]),
                gradient_checkpointing_use_reentrant=bool(
                    training["gradient_checkpointing_use_reentrant"]
                ),
                pad_to_multiple_of=_optional_int(training.get("pad_to_multiple_of")),
                logging_steps=int(training["logging_steps"]),
                eval_steps=int(training["eval_steps"]),
                save_steps=int(training["save_steps"]),
                save_total_limit=int(training["save_total_limit"]),
                dataset_num_proc=_optional_int(training.get("dataset_num_proc")),
                dataloader_num_workers=int(training["dataloader_num_workers"]),
                seed=int(training["seed"]),
                report_to=tuple(str(value) for value in training.get("report_to", [])),
            ),
            monitoring=DpoMonitoringConfig(
                wandb_project=str(monitoring["wandb_project"]),
                wandb_mode=str(monitoring["wandb_mode"]),
                wandb_log_model=bool(monitoring["wandb_log_model"]),
                wandb_watch=bool(monitoring["wandb_watch"]),
            ),
            source_path=config_path,
        )
        config.validate()
        return config

    def with_overrides(
        self,
        *,
        base_model_name_or_path: str | None = None,
        sft_adapter_path: str | Path | None = None,
        output_dir: str | Path | None = None,
        max_steps: int | None = None,
        num_train_epochs: float | None = None,
    ) -> DpoRunConfig:
        model = self.model
        training = self.training
        if base_model_name_or_path is not None:
            model = replace(model, base_model_name_or_path=base_model_name_or_path)
        if sft_adapter_path is not None:
            model = replace(model, sft_adapter_path=Path(sft_adapter_path))
        if output_dir is not None:
            training = replace(training, output_dir=Path(output_dir))
        if max_steps is not None:
            training = replace(training, max_steps=max_steps)
        if num_train_epochs is not None:
            training = replace(training, num_train_epochs=num_train_epochs)
        updated = replace(self, model=model, training=training)
        updated.validate()
        return updated

    def validate(self) -> None:
        if not self.model.base_model_name_or_path.strip():
            raise ValueError("model.base_model_name_or_path must not be empty")
        if self.model.attn_implementation not in {"sdpa", "flash_attention_2", "eager"}:
            raise ValueError("Unsupported model.attn_implementation")
        if not self.quantization.load_in_4bit:
            raise ValueError("Stage 4 is intentionally configured as 4-bit QLoRA-DPO")
        if self.quantization.quant_type != "nf4":
            raise ValueError("QLoRA quantization type must be nf4")
        if self.quantization.compute_dtype != "bfloat16":
            raise ValueError("A800 QLoRA compute dtype must be bfloat16")
        if self.data.max_length <= 0 or self.data.max_length % 8 != 0:
            raise ValueError("data.max_length must be a positive multiple of 8")
        if self.data.overlength_policy != "error":
            raise ValueError("DPO overlength_policy must be error; silent truncation is forbidden")
        if self.dpo.loss_type != "sigmoid":
            raise ValueError("The first DPO run must use the standard sigmoid loss")
        if self.dpo.beta <= 0:
            raise ValueError("dpo.beta must be positive")
        if self.training.num_train_epochs <= 0:
            raise ValueError("num_train_epochs must be positive")
        if self.training.max_steps == 0 or self.training.max_steps < -1:
            raise ValueError("max_steps must be -1 or a positive integer")
        positive_integer_fields = {
            "per_device_train_batch_size": self.training.per_device_train_batch_size,
            "per_device_eval_batch_size": self.training.per_device_eval_batch_size,
            "gradient_accumulation_steps": self.training.gradient_accumulation_steps,
            "logging_steps": self.training.logging_steps,
            "eval_steps": self.training.eval_steps,
            "save_steps": self.training.save_steps,
            "save_total_limit": self.training.save_total_limit,
        }
        for name, value in positive_integer_fields.items():
            if value <= 0:
                raise ValueError(f"training.{name} must be positive")
        if self.training.learning_rate <= 0:
            raise ValueError("training.learning_rate must be positive")
        if self.training.warmup_steps < 0:
            raise ValueError("training.warmup_steps must not be negative")
        if not self.training.bf16:
            raise ValueError("A800 DPO training must use bf16")
        if (
            self.training.pad_to_multiple_of is not None
            and self.training.pad_to_multiple_of <= 0
        ):
            raise ValueError("pad_to_multiple_of must be positive when set")
        if not self.training.output_dir.is_absolute():
            raise ValueError("training.output_dir must be absolute")
        if not self.monitoring.wandb_project:
            raise ValueError("monitoring.wandb_project must not be empty")
        if self.monitoring.wandb_mode not in {"online", "offline", "disabled"}:
            raise ValueError("wandb_mode must be online, offline, or disabled")
        if self.monitoring.wandb_mode != "disabled" and "wandb" not in self.training.report_to:
            raise ValueError("training.report_to must include wandb when W&B is enabled")

    def validate_input_files(self) -> None:
        files = (
            self.data.train_file,
            self.data.validation_file,
            self.data.mining_manifest_file,
            self.model.sft_adapter_path / "adapter_config.json",
        )
        missing = [path for path in files if not path.is_file()]
        if missing:
            paths = ", ".join(str(path) for path in missing)
            raise FileNotFoundError(f"Missing DPO input artifacts: {paths}")


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)

"""Configuration schema for QLoRA supervised fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    name_or_path: str
    trust_remote_code: bool
    attn_implementation: str


@dataclass(frozen=True)
class QuantizationConfig:
    load_in_4bit: bool
    quant_type: str
    use_double_quant: bool
    compute_dtype: str


@dataclass(frozen=True)
class LoraConfig:
    r: int
    alpha: int
    dropout: float
    bias: str
    target_modules: str | tuple[str, ...]
    use_rslora: bool


@dataclass(frozen=True)
class DataConfig:
    train_file: Path
    validation_file: Path
    max_length: int
    overlength_policy: str
    expected_dropped_train: int | None
    expected_dropped_validation: int | None


@dataclass(frozen=True)
class TrainingConfig:
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
    packing: bool
    completion_only_loss: bool
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
class MonitoringConfig:
    wandb_project: str
    wandb_mode: str
    wandb_log_model: bool
    wandb_watch: bool


@dataclass(frozen=True)
class SftRunConfig:
    model: ModelConfig
    quantization: QuantizationConfig
    lora: LoraConfig
    data: DataConfig
    training: TrainingConfig
    monitoring: MonitoringConfig
    source_path: Path

    @classmethod
    def from_yaml(cls, path: str | Path) -> SftRunConfig:
        config_path = Path(path)
        with config_path.open(encoding="utf-8") as handle:
            raw: dict[str, Any] = yaml.safe_load(handle)

        model = raw["model"]
        quantization = raw["quantization"]
        lora = raw["lora"]
        data = raw["data"]
        training = raw["training"]
        monitoring = raw["monitoring"]
        target_modules_raw = lora["target_modules"]
        target_modules = (
            str(target_modules_raw)
            if isinstance(target_modules_raw, str)
            else tuple(str(value) for value in target_modules_raw)
        )

        config = cls(
            model=ModelConfig(
                name_or_path=str(model["name_or_path"]),
                trust_remote_code=bool(model["trust_remote_code"]),
                attn_implementation=str(model["attn_implementation"]),
            ),
            quantization=QuantizationConfig(
                load_in_4bit=bool(quantization["load_in_4bit"]),
                quant_type=str(quantization["bnb_4bit_quant_type"]),
                use_double_quant=bool(quantization["bnb_4bit_use_double_quant"]),
                compute_dtype=str(quantization["bnb_4bit_compute_dtype"]),
            ),
            lora=LoraConfig(
                r=int(lora["r"]),
                alpha=int(lora["alpha"]),
                dropout=float(lora["dropout"]),
                bias=str(lora["bias"]),
                target_modules=target_modules,
                use_rslora=bool(lora["use_rslora"]),
            ),
            data=DataConfig(
                train_file=Path(data["train_file"]),
                validation_file=Path(data["validation_file"]),
                max_length=int(data["max_length"]),
                overlength_policy=str(data["overlength_policy"]),
                expected_dropped_train=_optional_int(data.get("expected_dropped_train")),
                expected_dropped_validation=_optional_int(
                    data.get("expected_dropped_validation")
                ),
            ),
            training=TrainingConfig(
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
                packing=bool(training["packing"]),
                completion_only_loss=bool(training["completion_only_loss"]),
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
            monitoring=MonitoringConfig(
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
        model_name_or_path: str | None = None,
        output_dir: str | Path | None = None,
        max_steps: int | None = None,
    ) -> SftRunConfig:
        model = self.model
        training = self.training
        if model_name_or_path is not None:
            model = replace(model, name_or_path=model_name_or_path)
        if output_dir is not None:
            training = replace(training, output_dir=Path(output_dir))
        if max_steps is not None:
            training = replace(training, max_steps=max_steps)
        updated = replace(self, model=model, training=training)
        updated.validate()
        return updated

    def validate(self) -> None:
        if not self.model.name_or_path:
            raise ValueError("model.name_or_path must not be empty")
        if self.model.attn_implementation not in {"sdpa", "flash_attention_2", "eager"}:
            raise ValueError("Unsupported model.attn_implementation")
        if not self.quantization.load_in_4bit:
            raise ValueError("Stage 2 is intentionally configured as 4-bit QLoRA")
        if self.quantization.quant_type != "nf4":
            raise ValueError("QLoRA quantization type must be nf4")
        if self.quantization.compute_dtype != "bfloat16":
            raise ValueError("A800 QLoRA compute dtype must be bfloat16")
        if self.lora.r <= 0 or self.lora.alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        if not 0.0 <= self.lora.dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0, 1)")
        if self.lora.bias not in {"none", "all", "lora_only"}:
            raise ValueError("Unsupported LoRA bias mode")
        if self.data.max_length <= 0 or self.data.max_length % 8 != 0:
            raise ValueError("data.max_length must be a positive multiple of 8")
        if self.data.overlength_policy != "drop":
            raise ValueError("Only explicit dropping of overlength samples is supported")
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
            raise ValueError("learning_rate must be positive")
        if not self.training.bf16:
            raise ValueError("A800 training must use bf16")
        if self.training.packing:
            raise ValueError("Packing is disabled for the first reproducible SFT run")
        if not self.training.completion_only_loss:
            raise ValueError("SFT must compute loss only on the assistant completion")
        if (
            self.training.pad_to_multiple_of is not None
            and self.training.pad_to_multiple_of <= 0
        ):
            raise ValueError("pad_to_multiple_of must be positive when set")
        if not self.monitoring.wandb_project:
            raise ValueError("monitoring.wandb_project must not be empty")
        if self.monitoring.wandb_mode not in {"online", "offline", "disabled"}:
            raise ValueError("wandb_mode must be online, offline, or disabled")
        if self.monitoring.wandb_mode != "disabled" and "wandb" not in self.training.report_to:
            raise ValueError("training.report_to must include wandb when W&B is enabled")

    def validate_data_files(self) -> None:
        data_files = (self.data.train_file, self.data.validation_file)
        missing = [path for path in data_files if not path.is_file()]
        if missing:
            paths = ", ".join(str(path) for path in missing)
            raise FileNotFoundError(f"Missing processed SFT data: {paths}")


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)

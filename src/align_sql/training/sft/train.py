"""Single-A800 QLoRA CoT-SFT entry point."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import warnings
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig as PeftLoraConfig
from peft import TaskType, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from trl import SFTConfig, SFTTrainer

from align_sql.training.sft.config import SftRunConfig
from align_sql.training.sft.data import SplitAudit, prepare_split


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run QLoRA CoT-SFT for AlignSQL")
    parser.add_argument("--config", default="configs/sft_qlora.yaml")
    parser.add_argument("--model-name-or-path")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate config, tokenizer, and datasets without loading model weights or CUDA",
    )
    return parser.parse_args()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _hardware_info(require_cuda: bool) -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
    }
    if not torch.cuda.is_available():
        if require_cuda:
            raise RuntimeError(
                "QLoRA training requires CUDA. Use --validate-only on macOS/CPU machines."
            )
        return info

    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    info.update(
        {
            "cuda_device": device,
            "gpu_name": properties.name,
            "gpu_compute_capability": f"{properties.major}.{properties.minor}",
            "gpu_memory_gib": round(properties.total_memory / 1024**3, 2),
            "bf16_supported": torch.cuda.is_bf16_supported(),
        }
    )
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected CUDA device does not support bf16 training")
    if properties.total_memory < 70 * 1024**3:
        warnings.warn(
            "The default batch size was selected for an A800 80GB; this GPU has less than 70 GiB.",
            stacklevel=2,
        )
    return info


def _configure_monitoring(config: SftRunConfig) -> None:
    monitoring = config.monitoring
    os.environ.setdefault("WANDB_PROJECT", monitoring.wandb_project)
    os.environ.setdefault("WANDB_MODE", monitoring.wandb_mode)
    os.environ.setdefault("WANDB_LOG_MODEL", str(monitoring.wandb_log_model).lower())
    os.environ.setdefault("WANDB_WATCH", str(monitoring.wandb_watch).lower())
    os.environ.setdefault("WANDB_DIR", str(config.training.output_dir / "wandb"))


def _load_tokenizer(config: SftRunConfig):
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.name_or_path,
        use_fast=True,
        trust_remote_code=config.model.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither pad_token nor eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def _prepare_datasets(config: SftRunConfig, tokenizer):
    train_dataset, train_audit = prepare_split(
        config.data.train_file,
        split="train",
        tokenizer=tokenizer,
        max_length=config.data.max_length,
        expected_dropped=config.data.expected_dropped_train,
    )
    validation_dataset, validation_audit = prepare_split(
        config.data.validation_file,
        split="validation",
        tokenizer=tokenizer,
        max_length=config.data.max_length,
        expected_dropped=config.data.expected_dropped_validation,
    )
    return train_dataset, validation_dataset, train_audit, validation_audit


def _quantization_config(config: SftRunConfig) -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=config.quantization.load_in_4bit,
        bnb_4bit_quant_type=config.quantization.quant_type,
        bnb_4bit_use_double_quant=config.quantization.use_double_quant,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def _load_qlora_model(config: SftRunConfig):
    try:
        import bitsandbytes  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            "bitsandbytes is required on the Linux/A800 training machine. "
            "Install requirements-a800.txt first."
        ) from error

    device = torch.cuda.current_device()
    model = AutoModelForCausalLM.from_pretrained(
        config.model.name_or_path,
        quantization_config=_quantization_config(config),
        dtype=torch.bfloat16,
        device_map={"": device},
        attn_implementation=config.model.attn_implementation,
        trust_remote_code=config.model.trust_remote_code,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    return prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=config.training.gradient_checkpointing,
        gradient_checkpointing_kwargs={
            "use_reentrant": config.training.gradient_checkpointing_use_reentrant
        },
    )


def _peft_config(config: SftRunConfig) -> PeftLoraConfig:
    return PeftLoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        bias=config.lora.bias,
        target_modules=config.lora.target_modules,
        use_rslora=config.lora.use_rslora,
    )


def _training_arguments(config: SftRunConfig) -> SFTConfig:
    training = config.training
    return SFTConfig(
        output_dir=str(training.output_dir),
        run_name=training.run_name,
        do_train=True,
        do_eval=True,
        num_train_epochs=training.num_train_epochs,
        max_steps=training.max_steps,
        per_device_train_batch_size=training.per_device_train_batch_size,
        per_device_eval_batch_size=training.per_device_eval_batch_size,
        gradient_accumulation_steps=training.gradient_accumulation_steps,
        learning_rate=training.learning_rate,
        warmup_steps=training.warmup_steps,
        lr_scheduler_type=training.lr_scheduler_type,
        weight_decay=training.weight_decay,
        max_grad_norm=training.max_grad_norm,
        optim=training.optim,
        bf16=training.bf16,
        tf32=training.tf32,
        gradient_checkpointing=training.gradient_checkpointing,
        gradient_checkpointing_kwargs={
            "use_reentrant": training.gradient_checkpointing_use_reentrant
        },
        use_cache=False,
        max_length=config.data.max_length,
        truncation_mode="keep_start",
        packing=training.packing,
        padding_free=False,
        pad_to_multiple_of=training.pad_to_multiple_of,
        completion_only_loss=training.completion_only_loss,
        assistant_only_loss=False,
        logging_strategy="steps",
        logging_steps=training.logging_steps,
        logging_first_step=True,
        eval_strategy="steps",
        eval_steps=training.eval_steps,
        prediction_loss_only=True,
        save_strategy="steps",
        save_steps=training.save_steps,
        save_total_limit=training.save_total_limit,
        dataset_num_proc=training.dataset_num_proc,
        dataloader_num_workers=training.dataloader_num_workers,
        dataloader_pin_memory=True,
        seed=training.seed,
        data_seed=training.seed,
        report_to=list(training.report_to) or "none",
        include_num_input_tokens_seen=True,
        remove_unused_columns=True,
    )


def _trainable_parameters(model) -> dict[str, int | float]:
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    return {
        "trainable": trainable,
        "total": total,
        "trainable_percent": round(100 * trainable / total, 6),
    }


def _manifest(
    config: SftRunConfig,
    train_audit: SplitAudit,
    validation_audit: SplitAudit,
    hardware: dict[str, Any],
) -> dict[str, Any]:
    updates_per_epoch = math.ceil(
        train_audit.kept_count / config.training.effective_batch_size
    )
    planned_optimizer_steps = (
        config.training.max_steps
        if config.training.max_steps > 0
        else math.ceil(updates_per_epoch * config.training.num_train_epochs)
    )
    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config_path": str(config.source_path),
        "model": asdict(config.model),
        "quantization": asdict(config.quantization),
        "lora": asdict(config.lora),
        "data": {
            **asdict(config.data),
            "train_file": str(config.data.train_file),
            "validation_file": str(config.data.validation_file),
        },
        "training": {
            **asdict(config.training),
            "output_dir": str(config.training.output_dir),
            "effective_batch_size": config.training.effective_batch_size,
            "updates_per_epoch": updates_per_epoch,
            "planned_optimizer_steps": planned_optimizer_steps,
        },
        "monitoring": {
            **asdict(config.monitoring),
            "effective_wandb_project": os.environ.get("WANDB_PROJECT"),
            "effective_wandb_mode": os.environ.get("WANDB_MODE"),
            "effective_wandb_dir": os.environ.get("WANDB_DIR"),
            "wandb_api_key_present": bool(os.environ.get("WANDB_API_KEY")),
        },
        "dataset_audit": {
            "train": train_audit.to_dict(),
            "validation": validation_audit.to_dict(),
        },
        "hardware": hardware,
        "environment": {
            name: _package_version(name)
            for name in (
                "torch",
                "transformers",
                "datasets",
                "accelerate",
                "peft",
                "trl",
                "bitsandbytes",
                "wandb",
            )
        },
        "hf_cache": {
            "HF_HOME": os.environ.get("HF_HOME"),
            "HF_HUB_CACHE": os.environ.get("HF_HUB_CACHE"),
        },
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def _ensure_fresh_output_dir(output_dir: Path, resume_from_checkpoint: str | None) -> None:
    if resume_from_checkpoint is not None:
        if not Path(resume_from_checkpoint).is_dir():
            raise FileNotFoundError(
                f"Resume checkpoint does not exist: {resume_from_checkpoint}"
            )
        return
    if output_dir.is_dir() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. "
            "Use a new --output-dir or pass --resume-from-checkpoint."
        )


def main() -> None:
    args = _parse_args()
    config = SftRunConfig.from_yaml(args.config).with_overrides(
        model_name_or_path=args.model_name_or_path,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
    )
    config.validate_data_files()
    set_seed(config.training.seed)
    _configure_monitoring(config)
    tokenizer = _load_tokenizer(config)
    train_dataset, validation_dataset, train_audit, validation_audit = _prepare_datasets(
        config, tokenizer
    )
    hardware = _hardware_info(require_cuda=not args.validate_only)
    manifest = _manifest(config, train_audit, validation_audit, hardware)

    if args.validate_only:
        print(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
        return

    _ensure_fresh_output_dir(config.training.output_dir, args.resume_from_checkpoint)
    model = _load_qlora_model(config)
    trainer = SFTTrainer(
        model=model,
        args=_training_arguments(config),
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        peft_config=_peft_config(config),
    )
    manifest["parameters"] = _trainable_parameters(trainer.model)
    _write_json(config.training.output_dir / "run_manifest.json", manifest)
    trainer.model.print_trainable_parameters()

    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    train_metrics = dict(train_result.metrics)
    train_metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)
    trainer.save_state()

    eval_metrics = dict(trainer.evaluate())
    eval_metrics["eval_samples"] = len(validation_dataset)
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    final_adapter_dir = config.training.output_dir / "final_adapter"
    trainer.model.config.use_cache = True
    trainer.save_model(str(final_adapter_dir))
    tokenizer.save_pretrained(final_adapter_dir)
    manifest["completed_at_utc"] = datetime.now(UTC).isoformat()
    manifest["final_adapter"] = str(final_adapter_dir)
    manifest["train_metrics"] = train_metrics
    manifest["eval_metrics"] = eval_metrics
    _write_json(config.training.output_dir / "run_manifest.json", manifest)


if __name__ == "__main__":
    main()

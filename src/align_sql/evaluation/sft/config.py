"""Configuration schema for base-model and SFT generation evaluation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class EvalModelConfig:
    mode: str
    base_model_name_or_path: str
    adapter_path: Path | None
    attn_implementation: str
    trust_remote_code: bool
    load_in_4bit: bool


@dataclass(frozen=True)
class EvalDataConfig:
    path: Path
    max_input_length: int
    limit: int | None


@dataclass(frozen=True)
class GenerationConfig:
    batch_size: int
    max_new_tokens: int
    seed: int


@dataclass(frozen=True)
class ExecutionConfig:
    db_root: Path | None
    timeout_seconds: float
    max_result_rows: int


@dataclass(frozen=True)
class EvalOutputConfig:
    directory: Path


@dataclass(frozen=True)
class SftEvalConfig:
    model: EvalModelConfig
    data: EvalDataConfig
    generation: GenerationConfig
    execution: ExecutionConfig
    output: EvalOutputConfig
    source_path: Path

    @classmethod
    def from_yaml(cls, path: str | Path) -> SftEvalConfig:
        config_path = Path(path)
        with config_path.open(encoding="utf-8") as handle:
            raw: dict[str, Any] = yaml.safe_load(handle)

        model = raw["model"]
        data = raw["data"]
        generation = raw["generation"]
        execution = raw["execution"]
        output = raw["output"]
        config = cls(
            model=EvalModelConfig(
                mode=str(model["mode"]),
                base_model_name_or_path=str(model["base_model_name_or_path"]),
                adapter_path=_optional_path(model.get("adapter_path")),
                attn_implementation=str(model["attn_implementation"]),
                trust_remote_code=bool(model["trust_remote_code"]),
                load_in_4bit=bool(model["load_in_4bit"]),
            ),
            data=EvalDataConfig(
                path=Path(data["path"]),
                max_input_length=int(data["max_input_length"]),
                limit=_optional_int(data.get("limit")),
            ),
            generation=GenerationConfig(
                batch_size=int(generation["batch_size"]),
                max_new_tokens=int(generation["max_new_tokens"]),
                seed=int(generation["seed"]),
            ),
            execution=ExecutionConfig(
                db_root=_optional_path(execution.get("db_root")),
                timeout_seconds=float(execution["timeout_seconds"]),
                max_result_rows=int(execution["max_result_rows"]),
            ),
            output=EvalOutputConfig(directory=Path(output["dir"])),
            source_path=config_path,
        )
        config.validate()
        return config

    def with_overrides(
        self,
        *,
        model_mode: str | None = None,
        base_model_name_or_path: str | None = None,
        adapter_path: str | Path | None = None,
        data_path: str | Path | None = None,
        output_dir: str | Path | None = None,
        db_root: str | Path | None = None,
        limit: int | None = None,
        batch_size: int | None = None,
        max_new_tokens: int | None = None,
    ) -> SftEvalConfig:
        model = self.model
        data = self.data
        generation = self.generation
        execution = self.execution
        output = self.output
        if model_mode is not None:
            model = replace(model, mode=model_mode)
        if base_model_name_or_path is not None:
            model = replace(model, base_model_name_or_path=base_model_name_or_path)
        if adapter_path is not None:
            if not str(adapter_path).strip():
                raise ValueError("adapter_path override must not be empty")
            model = replace(model, adapter_path=Path(adapter_path))
        if data_path is not None:
            data = replace(data, path=Path(data_path))
        if output_dir is not None:
            output = replace(output, directory=Path(output_dir))
        if db_root is not None:
            execution = replace(execution, db_root=Path(db_root))
        if limit is not None:
            data = replace(data, limit=limit)
        if batch_size is not None:
            generation = replace(generation, batch_size=batch_size)
        if max_new_tokens is not None:
            generation = replace(generation, max_new_tokens=max_new_tokens)
        updated = replace(
            self,
            model=model,
            data=data,
            generation=generation,
            execution=execution,
            output=output,
        )
        updated.validate()
        return updated

    def validate(self) -> None:
        if self.model.mode not in {"base", "adapter"}:
            raise ValueError("model.mode must be either 'base' or 'adapter'")
        if not self.model.base_model_name_or_path.strip():
            raise ValueError("base_model_name_or_path must not be empty")
        if self.model.mode == "adapter" and self.model.adapter_path is None:
            raise ValueError("adapter mode requires adapter_path")
        if self.model.mode == "base" and self.model.adapter_path is not None:
            raise ValueError("base mode requires adapter_path: null")
        if self.model.attn_implementation not in {"sdpa", "flash_attention_2", "eager"}:
            raise ValueError("Unsupported attention implementation")
        if not self.model.load_in_4bit:
            raise ValueError("Model evaluation is configured for 4-bit loading")
        if self.data.max_input_length <= 0:
            raise ValueError("max_input_length must be positive")
        if self.data.limit is not None and self.data.limit <= 0:
            raise ValueError("limit must be positive when set")
        if self.generation.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.generation.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.execution.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.execution.max_result_rows <= 0:
            raise ValueError("max_result_rows must be positive")
        if not self.output.directory.is_absolute():
            raise ValueError("Evaluation output directory must be an absolute path")

    def validate_inputs(self, *, require_model: bool) -> None:
        if not self.data.path.is_file():
            raise FileNotFoundError(f"Evaluation data does not exist: {self.data.path}")
        if (
            require_model
            and self.model.mode == "adapter"
            and self.model.adapter_path is not None
            and not self.model.adapter_path.is_dir()
        ):
            raise FileNotFoundError(f"SFT adapter does not exist: {self.model.adapter_path}")
        if self.execution.db_root is not None and not self.execution.db_root.is_dir():
            raise FileNotFoundError(f"Database root does not exist: {self.execution.db_root}")


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_path(value: Any) -> Path | None:
    return None if value in {None, ""} else Path(value)

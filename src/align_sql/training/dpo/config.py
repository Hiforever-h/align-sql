"""Configuration schema for execution-guided DPO preference mining."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class MiningModelConfig:
    adapter_path: Path
    trust_remote_code: bool
    attn_implementation: str
    load_in_4bit: bool


@dataclass(frozen=True)
class MiningDataConfig:
    path: Path
    sample_size: int | None
    exclude_question_ids: tuple[int, ...]
    seed: int
    max_input_length: int


@dataclass(frozen=True)
class SamplingConfig:
    num_candidates: int
    prompt_batch_size: int
    max_new_tokens: int
    temperature: float
    top_p: float
    seed: int


@dataclass(frozen=True)
class MiningExecutionConfig:
    db_root: Path
    timeout_seconds: float
    max_result_rows: int


@dataclass(frozen=True)
class PairingConfig:
    include_execution_error_rejected: bool
    validation_ratio: float
    seed: int


@dataclass(frozen=True)
class MiningOutputConfig:
    directory: Path

    @property
    def candidates_file(self) -> Path:
        return self.directory / "candidates.jsonl"

    @property
    def verified_candidates_file(self) -> Path:
        return self.directory / "verified_candidates.jsonl"

    @property
    def train_file(self) -> Path:
        return self.directory / "dpo_train.jsonl"

    @property
    def validation_file(self) -> Path:
        return self.directory / "dpo_validation.jsonl"

    @property
    def report_file(self) -> Path:
        return self.directory / "mining_report.json"

    @property
    def manifest_file(self) -> Path:
        return self.directory / "run_manifest.json"


@dataclass(frozen=True)
class DpoMiningConfig:
    model: MiningModelConfig
    data: MiningDataConfig
    sampling: SamplingConfig
    execution: MiningExecutionConfig
    pairing: PairingConfig
    output: MiningOutputConfig
    source_path: Path

    @classmethod
    def from_yaml(cls, path: str | Path) -> DpoMiningConfig:
        config_path = Path(path)
        with config_path.open(encoding="utf-8") as handle:
            raw: dict[str, Any] = yaml.safe_load(handle)

        model = raw["model"]
        data = raw["data"]
        sampling = raw["sampling"]
        execution = raw["execution"]
        pairing = raw["pairing"]
        output = raw["output"]
        config = cls(
            model=MiningModelConfig(
                adapter_path=Path(model["adapter_path"]),
                trust_remote_code=bool(model["trust_remote_code"]),
                attn_implementation=str(model["attn_implementation"]),
                load_in_4bit=bool(model["load_in_4bit"]),
            ),
            data=MiningDataConfig(
                path=Path(data["path"]),
                sample_size=_optional_int(data.get("sample_size")),
                exclude_question_ids=tuple(
                    int(value) for value in data.get("exclude_question_ids", [])
                ),
                seed=int(data["seed"]),
                max_input_length=int(data["max_input_length"]),
            ),
            sampling=SamplingConfig(
                num_candidates=int(sampling["num_candidates"]),
                prompt_batch_size=int(sampling["prompt_batch_size"]),
                max_new_tokens=int(sampling["max_new_tokens"]),
                temperature=float(sampling["temperature"]),
                top_p=float(sampling["top_p"]),
                seed=int(sampling["seed"]),
            ),
            execution=MiningExecutionConfig(
                db_root=Path(execution["db_root"]),
                timeout_seconds=float(execution["timeout_seconds"]),
                max_result_rows=int(execution["max_result_rows"]),
            ),
            pairing=PairingConfig(
                include_execution_error_rejected=bool(
                    pairing["include_execution_error_rejected"]
                ),
                validation_ratio=float(pairing["validation_ratio"]),
                seed=int(pairing["seed"]),
            ),
            output=MiningOutputConfig(directory=Path(output["dir"])),
            source_path=config_path,
        )
        config.validate()
        return config

    def with_overrides(
        self,
        *,
        adapter_path: str | Path | None = None,
        db_root: str | Path | None = None,
        output_dir: str | Path | None = None,
        sample_size: int | None = None,
    ) -> DpoMiningConfig:
        model = self.model
        data = self.data
        execution = self.execution
        output = self.output
        if adapter_path is not None:
            model = replace(model, adapter_path=Path(adapter_path))
        if db_root is not None:
            execution = replace(execution, db_root=Path(db_root))
        if output_dir is not None:
            output = replace(output, directory=Path(output_dir))
        if sample_size is not None:
            data = replace(data, sample_size=sample_size)
        updated = replace(
            self,
            model=model,
            data=data,
            execution=execution,
            output=output,
        )
        updated.validate()
        return updated

    def validate(self) -> None:
        if self.model.attn_implementation not in {"sdpa", "flash_attention_2", "eager"}:
            raise ValueError("Unsupported model.attn_implementation")
        if not self.model.load_in_4bit:
            raise ValueError("DPO candidate generation is configured for 4-bit loading")
        if self.data.sample_size is not None and self.data.sample_size <= 0:
            raise ValueError("data.sample_size must be positive when set")
        if len(set(self.data.exclude_question_ids)) != len(
            self.data.exclude_question_ids
        ):
            raise ValueError("data.exclude_question_ids must not contain duplicates")
        if self.data.max_input_length <= 0:
            raise ValueError("data.max_input_length must be positive")
        if self.sampling.num_candidates < 2:
            raise ValueError("sampling.num_candidates must be at least 2")
        if self.sampling.prompt_batch_size <= 0:
            raise ValueError("sampling.prompt_batch_size must be positive")
        if self.sampling.max_new_tokens <= 0:
            raise ValueError("sampling.max_new_tokens must be positive")
        if self.sampling.temperature <= 0:
            raise ValueError("sampling.temperature must be positive")
        if not 0 < self.sampling.top_p <= 1:
            raise ValueError("sampling.top_p must be in (0, 1]")
        if self.execution.timeout_seconds <= 0:
            raise ValueError("execution.timeout_seconds must be positive")
        if self.execution.max_result_rows <= 0:
            raise ValueError("execution.max_result_rows must be positive")
        if not 0 < self.pairing.validation_ratio < 1:
            raise ValueError("pairing.validation_ratio must be in (0, 1)")
        if not self.output.directory.is_absolute():
            raise ValueError("output.dir must be an absolute path")

    def validate_inputs(self, *, require_adapter: bool, require_databases: bool) -> None:
        if not self.data.path.is_file():
            raise FileNotFoundError(f"DPO mining data does not exist: {self.data.path}")
        if require_adapter and not self.model.adapter_path.is_dir():
            raise FileNotFoundError(f"SFT adapter does not exist: {self.model.adapter_path}")
        if require_databases and not self.execution.db_root.is_dir():
            raise FileNotFoundError(
                f"Train database root does not exist: {self.execution.db_root}"
            )


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)

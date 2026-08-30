from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SftDataConfig:
    synthetic_path: Path
    train_path: Path
    raw_report_path: Path
    sft_train_path: Path
    sft_validation_path: Path
    sft_report_path: Path
    selection_method: str
    validation_ratio: float
    seed: int
    tokenizer_name_or_path: str
    cutoff_candidates: tuple[int, ...]
    target_coverage: float

    @classmethod
    def from_yaml(cls, path: str | Path) -> SftDataConfig:
        config_path = Path(path)
        with config_path.open(encoding="utf-8") as handle:
            raw: dict[str, Any] = yaml.safe_load(handle)

        paths = raw["paths"]
        selection = raw["selection"]
        tokenizer = raw["tokenizer"]
        config = cls(
            synthetic_path=Path(paths["synthetic"]),
            train_path=Path(paths["train"]),
            raw_report_path=Path(paths["raw_report"]),
            sft_train_path=Path(paths["sft_train"]),
            sft_validation_path=Path(paths["sft_validation"]),
            sft_report_path=Path(paths["sft_report"]),
            selection_method=str(selection["method"]),
            validation_ratio=float(selection["validation_ratio"]),
            seed=int(selection["seed"]),
            tokenizer_name_or_path=str(tokenizer["name_or_path"]),
            cutoff_candidates=tuple(int(value) for value in tokenizer["cutoff_candidates"]),
            target_coverage=float(tokenizer["target_coverage"]),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.selection_method != "median_gold_matching_response_length":
            raise ValueError(f"Unsupported selection method: {self.selection_method}")
        if not 0.0 < self.validation_ratio < 1.0:
            raise ValueError("validation_ratio must be between 0 and 1")
        if not self.cutoff_candidates:
            raise ValueError("At least one cutoff candidate is required")
        if tuple(sorted(set(self.cutoff_candidates))) != self.cutoff_candidates:
            raise ValueError("cutoff_candidates must be unique and sorted")
        if not 0.0 < self.target_coverage <= 1.0:
            raise ValueError("target_coverage must be in (0, 1]")

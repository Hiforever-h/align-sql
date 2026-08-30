from __future__ import annotations

from pathlib import Path

from align_sql.data.config import SftDataConfig


def test_load_data_config() -> None:
    config = SftDataConfig.from_yaml(Path("configs/data_sft.yaml"))
    assert config.validation_ratio == 0.02
    assert config.seed == 42
    assert config.cutoff_candidates == (2048, 3072, 4096)


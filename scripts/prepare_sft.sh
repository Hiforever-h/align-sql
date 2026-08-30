#!/usr/bin/env bash
set -euo pipefail

python -m align_sql.data.validate_raw --config configs/data_sft.yaml
python -m align_sql.data.build_sft --config configs/data_sft.yaml


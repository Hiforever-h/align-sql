#!/usr/bin/env bash
set -euo pipefail

python -m align_sql.training.sft --config configs/sft_qlora.yaml "$@"


#!/usr/bin/env bash
set -euo pipefail

python -m align_sql.evaluation.sft.evaluate --config configs/eval_base.yaml "$@"

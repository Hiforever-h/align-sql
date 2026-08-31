#!/usr/bin/env bash
set -euo pipefail

python -m align_sql.training.dpo.mine --config configs/dpo_mining.yaml "$@"

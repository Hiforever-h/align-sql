#!/usr/bin/env bash
set -euo pipefail

python -m align_sql.training.dpo.train --config configs/dpo_qlora.yaml "$@"

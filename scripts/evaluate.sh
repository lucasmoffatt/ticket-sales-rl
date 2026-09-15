#!/usr/bin/env bash
# Arguments are passed to the evaluation command.
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=src

python scripts/evaluate_q_learning.py "$@"

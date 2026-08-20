#!/usr/bin/env bash
# Evaluate and compare the trained agents, writing metrics and charts to data/.
# Extra arguments are forwarded, e.g.: ./scripts/evaluate.sh --episodes 200
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=src

python scripts/evaluate_q_learning.py "$@"

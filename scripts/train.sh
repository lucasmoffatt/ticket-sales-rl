#!/usr/bin/env bash
# Arguments are passed to both training commands.
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=src

echo "== Training tabular Q-learning =="
python scripts/train_q_learning.py "$@"

echo "== Training Keras DQN =="
python scripts/train_dqn.py "$@"

#!/usr/bin/env bash
# Train both pricing agents (tabular Q-learning, then the Keras DQN).
# Extra arguments are forwarded to both training scripts, e.g.:
#   ./scripts/train.sh --episodes 1000
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=src

echo "== Training tabular Q-learning =="
python scripts/train_q_learning.py "$@"

echo "== Training Keras DQN =="
python scripts/train_dqn.py "$@"

#!/usr/bin/env bash
# Create a virtual environment and install project dependencies.
set -euo pipefail

# Always run from the project root, regardless of where the script is called.
cd "$(dirname "$0")/.."

python3 -m venv .venv
# shellcheck source=/dev/null
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "Environment ready. Activate it with: source .venv/bin/activate"

#!/usr/bin/env bash
# Launch the interactive Streamlit dashboard.
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=src

streamlit run app/streamlit_app.py "$@"

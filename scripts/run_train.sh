#!/usr/bin/env bash
set -e

echo "Ensure your Python venv at ~/grahprnn_env is activated before running this script."
if [ -z "$VIRTUAL_ENV" ]; then
  echo "Warning: No virtualenv detected. Activate ~/grahprnn_env manually:"
  echo "  source ~/grahprnn_env/bin/activate"
fi

python -m src.train --config configs/example.yaml

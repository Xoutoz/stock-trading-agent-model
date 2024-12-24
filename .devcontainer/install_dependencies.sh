#!/bin/bash
set -e  # Exit on error

python -m venv .venv
source .venv/bin/activate

pip install -r .devcontainer/requirements.txt
pip install --upgrade pip

# echo 'export PYTHONPATH=$PYTHONPATH:/workspaces/stock-trading-agent-model/src' >> ~/.bashrc
echo 'source .venv/bin/activate' >> ~/.bashrc
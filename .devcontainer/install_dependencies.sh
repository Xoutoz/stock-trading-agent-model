#!/bin/bash
set -e  # Exit on error

pip install --upgrade pip

python -m venv .venv
source .venv/bin/activate

pip install -r .devcontainer/requirements.txt
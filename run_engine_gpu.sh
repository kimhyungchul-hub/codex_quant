#!/usr/bin/env bash
set -euo pipefail
source .venv311/bin/activate
# clear env to let code-set defaults
unset JAX_PLATFORM_NAME
unset JAX_PLATFORMS
exec python main.py

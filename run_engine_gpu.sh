#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate
# clear env to let code-set defaults
unset JAX_PLATFORM_NAME
unset JAX_PLATFORMS
exec python main_engine_mc_v2_final.py

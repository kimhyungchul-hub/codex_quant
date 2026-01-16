#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -x "$ROOT/.venv311/bin/activate" ]]; then
  source "$ROOT/.venv311/bin/activate"
elif [[ -x "$ROOT/venv_jax/bin/activate" ]]; then
  source "$ROOT/venv_jax/bin/activate"
elif [[ -x "$ROOT/.venv/bin/activate" ]]; then
  source "$ROOT/.venv/bin/activate"
fi
# clear env to let code-set defaults
unset JAX_PLATFORM_NAME
unset JAX_PLATFORMS
exec python "$ROOT/main.py"

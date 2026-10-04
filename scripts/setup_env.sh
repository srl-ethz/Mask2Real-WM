#!/usr/bin/env bash
# Create a conda environment with the pinned dependencies and check that the stack imports.
#
# Usage: scripts/setup_env.sh [env_name]   (default: mask2real-wm)
# Set ENV_PREFIX=/path/to/env to create the environment at a path instead of by name.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME=${1:-mask2real-wm}

if [[ -n "${ENV_PREFIX:-}" ]]; then
  conda create -y -p "$ENV_PREFIX" python=3.11 pip
  PY="$ENV_PREFIX/bin/python"
else
  conda create -y -n "$ENV_NAME" python=3.11 pip
  PY="$(conda run -n "$ENV_NAME" python -c 'import sys; print(sys.executable)')"
fi

"$PY" -m pip install --upgrade pip
"$PY" -m pip install -r "$REPO_ROOT/requirements-dev.txt"
# decord 0.6.0's wheel metadata claims a cp36 tag, so pip check flags it although it imports fine.
problems="$("$PY" -m pip check 2>&1 | grep -v -e '^decord 0.6.0 is not supported on this platform$' \
  -e '^No broken requirements found.$' || true)"
if [[ -n "$problems" ]]; then
  echo "$problems" >&2
  exit 1
fi

cd "$REPO_ROOT"
"$PY" - <<'EOF'
import torch, diffusers, transformers, accelerate, lpips, skimage, cv2, decord, torch_fidelity, moviepy
from models.ctrl_world import CtrlWorld  # noqa: F401
print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available())
EOF
command -v ffmpeg >/dev/null || echo "Warning: ffmpeg was not found on PATH; reading and writing videos needs it." >&2
echo "Environment ready. Run the unit tests with: $PY -m pytest"

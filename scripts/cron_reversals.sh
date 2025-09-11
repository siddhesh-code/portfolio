#!/usr/bin/env bash
set -euo pipefail

# Resolve repo root from this script's location
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Load .env if present (exports all vars)
if [[ -f .env ]]; then
  set -a
  # shellcheck source=/dev/null
  . ./.env
  set +a
fi

# Create venv if missing
if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip >/dev/null
pip install -q -r requirements.txt

# Optional flags from env
FLAGS=("--horizons" "--picks")
if [[ "${REVERSALS_EMAIL_MIN_SCORE:-}" != "" ]]; then
  FLAGS+=("--min-score" "${REVERSALS_EMAIL_MIN_SCORE}")
fi
if [[ "${REVERSALS_EMAIL_UNIVERSE:-}" != "" ]]; then
  FLAGS+=("--universe" "${REVERSALS_EMAIL_UNIVERSE}")
fi
if [[ "${DEBUG_EMAIL:-}" == "1" ]]; then
  FLAGS+=("--debug")
fi

export PYTHONUNBUFFERED=1
python scripts/send_reversals_email.py "${FLAGS[@]}"


#!/usr/bin/env bash
# HedgeRock evolution-sidecar bootstrap.
#
# Spins a venv, installs runtime + dev dependencies, copies
# ``.env.example`` to ``.env`` (if absent), and runs the evolution
# test suite to verify the install. No external data, no operator
# credentials, no LLM calls.
#
# Usage:
#   bash scripts/hedgerock_bootstrap.sh
#
# Idempotent: safe to re-run.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# 1. Locate a usable Python (>=3.11).
# ---------------------------------------------------------------------------

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
    for candidate in python3.12 python3.11 python3; do
        if command -v "$candidate" >/dev/null 2>&1; then
            ver="$("$candidate" -c 'import sys; print("{}.{}".format(*sys.version_info[:2]))' 2>/dev/null || echo "0.0")"
            major="${ver%%.*}"
            minor="${ver##*.}"
            if [[ "$major" -gt 3 ]] || { [[ "$major" == 3 ]] && [[ "$minor" -ge 11 ]]; }; then
                PYTHON_BIN="$candidate"
                break
            fi
        fi
    done
fi

if [[ -z "$PYTHON_BIN" ]]; then
    echo "ERROR: Python 3.11+ not found on PATH. Install it or set PYTHON_BIN." >&2
    exit 1
fi

echo "[bootstrap] using $PYTHON_BIN ($("$PYTHON_BIN" --version))"

# ---------------------------------------------------------------------------
# 2. Create / refresh the venv.
# ---------------------------------------------------------------------------

VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"
if [[ ! -d "$VENV_DIR" ]]; then
    echo "[bootstrap] creating venv at $VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
else
    echo "[bootstrap] reusing venv at $VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# ---------------------------------------------------------------------------
# 3. Install dependencies.
# ---------------------------------------------------------------------------

python -m pip install --upgrade pip wheel >/dev/null
echo "[bootstrap] installing ai-smc[dev]"
python -m pip install -e ".[dev]"

# ---------------------------------------------------------------------------
# 4. Seed .env from .env.example if missing (no overwrite).
# ---------------------------------------------------------------------------

if [[ -f ".env.example" ]] && [[ ! -f ".env" ]]; then
    cp .env.example .env
    echo "[bootstrap] copied .env.example → .env (edit as needed)"
fi

# ---------------------------------------------------------------------------
# 5. Run the evolution-sidecar test suite to verify the install.
# ---------------------------------------------------------------------------

echo "[bootstrap] running tests/hedgerock/evolution"
python -m pytest tests/hedgerock/evolution -q --no-header

cat <<'EOF'

[bootstrap] OK — evolution sidecar is installed and green.

Next steps:
  source .venv/bin/activate
  # Try the demo (writes only to a tmp workspace):
  python scripts/hedgerock_evolution_demo.py --workspace /tmp/hedgerock-demo
  # Read the operator runbook:
  open docs/hedgerock-quickstart.md

Override the operator-team registry layout via env vars (optional;
all defaults work on a fresh machine):
  HEDGEROCK_HOME=/path/to/HedgeRock        # default: $HOME/HedgeRock
  AI_SMC_HOME=/path/to/AI-SMC              # default: this repo

EOF

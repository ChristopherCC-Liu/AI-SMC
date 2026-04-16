#!/usr/bin/env bash
# AI-SMC Deployment Script — Linux/macOS (Mock Mode)
#
# This script sets up AI-SMC for local development and mock testing.
# On Linux/macOS, MT5 is not available natively, so SMC_MT5_MOCK=1 is enforced.
#
# Prerequisites:
#   - Python 3.11+
#   - Git
#
# Usage:
#   chmod +x scripts/deploy_vps.sh
#   ./scripts/deploy_vps.sh

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
INSTALL_DIR="${AI_SMC_DIR:-$(pwd)}"
PYTHON_MIN="3.11"
BACKFILL_START="2020-01-01"
BACKFILL_END="2025-01-01"

# ── Colors ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

step() { echo -e "\n${CYAN}===== $1 =====${NC}"; }
ok()   { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
fail() { echo -e "${RED}[FAIL]${NC} $1"; exit 1; }

# ── Step 0: Preflight ────────────────────────────────────────────────────────
step "Step 0: Preflight Checks"

# Check Python
command -v python3 >/dev/null 2>&1 || fail "python3 not found. Install Python 3.11+."
PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)
MIN_MAJOR=$(echo "$PYTHON_MIN" | cut -d. -f1)
MIN_MINOR=$(echo "$PYTHON_MIN" | cut -d. -f2)

if [ "$PY_MAJOR" -lt "$MIN_MAJOR" ] || { [ "$PY_MAJOR" -eq "$MIN_MAJOR" ] && [ "$PY_MINOR" -lt "$MIN_MINOR" ]; }; then
    fail "Python $PY_VER found, but $PYTHON_MIN+ is required."
fi
ok "Python $PY_VER"

# Check Git
command -v git >/dev/null 2>&1 || fail "git not found. Install Git."
ok "Git $(git --version | awk '{print $3}')"

# ── Step 1: Virtual environment ───────────────────────────────────────────────
step "Step 1: Create Virtual Environment"

cd "$INSTALL_DIR"

if [ ! -f ".venv/bin/python" ]; then
    python3 -m venv .venv
    ok "Virtual environment created."
else
    ok "Virtual environment already exists."
fi

# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip --quiet

# ── Step 2: Install dependencies ──────────────────────────────────────────────
step "Step 2: Install Dependencies"

pip install -e ".[dev]" --quiet
ok "Dependencies installed (dev extras, no MT5 on Linux/macOS)."

# ── Step 3: Configure environment ─────────────────────────────────────────────
step "Step 3: Configure Environment"

if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        ok "Copied .env.example -> .env"
    fi
fi

# Force mock mode on Linux/macOS
if grep -q "SMC_MT5_MOCK" .env 2>/dev/null; then
    sed -i.bak 's/^SMC_MT5_MOCK=.*/SMC_MT5_MOCK=1/' .env && rm -f .env.bak
else
    echo "SMC_MT5_MOCK=1" >> .env
fi
ok "SMC_MT5_MOCK=1 enforced (Linux/macOS mock mode)."

# Create required directories
mkdir -p data/parquet data/journal data/manifests logs
ok "Data directories created."

# ── Step 4: Data backfill ─────────────────────────────────────────────────────
step "Step 4: Historical Data Backfill"

echo "Backfilling XAUUSD data from $BACKFILL_START to $BACKFILL_END ..."

if python scripts/backfill_data.py --start "$BACKFILL_START" --end "$BACKFILL_END"; then
    ok "Data backfill complete."
else
    warn "Backfill failed. You can retry later:"
    echo "  python scripts/backfill_data.py --start $BACKFILL_START --end $BACKFILL_END"
fi

# ── Step 5: Run tests ────────────────────────────────────────────────────────
step "Step 5: Run Tests"

if pytest tests/ -x -q --tb=short 2>/dev/null; then
    ok "All tests passed."
else
    warn "Some tests failed. Check output above."
fi

# ── Step 6: Health check ─────────────────────────────────────────────────────
step "Step 6: Health Check"

python -m smc.cli.main health

# ── Done ──────────────────────────────────────────────────────────────────────
step "Deployment Complete"

echo ""
echo -e "${GREEN}To start mock paper trading:${NC}"
echo "  cd $INSTALL_DIR"
echo "  source .venv/bin/activate"
echo "  SMC_MT5_MOCK=1 python -m smc.cli.main live --mode demo"
echo ""
echo -e "${YELLOW}Note: Real MT5 trading requires Windows VPS. Use deploy_vps.ps1 instead.${NC}"

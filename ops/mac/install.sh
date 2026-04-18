#!/bin/bash
# AI-SMC Mac launchd agent installer
# Run from the project root: bash ops/mac/install.sh
#
# Auth setup (pick one — script will guide you):
#   A) ssh-key (recommended, no plaintext credentials):
#        ssh-keygen -t ed25519 -f ~/.ssh/id_aismc -N ""
#        ssh-copy-id -i ~/.ssh/id_aismc.pub Administrator@<VPS>
#   B) password via ~/.aismc/env (chmod 600):
#        mkdir -p ~/.aismc && chmod 700 ~/.aismc
#        echo "export SSHPASS='your-password'" > ~/.aismc/env
#        chmod 600 ~/.aismc/env

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCH_AGENTS="$HOME/Library/LaunchAgents"
VPS_HOST_DEFAULT="Administrator@43.163.107.158"

echo "==> Installing AI-SMC Mac launchd agents"

# ── Auth setup (ssh-key preferred) ──────────────────────────────────────────

SSH_KEY="$HOME/.ssh/id_aismc"

if [ -f "$SSH_KEY" ]; then
    echo "  SSH key detected: $SSH_KEY — using key auth (no password needed)"
elif [ -f "$HOME/.aismc/env" ]; then
    echo "  ~/.aismc/env detected — using password auth fallback"
    if ! command -v sshpass >/dev/null 2>&1; then
        echo "ERROR: sshpass required when using password auth. Install: brew install sshpass" >&2
        exit 1
    fi
else
    cat >&2 <<EOF
ERROR: no auth configured. Pick one:

  A) ssh-key (recommended):
       ssh-keygen -t ed25519 -f ~/.ssh/id_aismc -N ""
       ssh-copy-id -i ~/.ssh/id_aismc.pub $VPS_HOST_DEFAULT

  B) password file (fallback):
       mkdir -p ~/.aismc && chmod 700 ~/.aismc
       printf "export SSHPASS='%s'\n" 'your-password' > ~/.aismc/env
       chmod 600 ~/.aismc/env
       brew install sshpass

Rerun this installer after either A or B is done.
EOF
    exit 1
fi

mkdir -p "$HOME/Backups/AI-SMC/XAUUSD" "$HOME/Backups/AI-SMC/BTCUSD"
echo "  Backup directories ready: ~/Backups/AI-SMC/{XAUUSD,BTCUSD}"

# ── Unload existing (if loaded) ─────────────────────────────────────────────

for label in com.aismc.backup com.aismc.health; do
    if launchctl list | grep -q "$label" 2>/dev/null; then
        echo "  Unloading existing: $label"
        launchctl unload "$LAUNCH_AGENTS/$label.plist" 2>/dev/null || true
    fi
done

# ── Copy plists ─────────────────────────────────────────────────────────────

mkdir -p "$LAUNCH_AGENTS"
cp "$SCRIPT_DIR/com.aismc.backup.plist" "$LAUNCH_AGENTS/"
cp "$SCRIPT_DIR/com.aismc.health.plist" "$LAUNCH_AGENTS/"
echo "  Copied plists to $LAUNCH_AGENTS"

# ── Load ────────────────────────────────────────────────────────────────────

launchctl load "$LAUNCH_AGENTS/com.aismc.backup.plist"
launchctl load "$LAUNCH_AGENTS/com.aismc.health.plist"

echo ""
echo "Installed successfully."
echo ""
echo "Status:"
launchctl list | grep aismc || echo "  (none found — may need a moment to register)"
echo ""
echo "Logs:"
echo "  Backup:  tail -f /tmp/aismc_backup.log"
echo "  Health:  tail -f /tmp/aismc_health.log"
echo ""
echo "To uninstall:"
echo "  launchctl unload ~/Library/LaunchAgents/com.aismc.backup.plist"
echo "  launchctl unload ~/Library/LaunchAgents/com.aismc.health.plist"
echo "  rm ~/Library/LaunchAgents/com.aismc.{backup,health}.plist"

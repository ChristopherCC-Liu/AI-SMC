#!/bin/bash
# AI-SMC Mac launchd agent installer
# Run from the project root: bash ops/mac/install.sh
#
# Prerequisites:
#   brew install sshpass
#   mkdir -p ~/Backups/AI-SMC/XAUUSD ~/Backups/AI-SMC/BTCUSD
#
# Optional (recommended - avoids storing password in plist):
#   ssh-keygen -t ed25519 -f ~/.ssh/id_aismc -N ""
#   ssh-copy-id -i ~/.ssh/id_aismc.pub Administrator@43.163.107.158
#   Then edit plist files to use: ssh -i ~/.ssh/id_aismc (no sshpass)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCH_AGENTS="$HOME/Library/LaunchAgents"

echo "==> Installing AI-SMC Mac launchd agents"

# ── Verify prerequisites ────────────────────────────────────────────────────

if ! command -v sshpass >/dev/null 2>&1; then
    echo "ERROR: sshpass not found. Install with: brew install sshpass"
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

# AI-SMC VPS Deployment Guide

Complete step-by-step guide for deploying AI-SMC v1 paper trading on a Windows VPS with TMGM MT5 Demo.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [MT5 Terminal Setup](#mt5-terminal-setup)
3. [Install AI-SMC](#install-ai-smc)
4. [Configure Environment](#configure-environment)
5. [Data Backfill](#data-backfill)
6. [Health Pre-flight Checks](#health-pre-flight-checks)
7. [Start Paper Trading](#start-paper-trading)
8. [Auto-Restart with nssm](#auto-restart-with-nssm)
9. [Telegram Alert Setup](#telegram-alert-setup)
10. [Journal & Backup](#journal--backup)
11. [Monitoring Checklist](#monitoring-checklist)
12. [Updating AI-SMC](#updating-ai-smc)
13. [Troubleshooting](#troubleshooting)

---

## Prerequisites

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| OS | Windows Server 2019 | Windows Server 2022 |
| CPU | 2 cores | 4 cores |
| RAM | 4 GB | 8 GB |
| Disk | 20 GB SSD | 50 GB SSD |
| Python | 3.11 | 3.12 |
| Network | Stable internet | Low-latency to broker |

### Install Python

1. Download Python 3.12+ from [python.org/downloads](https://www.python.org/downloads/).
2. During installation, **check "Add Python to PATH"**.
3. Verify in PowerShell:
   ```powershell
   python --version
   # Expected: Python 3.12.x
   ```

### Install Git

1. Download from [git-scm.com/download/win](https://git-scm.com/download/win).
2. Use default options during installation.
3. Verify:
   ```powershell
   git --version
   ```

---

## MT5 Terminal Setup

### 1. Install MetaTrader 5

Download the TMGM MT5 from your broker or from [metatrader5.com](https://www.metatrader5.com/en/download).

### 2. Log into TMGM Demo Account

1. Launch MT5 Terminal.
2. Go to **File > Open an Account**.
3. Search for **TMGM** (or your broker server name).
4. Select **TMGM-Demo** server.
5. Create a demo account or log in with existing credentials.
6. **Note your login number, password, and server name** -- you will need them for `.env`.

### 3. Enable XAUUSD

1. In MT5 Terminal, right-click **Market Watch** panel.
2. Click **Symbols**.
3. Search for **XAUUSD** and click **Show**.
4. Verify XAUUSD appears in Market Watch with live prices.

### 4. Enable Algorithmic Trading

1. Go to **Tools > Options > Expert Advisors**.
2. Check **Allow algorithmic trading**.
3. Check **Allow DLL imports** (required for MetaTrader5 Python package).

### 5. Keep MT5 Running

The MT5 Terminal **must be running** for the Python API to connect. On a VPS, you can:
- Set MT5 to start automatically on Windows login.
- Use a Remote Desktop session that stays active.

---

## Install AI-SMC

### Automated (recommended)

Run the deployment script in PowerShell (as Administrator):

```powershell
# Allow script execution for this session
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

# Clone and run
git clone https://github.com/ChristopherCC-Liu/AI-SMC.git C:\AI-SMC
cd C:\AI-SMC
.\scripts\deploy_vps.ps1
```

The script handles: clone, venv, install, configure, backfill, and health check.

### Manual

```powershell
# 1. Clone
git clone https://github.com/ChristopherCC-Liu/AI-SMC.git C:\AI-SMC
cd C:\AI-SMC

# 2. Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install with MT5 support
pip install -e ".[mt5,dev]"

# 4. Copy and edit configuration
Copy-Item .env.production .env
notepad .env
# Fill in MT5_LOGIN, MT5_PASSWORD, MT5_SERVER

# 5. Create data directories
mkdir data\parquet, data\journal, data\manifests, logs -Force
```

---

## Configure Environment

Copy the production template and edit:

```powershell
Copy-Item .env.production .env
notepad .env
```

### Required Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `SMC_MT5_LOGIN` | MT5 account number | `12345678` |
| `SMC_MT5_PASSWORD` | MT5 account password | `yourpassword` |
| `SMC_MT5_SERVER` | Broker server name | `TMGM-Demo` |
| `SMC_MT5_MOCK` | Must be `0` on Windows VPS | `0` |
| `SMC_ENV` | Trading environment | `paper` |

### Risk Variables (conservative defaults)

| Variable | Default | Description |
|----------|---------|-------------|
| `SMC_RISK_PER_TRADE_PCT` | `1.0` | % equity risked per trade |
| `SMC_MAX_DAILY_LOSS_PCT` | `3.0` | Daily loss halt threshold |
| `SMC_MAX_DRAWDOWN_PCT` | `10.0` | Drawdown halt threshold |
| `SMC_MAX_LOT_SIZE` | `0.01` | Maximum lot size per order |

### Optional Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SMC_AI_REGIME_ENABLED` | `0` | AI regime classifier (v1.1+) |
| `SMC_TELEGRAM_BOT_TOKEN` | _(empty)_ | Telegram alert bot token |
| `SMC_TELEGRAM_CHAT_ID` | _(empty)_ | Telegram chat ID |
| `SMC_LOG_LEVEL` | `INFO` | Logging verbosity |

---

## Data Backfill

Before running live, you need historical XAUUSD data for the strategy's multi-timeframe analysis:

```powershell
# Activate virtual environment
cd C:\AI-SMC
.\.venv\Scripts\Activate.ps1

# Backfill 5 years of XAUUSD data
python scripts/backfill_data.py --start 2020-01-01 --end 2025-01-01
```

This process:
- Downloads XAUUSD OHLCV data for all required timeframes (D1, H4, H1, M15).
- Stores data as Parquet files in `data/parquet/`.
- Creates manifests in `data/manifests/` for data provenance tracking.

Verify the data lake:
```powershell
python -m smc.cli.main lake-info
```

---

## Health Pre-flight Checks

Run the health check before starting trading:

```powershell
python -m smc.cli.main health
```

Expected output for a properly configured VPS:

| Check | Expected Status |
|-------|----------------|
| Config | OK |
| Data directory | OK |
| Data freshness | OK (recent data) |
| MT5 connection | OK (Connected) |
| LLM (Anthropic) | WARN (not set, expected for v1) |
| Telegram alerts | OK or WARN |

**All critical checks (Config, Data, MT5) must be OK before proceeding.**

If MT5 connection shows FAIL:
1. Verify MT5 Terminal is running and logged in.
2. Check `.env` credentials match MT5 Terminal.
3. Ensure "Allow algorithmic trading" is enabled.
4. Restart MT5 Terminal and try again.

---

## Start Paper Trading

### Manual start (for testing)

```powershell
cd C:\AI-SMC
.\.venv\Scripts\Activate.ps1
python -m smc.cli.main live --mode demo
```

The system will:
1. Connect to MT5 Terminal.
2. Poll for new M15 bar closes.
3. Run the SMC multi-timeframe strategy pipeline.
4. Execute paper trades via the risk/execution layer.
5. Log all activity to `logs/`.

Press `Ctrl+C` to stop gracefully.

### Verify first signals

After starting, watch the console output for:
- `[HEALTH] Config OK` -- configuration loaded.
- `[LOOP] Waiting for next M15 bar...` -- polling active.
- `[SIGNAL] ...` -- strategy generating signals (may take time depending on market conditions).

---

## Auto-Restart with nssm

For production use, install AI-SMC as a Windows service that auto-starts on boot:

```powershell
# Run as Administrator
.\scripts\install_service.ps1
```

This installs AI-SMC using [nssm](https://nssm.cc/) (Non-Sucking Service Manager):
- **Auto-starts** on Windows boot.
- **Auto-restarts** on crash (30-second cooldown).
- **Rotates logs** daily or at 10 MB.

### Service management

```powershell
nssm status AI-SMC        # Check status
nssm stop AI-SMC          # Stop
nssm start AI-SMC         # Start
nssm restart AI-SMC       # Restart
nssm edit AI-SMC           # Edit settings (GUI)
```

### Uninstall service

```powershell
.\scripts\install_service.ps1 -Uninstall
```

---

## Telegram Alert Setup

Optional but recommended for monitoring trades remotely.

### 1. Create a Telegram Bot

1. Open Telegram and message [@BotFather](https://t.me/BotFather).
2. Send `/newbot` and follow the prompts.
3. Copy the **bot token** (e.g., `123456789:ABCdefGHIjklMNOpqrsTUVwxyz`).

### 2. Get Your Chat ID

1. Message your new bot (send any message).
2. Visit `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates` in a browser.
3. Find `"chat":{"id": <NUMBER>}` in the JSON response.
4. Copy the chat ID number.

### 3. Configure .env

```ini
SMC_TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz
SMC_TELEGRAM_CHAT_ID=-1001234567890
```

### 4. Verify

Run `smc health` and check that "Telegram alerts" shows OK.

---

## Journal & Backup

### Journal location

All trade activity is logged to:

```
C:\AI-SMC\logs\               # Application logs (stdout/stderr)
C:\AI-SMC\data\journal\       # Trade journal (Parquet format)
```

### Backup strategy

Set up a daily backup of the data directory:

```powershell
# Example: daily backup to a network drive or cloud folder
# Add to Windows Task Scheduler
$src = "C:\AI-SMC\data"
$dst = "D:\Backups\AI-SMC\data-$(Get-Date -Format 'yyyy-MM-dd')"
Copy-Item -Recurse -Force $src $dst
```

Key files to backup:
- `data/journal/` -- trade history (irreplaceable).
- `data/parquet/` -- price data (can be re-downloaded, but slow).
- `data/manifests/` -- data provenance records.
- `.env` -- configuration (keep separate, contains credentials).
- `logs/` -- diagnostic logs (rotate weekly).

---

## Monitoring Checklist

### First 2 weeks of paper trading

Daily checks:

- [ ] Service is running: `nssm status AI-SMC`
- [ ] MT5 Terminal is logged in and connected.
- [ ] No errors in `logs/stderr.log`.
- [ ] Trade journal has new entries (if market is open).
- [ ] Account equity is tracking as expected (check MT5 Terminal).

Weekly checks:

- [ ] Review all trades in journal: win rate, R:R, drawdown.
- [ ] Compare with manual SMC analysis for consistency.
- [ ] Check data freshness: `smc lake-info`.
- [ ] Verify risk limits are not being breached.
- [ ] Review and rotate old log files.

### Key metrics to watch

| Metric | Acceptable Range (v1) | Action if Breached |
|--------|----------------------|-------------------|
| Win Rate | > 35% | Review strategy parameters |
| Profit Factor | > 1.5 | Check entry/exit logic |
| Max Drawdown | < 10% | System auto-halts |
| Daily Loss | < 3% | System auto-halts |
| Avg R:R | > 2.0 | Check SL/TP placement |

---

## Updating AI-SMC

To update to a new version:

```powershell
# Stop the service
nssm stop AI-SMC

# Pull latest code
cd C:\AI-SMC
git fetch origin
git pull origin main

# Update dependencies
.\.venv\Scripts\Activate.ps1
pip install -e ".[mt5,dev]"

# Run health check
python -m smc.cli.main health

# Restart
nssm start AI-SMC
```

---

## Troubleshooting

### MT5 Connection Fails

**Symptom**: `smc health` shows MT5 connection FAIL.

**Fixes**:
1. Ensure MT5 Terminal is running and logged in.
2. Verify credentials in `.env` match exactly (case-sensitive).
3. Check "Allow algorithmic trading" in MT5 settings.
4. Restart MT5 Terminal.
5. If using RDP, ensure the session stays active (use `tscon` or a keep-alive tool).

### Service Won't Start

**Symptom**: `nssm status AI-SMC` shows "Stopped" or "Paused".

**Fixes**:
1. Check `logs/stderr.log` for error messages.
2. Verify `.env` exists and is correctly configured.
3. Test manually first: `.\.venv\Scripts\Activate.ps1 && python -m smc.cli.main live --mode demo`
4. Check Python path: `nssm get AI-SMC Application`

### Data Backfill Fails

**Symptom**: `backfill_data.py` exits with an error.

**Fixes**:
1. Check internet connectivity.
2. Verify MT5 Terminal has XAUUSD enabled in Market Watch.
3. Try a smaller date range first: `--start 2024-01-01 --end 2025-01-01`
4. Check firewall rules are not blocking outbound connections.

### High Memory Usage

**Symptom**: Python process using excessive RAM.

**Fixes**:
1. Check data lake size: `smc lake-info`.
2. Consider reducing backfill range.
3. Restart the service: `nssm restart AI-SMC`.

### No Trades Being Generated

**Symptom**: System is running but no trades appear.

**This is normal.** The SMC strategy is selective:
- It requires multi-timeframe confluence (D1+H4+H1+M15).
- It filters by regime (trending vs ranging).
- It enforces minimum R:R ratio of 2.0.
- During low-volatility periods, few setups qualify.

Check the logs for signal activity:
```powershell
Select-String -Path "logs\stdout.log" -Pattern "SIGNAL|SETUP|TRADE" | Select-Object -Last 20
```

### Telegram Alerts Not Working

**Fixes**:
1. Verify bot token and chat ID in `.env`.
2. Ensure you have messaged the bot at least once.
3. Test the bot manually:
   ```powershell
   $token = "YOUR_TOKEN"
   $chatId = "YOUR_CHAT_ID"
   Invoke-RestMethod "https://api.telegram.org/bot$token/sendMessage" -Method POST -Body @{
       chat_id = $chatId
       text = "AI-SMC test alert"
   }
   ```

### Python Import Errors

**Symptom**: `ModuleNotFoundError: No module named 'MetaTrader5'`

**Fixes**:
1. Ensure you installed with MT5 extra: `pip install -e ".[mt5,dev]"`
2. Verify you are using the correct venv: `where python` should show `C:\AI-SMC\.venv\Scripts\python.exe`
3. Reinstall: `pip install --force-reinstall MetaTrader5`

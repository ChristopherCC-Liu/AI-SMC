#Requires -Version 5.1
<#
.SYNOPSIS
    AI-SMC VPS Deployment Script for Windows Server.
.DESCRIPTION
    Deploys the AI-SMC trading system on a Windows VPS with MT5 Terminal.
    Run as Administrator on a fresh Windows Server with Python 3.11+ and Git installed.
.NOTES
    Prerequisites:
      - Windows Server 2019+ or Windows 10/11
      - Python 3.11+ (python.exe in PATH)
      - Git (git.exe in PATH)
      - MetaTrader 5 Terminal installed and logged into TMGM Demo
      - Internet access for pip install
.EXAMPLE
    # Run from PowerShell (as Administrator):
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
    .\deploy_vps.ps1
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ── Configuration ─────────────────────────────────────────────────────────────
$INSTALL_DIR   = "C:\AI-SMC"
$REPO_URL      = "https://github.com/ChristopherCC-Liu/AI-SMC.git"
$BRANCH        = "main"
$PYTHON_MIN    = "3.11"
$BACKFILL_START = "2020-01-01"
$BACKFILL_END   = "2025-01-01"

# ── Helper functions ──────────────────────────────────────────────────────────

function Write-Step {
    param([string]$Message)
    Write-Host "`n=====================================" -ForegroundColor Cyan
    Write-Host "  $Message" -ForegroundColor Cyan
    Write-Host "=====================================" -ForegroundColor Cyan
}

function Assert-Command {
    param([string]$Name, [string]$HelpUrl)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        Write-Host "[FAIL] '$Name' not found in PATH." -ForegroundColor Red
        if ($HelpUrl) { Write-Host "  Install from: $HelpUrl" -ForegroundColor Yellow }
        exit 1
    }
    Write-Host "[OK] $Name found: $(Get-Command $Name | Select-Object -ExpandProperty Source)" -ForegroundColor Green
}

function Assert-PythonVersion {
    $version = python --version 2>&1
    if ($version -match "Python (\d+\.\d+)") {
        $ver = [version]$Matches[1]
        $min = [version]$PYTHON_MIN
        if ($ver -lt $min) {
            Write-Host "[FAIL] Python $ver found, but $PYTHON_MIN+ is required." -ForegroundColor Red
            exit 1
        }
        Write-Host "[OK] Python $ver" -ForegroundColor Green
    } else {
        Write-Host "[FAIL] Could not determine Python version." -ForegroundColor Red
        exit 1
    }
}

# ── Step 0: Preflight checks ─────────────────────────────────────────────────
Write-Step "Step 0: Preflight Checks"

Assert-Command "python" "https://www.python.org/downloads/"
Assert-Command "git"    "https://git-scm.com/download/win"
Assert-PythonVersion

# Check MT5 Terminal is installed
$mt5Paths = @(
    "$env:ProgramFiles\MetaTrader 5\terminal64.exe",
    "${env:ProgramFiles(x86)}\MetaTrader 5\terminal.exe",
    "$env:LOCALAPPDATA\Programs\MetaTrader 5\terminal64.exe"
)
$mt5Found = $false
foreach ($p in $mt5Paths) {
    if (Test-Path $p) {
        Write-Host "[OK] MT5 Terminal found: $p" -ForegroundColor Green
        $mt5Found = $true
        break
    }
}
if (-not $mt5Found) {
    Write-Host "[WARN] MT5 Terminal not found in standard paths." -ForegroundColor Yellow
    Write-Host "  Ensure MetaTrader 5 is installed and logged into your TMGM Demo account." -ForegroundColor Yellow
}

# ── Step 1: Clone repository ─────────────────────────────────────────────────
Write-Step "Step 1: Clone Repository"

if (Test-Path $INSTALL_DIR) {
    Write-Host "Directory $INSTALL_DIR already exists." -ForegroundColor Yellow
    $choice = Read-Host "  (U)pdate via git pull, (D)elete and re-clone, or (S)kip? [U/D/S]"
    switch ($choice.ToUpper()) {
        "D" {
            Remove-Item -Recurse -Force $INSTALL_DIR
            git clone --branch $BRANCH $REPO_URL $INSTALL_DIR
        }
        "U" {
            Push-Location $INSTALL_DIR
            git fetch origin
            git checkout $BRANCH
            git pull origin $BRANCH
            Pop-Location
        }
        "S" {
            Write-Host "Skipping clone." -ForegroundColor Yellow
        }
        default {
            Write-Host "Invalid choice. Aborting." -ForegroundColor Red
            exit 1
        }
    }
} else {
    git clone --branch $BRANCH $REPO_URL $INSTALL_DIR
}

Set-Location $INSTALL_DIR

# ── Step 2: Python virtual environment ────────────────────────────────────────
Write-Step "Step 2: Create Python Virtual Environment"

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    python -m venv .venv
    Write-Host "[OK] Virtual environment created." -ForegroundColor Green
} else {
    Write-Host "[OK] Virtual environment already exists." -ForegroundColor Green
}

# Activate
& .\.venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip

# Install package with MT5 and dev extras
Write-Host "Installing ai-smc with [mt5,dev] extras..." -ForegroundColor Cyan
pip install -e ".[mt5,dev]"

Write-Host "[OK] Dependencies installed." -ForegroundColor Green

# ── Step 3: Configure environment ─────────────────────────────────────────────
Write-Step "Step 3: Configure Environment"

if (-not (Test-Path ".env")) {
    if (Test-Path ".env.production") {
        Copy-Item ".env.production" ".env"
        Write-Host "[OK] Copied .env.production -> .env" -ForegroundColor Green
    } elseif (Test-Path ".env.example") {
        Copy-Item ".env.example" ".env"
        Write-Host "[OK] Copied .env.example -> .env" -ForegroundColor Green
    } else {
        Write-Host "[WARN] No .env template found. Create .env manually." -ForegroundColor Yellow
    }
    Write-Host ""
    Write-Host "  >>> IMPORTANT: Edit .env now with your MT5 credentials <<<" -ForegroundColor Yellow
    Write-Host "  Required fields:" -ForegroundColor Yellow
    Write-Host "    SMC_MT5_LOGIN=<your_demo_account_number>" -ForegroundColor White
    Write-Host "    SMC_MT5_PASSWORD=<your_password>" -ForegroundColor White
    Write-Host "    SMC_MT5_SERVER=TMGM-Demo" -ForegroundColor White
    Write-Host "    SMC_MT5_MOCK=0" -ForegroundColor White
    Write-Host ""
    $editNow = Read-Host "Open .env in notepad now? [Y/N]"
    if ($editNow.ToUpper() -eq "Y") {
        notepad .env
        Read-Host "Press Enter after saving .env to continue"
    }
} else {
    Write-Host "[OK] .env already exists." -ForegroundColor Green
}

# Create required directories
$dirs = @("data\parquet", "data\journal", "data\manifests", "logs")
foreach ($d in $dirs) {
    if (-not (Test-Path $d)) {
        New-Item -ItemType Directory -Path $d -Force | Out-Null
    }
}
Write-Host "[OK] Data directories created." -ForegroundColor Green

# ── Step 4: Data backfill ─────────────────────────────────────────────────────
Write-Step "Step 4: Historical Data Backfill"

Write-Host "Backfilling XAUUSD data from $BACKFILL_START to $BACKFILL_END ..." -ForegroundColor Cyan
Write-Host "(This may take several minutes depending on your connection)" -ForegroundColor Yellow

try {
    python scripts/backfill_data.py --start $BACKFILL_START --end $BACKFILL_END
    Write-Host "[OK] Data backfill complete." -ForegroundColor Green
} catch {
    Write-Host "[WARN] Backfill failed: $_" -ForegroundColor Yellow
    Write-Host "  You can retry later: python scripts/backfill_data.py --start $BACKFILL_START --end $BACKFILL_END" -ForegroundColor Yellow
}

# ── Step 5: Health check ──────────────────────────────────────────────────────
Write-Step "Step 5: Health Check"

python -m smc.cli.main health

# ── Step 6: Start paper trading ───────────────────────────────────────────────
Write-Step "Step 6: Ready to Start Paper Trading"

Write-Host ""
Write-Host "Deployment complete! To start paper trading:" -ForegroundColor Green
Write-Host ""
Write-Host "  cd $INSTALL_DIR" -ForegroundColor White
Write-Host '  .\.venv\Scripts\Activate.ps1' -ForegroundColor White
Write-Host '  python -m smc.cli.main live --mode demo' -ForegroundColor White
Write-Host ""
Write-Host "Or install as a Windows service (auto-restart):" -ForegroundColor Green
Write-Host '  .\scripts\install_service.ps1' -ForegroundColor White
Write-Host ""
Write-Host "Monitor logs at: $INSTALL_DIR\logs\" -ForegroundColor Yellow

$startNow = Read-Host "Start paper trading now? [Y/N]"
if ($startNow.ToUpper() -eq "Y") {
    Write-Host "Starting live loop (Ctrl+C to stop)..." -ForegroundColor Cyan
    python -m smc.cli.main live --mode demo
}

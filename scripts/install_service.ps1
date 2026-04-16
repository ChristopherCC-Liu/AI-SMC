#Requires -RunAsAdministrator
<#
.SYNOPSIS
    Install AI-SMC as a Windows service using nssm.
.DESCRIPTION
    Downloads nssm (if needed) and registers AI-SMC paper trading as a Windows
    service that auto-starts on boot and restarts on failure.
.NOTES
    Must be run as Administrator.
    Assumes AI-SMC is installed at C:\AI-SMC with a configured .env file.
.EXAMPLE
    # Install and start:
    .\install_service.ps1

    # Uninstall:
    .\install_service.ps1 -Uninstall
#>

param(
    [switch]$Uninstall,
    [string]$InstallDir = "C:\AI-SMC",
    [string]$ServiceName = "AI-SMC"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ── Helper ────────────────────────────────────────────────────────────────────

function Write-Step {
    param([string]$Message)
    Write-Host "`n>> $Message" -ForegroundColor Cyan
}

# ── Locate or download nssm ──────────────────────────────────────────────────

function Get-Nssm {
    # Check if nssm is already in PATH
    $existing = Get-Command nssm -ErrorAction SilentlyContinue
    if ($existing) {
        return $existing.Source
    }

    # Check local tools directory
    $localNssm = "$InstallDir\tools\nssm.exe"
    if (Test-Path $localNssm) {
        return $localNssm
    }

    # Download nssm
    Write-Step "Downloading nssm..."
    $toolsDir = "$InstallDir\tools"
    if (-not (Test-Path $toolsDir)) {
        New-Item -ItemType Directory -Path $toolsDir -Force | Out-Null
    }

    $nssmUrl = "https://nssm.cc/release/nssm-2.24.zip"
    $zipPath = "$toolsDir\nssm.zip"

    try {
        Invoke-WebRequest -Uri $nssmUrl -OutFile $zipPath -UseBasicParsing
        Expand-Archive -Path $zipPath -DestinationPath $toolsDir -Force
        $nssmExe = Get-ChildItem -Recurse "$toolsDir\nssm-*\win64\nssm.exe" | Select-Object -First 1
        if ($nssmExe) {
            Copy-Item $nssmExe.FullName $localNssm
            Remove-Item $zipPath -Force
            return $localNssm
        }
    } catch {
        Write-Host "[FAIL] Could not download nssm." -ForegroundColor Red
        Write-Host "  Download manually from https://nssm.cc/ and place nssm.exe in $toolsDir" -ForegroundColor Yellow
        exit 1
    }

    Write-Host "[FAIL] nssm.exe not found after extraction." -ForegroundColor Red
    exit 1
}

# ── Uninstall ─────────────────────────────────────────────────────────────────

if ($Uninstall) {
    Write-Step "Uninstalling $ServiceName service..."

    $nssm = Get-Nssm

    # Stop service if running
    $svc = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
    if ($svc -and $svc.Status -eq "Running") {
        & $nssm stop $ServiceName
        Start-Sleep -Seconds 2
    }

    & $nssm remove $ServiceName confirm
    Write-Host "[OK] Service '$ServiceName' removed." -ForegroundColor Green
    exit 0
}

# ── Validate installation ────────────────────────────────────────────────────

Write-Step "Validating AI-SMC installation..."

$pythonExe = "$InstallDir\.venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    Write-Host "[FAIL] Virtual environment not found at $InstallDir\.venv" -ForegroundColor Red
    Write-Host "  Run deploy_vps.ps1 first." -ForegroundColor Yellow
    exit 1
}

$envFile = "$InstallDir\.env"
if (-not (Test-Path $envFile)) {
    Write-Host "[FAIL] .env not found at $InstallDir" -ForegroundColor Red
    Write-Host "  Copy .env.production to .env and configure MT5 credentials." -ForegroundColor Yellow
    exit 1
}

Write-Host "[OK] Installation validated." -ForegroundColor Green

# ── Create log directory ──────────────────────────────────────────────────────

$logDir = "$InstallDir\logs"
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
}

# ── Install service ───────────────────────────────────────────────────────────

Write-Step "Installing $ServiceName as Windows service..."

$nssm = Get-Nssm

# Remove existing service if present
$existing = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "Service '$ServiceName' already exists. Removing old installation..." -ForegroundColor Yellow
    if ($existing.Status -eq "Running") {
        & $nssm stop $ServiceName
        Start-Sleep -Seconds 2
    }
    & $nssm remove $ServiceName confirm
}

# Install
& $nssm install $ServiceName $pythonExe "-m" "smc.cli.main" "live" "--mode" "demo"

# Configure service parameters
& $nssm set $ServiceName AppDirectory $InstallDir
& $nssm set $ServiceName DisplayName "AI-SMC Paper Trading"
& $nssm set $ServiceName Description "AI-SMC Smart Money Concepts paper trading system for XAUUSD"
& $nssm set $ServiceName Start SERVICE_AUTO_START

# Logging
& $nssm set $ServiceName AppStdout "$logDir\stdout.log"
& $nssm set $ServiceName AppStderr "$logDir\stderr.log"
& $nssm set $ServiceName AppStdoutCreationDisposition 4  # Append
& $nssm set $ServiceName AppStderrCreationDisposition 4  # Append
& $nssm set $ServiceName AppRotateFiles 1
& $nssm set $ServiceName AppRotateSeconds 86400           # Rotate daily
& $nssm set $ServiceName AppRotateBytes 10485760          # Rotate at 10 MB

# Restart on failure (wait 30s before restart, max 3 restarts in 60s)
& $nssm set $ServiceName AppThrottle 30000
& $nssm set $ServiceName AppExit Default Restart

# Environment: ensure .env is loaded from the app directory
& $nssm set $ServiceName AppEnvironmentExtra "PYTHONUNBUFFERED=1"

Write-Host "[OK] Service '$ServiceName' installed." -ForegroundColor Green

# ── Start service ─────────────────────────────────────────────────────────────

Write-Step "Starting $ServiceName service..."

& $nssm start $ServiceName

Start-Sleep -Seconds 3

$svc = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
if ($svc -and $svc.Status -eq "Running") {
    Write-Host "[OK] Service '$ServiceName' is running." -ForegroundColor Green
} else {
    Write-Host "[WARN] Service may not have started. Check logs:" -ForegroundColor Yellow
    Write-Host "  $logDir\stderr.log" -ForegroundColor Yellow
}

# ── Summary ───────────────────────────────────────────────────────────────────

Write-Host ""
Write-Host "Service management commands:" -ForegroundColor Cyan
Write-Host "  nssm status $ServiceName       # Check status" -ForegroundColor White
Write-Host "  nssm stop $ServiceName         # Stop" -ForegroundColor White
Write-Host "  nssm start $ServiceName        # Start" -ForegroundColor White
Write-Host "  nssm restart $ServiceName      # Restart" -ForegroundColor White
Write-Host "  nssm edit $ServiceName          # Edit settings (GUI)" -ForegroundColor White
Write-Host ""
Write-Host "Log files:" -ForegroundColor Cyan
Write-Host "  $logDir\stdout.log" -ForegroundColor White
Write-Host "  $logDir\stderr.log" -ForegroundColor White
Write-Host ""
Write-Host "To uninstall:" -ForegroundColor Cyan
Write-Host "  .\scripts\install_service.ps1 -Uninstall" -ForegroundColor White

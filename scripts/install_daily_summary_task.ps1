#Requires -Version 5.1
<#
.SYNOPSIS
    Register the R5 M4 daily summary scheduled task on a VPS.
.DESCRIPTION
    Installs (or refreshes) a Windows Scheduled Task named
    ``AI-SMC-Daily-Summary`` that runs ``scripts/send_daily_summary.py``
    at 00:05 UTC every day.  The 5-minute delay after midnight lets
    structured.jsonl rotate (midnight rollover) and all end-of-day closures
    settle before we scan the day's data.
.NOTES
    Idempotent — re-running this script replaces the existing task.
    Requires SMC_TELEGRAM_BOT_TOKEN + SMC_TELEGRAM_CHAT_ID env vars on the
    VPS for the Telegram push to actually fire; otherwise the alert still
    logs to ``logs/structured.jsonl`` as a ``[CRIT]`` event.
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\install_daily_summary_task.ps1
#>

param(
    [string]$InstallDir = "C:\AI-SMC",
    [string]$TaskName = "AI-SMC-Daily-Summary",
    # Default run time is 00:05 UTC. On a VPS that reports UTC this matches
    # directly; on a local-time VPS, operator should override.
    [string]$StartTimeUtc = "00:05"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$python = Join-Path $InstallDir ".venv\Scripts\python.exe"
$script = Join-Path $InstallDir "scripts\send_daily_summary.py"

if (-not (Test-Path $python)) {
    throw "Python venv not found at $python — run bootstrap_vps.ps1 first."
}
if (-not (Test-Path $script)) {
    throw "send_daily_summary.py not found at $script — check deployment."
}

$action = "`"$python`" `"$script`""

# Build the schtasks command.  /sc DAILY + /st for daily cadence.
$cmd = @(
    "/create",
    "/tn", "`"$TaskName`"",
    "/tr", "`"$action`"",
    "/sc", "DAILY",
    "/st", $StartTimeUtc,
    "/ru", "SYSTEM",
    "/rl", "HIGHEST",
    "/f"  # force — overwrite if exists
) -join " "

Write-Host "[install_daily_summary_task] Registering $TaskName -> $action at $StartTimeUtc"
$output = & schtasks.exe @($cmd.Split(" "))
$exitCode = $LASTEXITCODE
$output | Write-Host
if ($exitCode -ne 0) {
    throw "schtasks failed with exit code $exitCode"
}
Write-Host "[install_daily_summary_task] Done. Task will run daily at $StartTimeUtc UTC."
Write-Host "[install_daily_summary_task] Verify: schtasks /query /tn `"$TaskName`""

#Requires -Version 5.1
<#
.SYNOPSIS
    AI-SMC watchdog — restart live_demo.py when its heartbeat goes stale.
.DESCRIPTION
    Round 5 T1 F4. Run every 5 minutes via Task Scheduler. Reads the
    ``data/live_state.json`` timestamp — if it is missing, unparseable, or
    older than 5 * M15 cycles (20 min), we assume live_demo has hung or
    crashed in a way PID-file self-heal doesn't catch (e.g. Python-level
    deadlock, MT5 socket wedge). The script then kills any matching python
    processes and re-launches via start_live.bat.
.NOTES
    Exits 0 on healthy OR successful restart.
    Exits 1 on failure to restart (caller: Task Scheduler will log the exit
    code; investigate logs\watchdog.log).
.EXAMPLE
    schtasks /create /tn "AI-SMC-Watchdog" /tr "powershell -File C:\AI-SMC\scripts\watchdog_heartbeat.ps1" /sc minute /mo 5 /ru SYSTEM /rl HIGHEST /f
#>

param(
    [string]$InstallDir = "C:\AI-SMC",
    [int]$StaleMinutes = 20,
    [ValidateSet("XAUUSD", "BTCUSD")]
    [string]$Symbol = "XAUUSD"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Continue"

# Round 5 T2 (BTC): per-symbol paths + per-symbol start script so XAU and BTC
# watchdogs operate independently (each schtasks task passes -Symbol).
$statePath    = Join-Path $InstallDir ("data\{0}\live_state.json" -f $Symbol)
$pidPath      = Join-Path $InstallDir ("data\{0}\live_demo.pid" -f $Symbol)
$logDir       = Join-Path $InstallDir "logs"
$logPath      = Join-Path $logDir ("watchdog_{0}.log" -f $Symbol.ToLower())
if ($Symbol -eq "XAUUSD") {
    $startScript = Join-Path $InstallDir "scripts\start_live.bat"
} else {
    $startScript = Join-Path $InstallDir ("scripts\start_live_{0}.bat" -f $Symbol.ToLower())
}

if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
}

function Write-Log {
    param([string]$Message)
    $stamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    $line  = "$stamp $Message"
    Add-Content -Path $logPath -Value $line -Encoding UTF8
    Write-Host $line
}

function Test-HeartbeatFresh {
    if (-not (Test-Path $statePath)) {
        Write-Log "STALE: live_state.json missing at $statePath"
        return $false
    }
    try {
        $state = Get-Content -Path $statePath -Raw -Encoding UTF8 | ConvertFrom-Json
    } catch {
        Write-Log "STALE: cannot parse live_state.json ($_)"
        return $false
    }
    if (-not $state.timestamp) {
        Write-Log "STALE: live_state.json has no timestamp field"
        return $false
    }
    try {
        $ts = [DateTime]::Parse($state.timestamp).ToUniversalTime()
    } catch {
        Write-Log "STALE: cannot parse timestamp '$($state.timestamp)'"
        return $false
    }
    $ageMinutes = ((Get-Date).ToUniversalTime() - $ts).TotalMinutes
    if ($ageMinutes -gt $StaleMinutes) {
        Write-Log ("STALE: heartbeat age {0:N1}min > threshold {1}min" -f $ageMinutes, $StaleMinutes)
        return $false
    }
    Write-Log ("HEALTHY: heartbeat age {0:N1}min (cycle {1})" -f $ageMinutes, $state.cycle)
    return $true
}

function Stop-LiveDemo {
    # Kill by PID file first (cleaner), then sweep any stray python processes
    # running live_demo.py as a belt-and-braces fallback.
    if (Test-Path $pidPath) {
        $pidRaw = (Get-Content -Path $pidPath -Raw).Trim()
        $livePid = 0
        if ([int]::TryParse($pidRaw, [ref]$livePid) -and $livePid -gt 0) {
            try {
                Stop-Process -Id $livePid -Force -ErrorAction Stop
                Write-Log "KILL: terminated PID $livePid from pid file"
            } catch {
                Write-Log "KILL: PID $livePid not found or already dead"
            }
        }
        Remove-Item -Path $pidPath -Force -ErrorAction SilentlyContinue
    }
    # Sweep stray instances for THIS symbol only (matching --symbol arg).
    # XAU default has no --symbol flag, so match any live_demo.py without --symbol BTCUSD.
    $symbolPattern = if ($Symbol -eq "XAUUSD") { "(?!.*--symbol BTCUSD)" } else { "--symbol $Symbol" }
    $stray = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
             Where-Object { $_.CommandLine -and $_.CommandLine -match "live_demo\.py" -and $_.CommandLine -match $symbolPattern }
    foreach ($proc in $stray) {
        try {
            Stop-Process -Id $proc.ProcessId -Force -ErrorAction Stop
            Write-Log "KILL: swept stray live_demo PID $($proc.ProcessId)"
        } catch {
            Write-Log "KILL: could not terminate PID $($proc.ProcessId) — $_"
        }
    }
}

function Start-LiveDemo {
    if (-not (Test-Path $startScript)) {
        Write-Log "FAIL: start script missing at $startScript"
        return $false
    }
    try {
        Start-Process -FilePath $startScript -WorkingDirectory $InstallDir -WindowStyle Hidden
        Write-Log "RESTART: launched $startScript"
        return $true
    } catch {
        Write-Log "FAIL: could not launch start script — $_"
        return $false
    }
}

# ── Main ──────────────────────────────────────────────────────────────────────

Write-Log "watchdog tick"

if (Test-HeartbeatFresh) {
    exit 0
}

Write-Log "recovery: killing any running live_demo and restarting"
Stop-LiveDemo
Start-Sleep -Seconds 3
if (Start-LiveDemo) {
    exit 0
} else {
    exit 1
}

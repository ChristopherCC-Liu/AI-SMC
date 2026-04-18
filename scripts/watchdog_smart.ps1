#Requires -Version 5.1
<#
.SYNOPSIS
    AI-SMC smart watchdog — three-axis health check with grace window.
.DESCRIPTION
    Round 5 T5. Replaces watchdog_heartbeat.ps1.  Three axes:

      Axis 1 — process alive (Win32_Process scan)
      Axis 2 — PID file freshness (presence check)
      Axis 3 — live_state.json age (timestamp parse)

    Grace window: if the process started < GraceMinutes ago, skip axes 2–3.
    This prevents cold-start false kills where the state file hasn't been
    written yet (backfill + first AI analysis can take several minutes).

    StrategyServer mode (-Symbol StrategyServer):
      HTTP GET http://127.0.0.1:8080/healthz — body must contain "ok":true.
      On failure: kill uvicorn process + exit 1 → Task Scheduler restarts.
      On success: exit 0.

    On DEAD: exit 1 → Task Scheduler RestartOnFailure triggers restart.
    On HEALTHY or GRACE: exit 0.
    On STALE (alive but frozen): kill + exit 1 → schtasks restart.

.NOTES
    Does NOT call start_live.bat directly.  Task Scheduler's RestartOnFailure
    policy handles re-launch — removing the double-launcher that caused
    death-loops in watchdog_heartbeat.ps1.
.EXAMPLE
    powershell -File C:\AI-SMC\scripts\watchdog_smart.ps1 -Symbol XAUUSD
    powershell -File C:\AI-SMC\scripts\watchdog_smart.ps1 -Symbol BTCUSD
    powershell -File C:\AI-SMC\scripts\watchdog_smart.ps1 -Symbol StrategyServer
#>

param(
    [string]$InstallDir    = "C:\AI-SMC",
    [int]$StaleMinutes     = 20,
    [int]$GraceMinutes     = 10,
    [ValidateSet("XAUUSD", "BTCUSD", "StrategyServer", "DashboardWeb")]
    [string]$Symbol        = "XAUUSD"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Continue"

# ── Paths ─────────────────────────────────────────────────────────────────────

$dataRoot = Join-Path $InstallDir "data\$Symbol"
$pidPath  = Join-Path $dataRoot  "live_demo.pid"
$statePath = Join-Path $dataRoot "live_state.json"
$logDir   = Join-Path $InstallDir "logs"
$logPath  = Join-Path $logDir ("watchdog_{0}_smart.log" -f $Symbol.ToLower())

if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
}

# ── Logging ───────────────────────────────────────────────────────────────────

function Write-Log {
    param([string]$msg)
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line  = "$stamp [$Symbol] $msg"
    Add-Content -Path $logPath -Value $line -Encoding UTF8
    Write-Host $line
}

# ── StrategyServer health check (early return) ────────────────────────────────
if ($Symbol -eq "StrategyServer") {
    Write-Log "watchdog_smart tick"
    Write-Log "StrategyServer mode: checking http://127.0.0.1:8080/healthz"
    $healthy = $false
    try {
        $resp = Invoke-WebRequest -Uri "http://127.0.0.1:8080/healthz" `
                    -UseBasicParsing -TimeoutSec 5 -ErrorAction Stop
        if ($resp.StatusCode -eq 200 -and $resp.Content -match '"ok"\s*:\s*true') {
            $healthy = $true
        } else {
            Write-Log "StrategyServer UNHEALTHY: StatusCode=$($resp.StatusCode) body=$($resp.Content)"
        }
    } catch {
        Write-Log "StrategyServer UNREACHABLE: $_"
    }

    if ($healthy) {
        Write-Log "StrategyServer HEALTHY: /healthz ok"
        exit 0
    }

    # Not healthy — kill uvicorn so Task Scheduler RestartOnFailure can revive it
    Write-Log "StrategyServer DEAD/FROZEN: killing uvicorn"
    try {
        $uvicorn = Get-CimInstance Win32_Process -ErrorAction Stop |
                   Where-Object { $_.CommandLine -and $_.CommandLine -match 'uvicorn.*strategy_server' }
        foreach ($p in $uvicorn) {
            try {
                Stop-Process -Id $p.ProcessId -Force -ErrorAction Stop
                Write-Log "  killed uvicorn PID $($p.ProcessId)"
            } catch {
                Write-Log "  could not kill PID $($p.ProcessId) — $_"
            }
        }
    } catch {
        Write-Log "  WMI query for uvicorn failed — $_"
    }
    Write-Log "StrategyServer: exiting 1 → Task Scheduler RestartOnFailure will restart"
    exit 1
}

# ── DashboardWeb health check (early return) ─────────────────────────────────
# Round 3 Sprint 2: mirror of StrategyServer probe for dashboard_server :8765.
# Catches uvicorn freeze cases that don't trigger RestartOnFailure.
if ($Symbol -eq "DashboardWeb") {
    Write-Log "watchdog_smart tick"
    Write-Log "DashboardWeb mode: checking http://127.0.0.1:8765/healthz"
    $healthy = $false
    try {
        $resp = Invoke-WebRequest -Uri "http://127.0.0.1:8765/healthz" `
                    -UseBasicParsing -TimeoutSec 5 -ErrorAction Stop
        if ($resp.StatusCode -eq 200 -and $resp.Content -match '"ok"\s*:\s*true') {
            $healthy = $true
        } else {
            Write-Log "DashboardWeb UNHEALTHY: StatusCode=$($resp.StatusCode) body=$($resp.Content)"
        }
    } catch {
        Write-Log "DashboardWeb UNREACHABLE: $_"
    }

    if ($healthy) {
        Write-Log "DashboardWeb HEALTHY: /healthz ok"
        exit 0
    }

    # Not healthy — kill python backing dashboard_server.py so Task Scheduler
    # RestartOnFailure can revive it.  Note: dashboard_server.py is run via
    # ``python scripts\dashboard_server.py`` (NOT ``uvicorn`` CLI), so match
    # the filename in the command line rather than the ``uvicorn`` binary.
    Write-Log "DashboardWeb DEAD/FROZEN: killing dashboard_server python"
    try {
        $procs = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction Stop |
                 Where-Object { $_.CommandLine -and $_.CommandLine -match 'dashboard_server' }
        foreach ($p in $procs) {
            try {
                Stop-Process -Id $p.ProcessId -Force -ErrorAction Stop
                Write-Log "  killed dashboard_server PID $($p.ProcessId)"
            } catch {
                Write-Log "  could not kill PID $($p.ProcessId) — $_"
            }
        }
    } catch {
        Write-Log "  WMI query for dashboard_server python failed — $_"
    }
    Write-Log "DashboardWeb: exiting 1 → Task Scheduler RestartOnFailure will restart"
    exit 1
}

# ── Axis 1: process alive ─────────────────────────────────────────────────────
# XAU default: live_demo.py without --symbol BTCUSD
# BTC: live_demo.py with --symbol BTCUSD (or --symbol XAUUSD)

Write-Log "watchdog_smart tick"

$procAlive     = $false
$procStartTime = $null

try {
    $allProcs = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction Stop |
                Where-Object { $_.CommandLine -and $_.CommandLine -match 'live_demo' }

    if ($Symbol -eq "XAUUSD") {
        $procs = $allProcs | Where-Object { $_.CommandLine -notmatch 'BTCUSD' }
    } else {
        $procs = $allProcs | Where-Object { $_.CommandLine -match $Symbol }
    }

    if ($procs) {
        $procAlive     = $true
        $procStartTime = ($procs | Select-Object -First 1).CreationDate
        $allPids       = ($procs | ForEach-Object { $_.ProcessId }) -join ","
        Write-Log "Axis1 ALIVE: PIDs=$allPids  started=$($procStartTime.ToString('HH:mm:ss'))"
    } else {
        Write-Log "Axis1 DEAD: no live_demo process found for $Symbol"
    }
} catch {
    Write-Log "Axis1 ERROR: WMI query failed — $_"
    # Cannot determine process state; be conservative and report healthy
    # to avoid spurious restarts on WMI hiccup.
    exit 0
}

# ── Early exit on DEAD ────────────────────────────────────────────────────────
if (-not $procAlive) {
    Write-Log "DEAD: exiting 1 → Task Scheduler RestartOnFailure will restart"
    exit 1
}

# ── Grace window ──────────────────────────────────────────────────────────────
if ($procStartTime) {
    $ageMins = ((Get-Date) - $procStartTime).TotalMinutes
    if ($ageMins -lt $GraceMinutes) {
        Write-Log ("GRACE: process age {0:F1}min < {1}min grace — skipping state checks" -f $ageMins, $GraceMinutes)
        exit 0
    }
    Write-Log ("Grace passed: process age {0:F1}min" -f $ageMins)
}

# ── Axis 2: PID file presence ─────────────────────────────────────────────────
if (-not (Test-Path $pidPath)) {
    Write-Log "Axis2 WARN: PID file missing at $pidPath (process alive but no PID file)"
    # Not fatal on its own; continue to Axis 3
} else {
    Write-Log "Axis2 OK: PID file present"
}

# ── Axis 3: live_state.json age ───────────────────────────────────────────────
$stateFresh = $false

if (-not (Test-Path $statePath)) {
    Write-Log "Axis3 STALE: live_state.json missing at $statePath"
} else {
    try {
        $raw   = Get-Content $statePath -Raw -Encoding UTF8
        $state = $raw | ConvertFrom-Json

        if (-not $state.timestamp) {
            Write-Log "Axis3 STALE: live_state.json has no timestamp field"
        } else {
            $ts      = [DateTime]::Parse($state.timestamp).ToUniversalTime()
            $ageMin  = ((Get-Date).ToUniversalTime() - $ts).TotalMinutes
            $cycleInfo = if ($state.cycle) { " cycle=$($state.cycle)" } else { "" }

            Write-Log ("Axis3: state age {0:F1}min (threshold {1}min){2}" -f $ageMin, $StaleMinutes, $cycleInfo)

            if ($ageMin -lt $StaleMinutes) {
                $stateFresh = $true
            }
        }
    } catch {
        Write-Log "Axis3 STALE: failed to parse live_state.json — $_"
    }
}

# ── Decision ──────────────────────────────────────────────────────────────────
if ($stateFresh) {
    Write-Log "HEALTHY: process alive + state fresh"
    exit 0
}

# Process is alive but state is stale → frozen/deadlocked process
Write-Log "STALE: process alive but live_state.json age exceeds threshold → killing"

try {
    $killProcs = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction Stop |
                 Where-Object { $_.CommandLine -and $_.CommandLine -match 'live_demo' }

    if ($Symbol -eq "XAUUSD") {
        $killProcs = $killProcs | Where-Object { $_.CommandLine -notmatch 'BTCUSD' }
    } else {
        $killProcs = $killProcs | Where-Object { $_.CommandLine -match $Symbol }
    }

    foreach ($p in $killProcs) {
        try {
            Stop-Process -Id $p.ProcessId -Force -ErrorAction Stop
            Write-Log "  killed PID $($p.ProcessId)"
        } catch {
            Write-Log "  could not kill PID $($p.ProcessId) — $_"
        }
    }
} catch {
    Write-Log "  WMI query for kill failed — $_"
}

Remove-Item $pidPath -Force -ErrorAction SilentlyContinue
Write-Log "STALE: exiting 1 → Task Scheduler RestartOnFailure will restart"
exit 1

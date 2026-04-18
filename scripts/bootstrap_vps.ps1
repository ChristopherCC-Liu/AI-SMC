#Requires -Version 5.1
<#
.SYNOPSIS
    AI-SMC one-time VPS bootstrap — registers all 6 Task Scheduler tasks.
.DESCRIPTION
    Run ONCE after a fresh VPS setup (or to re-register after changes).
    After this script completes, the Lead never needs to manually SSH for
    normal ops: all services auto-start on boot, auto-restart on failure,
    and watchdogs fire every 5 minutes.

    Tasks registered:
      \AI-SMC-Live              — XAUUSD live_demo signal loop
      \AI-SMC-Live-BTC          — BTCUSD live_demo signal loop
      \AI-SMC-StrategyServer    — FastAPI signal server (port 8080)
      \AI-SMC-DashboardWeb      — Dashboard web server (port 8765)
      \AI-SMC-Watchdog          — XAUUSD three-axis watchdog (every 5 min)
      \AI-SMC-Watchdog-BTC      — BTCUSD three-axis watchdog (every 5 min)

    All service tasks: BootTrigger + RestartOnFailure (PT5M, 10 retries)
                       + ExecutionTimeLimit PT2H + StartWhenAvailable
    Watchdog tasks:    RepetitionInterval PT5M (no RestartOnFailure needed;
                       next 5-min fire is the natural retry)

.NOTES
    Must run as Administrator (Task Scheduler requires elevation).
    InstallDir default: C:\AI-SMC
    Python venv must already exist: C:\AI-SMC\.venv
.EXAMPLE
    # Elevated PowerShell on VPS:
    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
    powershell -ExecutionPolicy Bypass -File C:\AI-SMC\scripts\bootstrap_vps.ps1
#>

param(
    [string]$InstallDir   = "C:\AI-SMC",
    [string]$RunAs        = "SYSTEM"   # or "Administrator" with -Password
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$msg)
    Write-Host "==> $msg" -ForegroundColor Cyan
}

function Unregister-Safe {
    param([string]$TaskName)
    try {
        Unregister-ScheduledTask -TaskName $TaskName -TaskPath "\" -Confirm:$false -ErrorAction Stop
        Write-Host "  Removed existing task: $TaskName"
    } catch {
        # Not registered — fine
    }
}

# ── XML helpers ───────────────────────────────────────────────────────────────

# Service task XML: BootTrigger + RestartOnFailure + 2h limit
function Get-ServiceTaskXml {
    param(
        [string]$Command,
        [string]$WorkingDir,
        [string]$Arguments = ""
    )
    return @"
<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.4" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>AI-SMC auto-managed service</Description>
  </RegistrationInfo>
  <Triggers>
    <BootTrigger>
      <Enabled>true</Enabled>
      <Delay>PT15S</Delay>
    </BootTrigger>
  </Triggers>
  <Principals>
    <Principal id="Author">
      <UserId>SYSTEM</UserId>
      <RunLevel>HighestAvailable</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>true</AllowHardTerminate>
    <StartWhenAvailable>true</StartWhenAvailable>
    <!-- Round 5 T5 audit r1 fix (P0-1): PT2H on a long-running daemon caused
         Task Scheduler to hard-kill the process every 2h, losing in-memory
         cycle/quota/circuit state. PT0S = no limit. RestartOnFailure still
         catches genuine crashes. -->
    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>
    <RestartOnFailure>
      <Interval>PT5M</Interval>
      <Count>10</Count>
    </RestartOnFailure>
    <Enabled>true</Enabled>
  </Settings>
  <Actions>
    <Exec>
      <Command>$Command</Command>
      <Arguments>$Arguments</Arguments>
      <WorkingDirectory>$WorkingDir</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
"@
}

# Watchdog task XML: repetition every 5 min for 18 hours (covers a full day), no RestartOnFailure
function Get-WatchdogTaskXml {
    param(
        [string]$Symbol
    )
    $psExe  = "powershell.exe"
    $args   = "-NonInteractive -NoProfile -ExecutionPolicy Bypass -File $InstallDir\scripts\watchdog_smart.ps1 -Symbol $Symbol"
    return @"
<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.4" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>AI-SMC $Symbol watchdog (three-axis health check)</Description>
  </RegistrationInfo>
  <Triggers>
    <TimeTrigger>
      <Repetition>
        <Interval>PT5M</Interval>
        <Duration>P1D</Duration>
        <StopAtDurationEnd>false</StopAtDurationEnd>
      </Repetition>
      <StartBoundary>2026-01-01T00:00:00</StartBoundary>
      <Enabled>true</Enabled>
    </TimeTrigger>
    <BootTrigger>
      <Enabled>true</Enabled>
      <Delay>PT2M</Delay>
    </BootTrigger>
  </Triggers>
  <Principals>
    <Principal id="Author">
      <UserId>SYSTEM</UserId>
      <RunLevel>HighestAvailable</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>true</AllowHardTerminate>
    <StartWhenAvailable>true</StartWhenAvailable>
    <ExecutionTimeLimit>PT5M</ExecutionTimeLimit>
    <Enabled>true</Enabled>
  </Settings>
  <Actions>
    <Exec>
      <Command>$psExe</Command>
      <Arguments>$args</Arguments>
      <WorkingDirectory>$InstallDir</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
"@
}

# ── Task definitions ──────────────────────────────────────────────────────────

$tasks = @(
    @{
        Name    = "AI-SMC-Live"
        Xml     = Get-ServiceTaskXml `
                      -Command  "$InstallDir\scripts\start_live_signal_only.bat" `
                      -WorkingDir $InstallDir
    },
    @{
        Name    = "AI-SMC-Live-BTC"
        Xml     = Get-ServiceTaskXml `
                      -Command  "$InstallDir\scripts\start_live_btc_signal_only.bat" `
                      -WorkingDir $InstallDir
    },
    @{
        Name    = "AI-SMC-StrategyServer"
        Xml     = Get-ServiceTaskXml `
                      -Command  "$InstallDir\scripts\start_strategy_server.bat" `
                      -WorkingDir $InstallDir
    },
    @{
        Name    = "AI-SMC-DashboardWeb"
        Xml     = Get-ServiceTaskXml `
                      -Command  "$InstallDir\scripts\start_dashboard_web.bat" `
                      -WorkingDir $InstallDir
    },
    @{
        Name    = "AI-SMC-Watchdog"
        Xml     = Get-WatchdogTaskXml -Symbol "XAUUSD"
    },
    @{
        Name    = "AI-SMC-Watchdog-BTC"
        Xml     = Get-WatchdogTaskXml -Symbol "BTCUSD"
    }
)

# ── Register ──────────────────────────────────────────────────────────────────

Write-Step "Registering AI-SMC Task Scheduler tasks (InstallDir=$InstallDir)"

foreach ($task in $tasks) {
    $name = $task.Name
    Write-Step "[$name]"

    Unregister-Safe -TaskName $name

    try {
        Register-ScheduledTask `
            -TaskName $name `
            -TaskPath "\" `
            -Xml $task.Xml `
            -Force | Out-Null
        Write-Host "  Registered: $name" -ForegroundColor Green
    } catch {
        Write-Host "  FAILED to register $name — $_" -ForegroundColor Red
    }
}

# ── Status summary ────────────────────────────────────────────────────────────

Write-Step "Task status summary:"
$taskNames = $tasks | ForEach-Object { $_.Name }
foreach ($name in $taskNames) {
    try {
        $t = Get-ScheduledTask -TaskName $name -TaskPath "\" -ErrorAction Stop
        $info = Get-ScheduledTaskInfo -TaskName $name -TaskPath "\" -ErrorAction SilentlyContinue
        $lastRun  = if ($info.LastRunTime) { $info.LastRunTime.ToString("yyyy-MM-dd HH:mm:ss") } else { "never" }
        $nextRun  = if ($info.NextRunTime) { $info.NextRunTime.ToString("yyyy-MM-dd HH:mm:ss") } else { "—" }
        Write-Host ("  {0,-30} state={1}  last={2}  next={3}" -f `
            $name, $t.State, $lastRun, $nextRun) -ForegroundColor Green
    } catch {
        Write-Host ("  {0,-30} NOT FOUND" -f $name) -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "Bootstrap complete. Reboot the VPS to trigger BootTriggers." -ForegroundColor Yellow
Write-Host "Or start tasks manually:"
foreach ($name in $taskNames) {
    Write-Host "  Start-ScheduledTask -TaskName '$name'"
}

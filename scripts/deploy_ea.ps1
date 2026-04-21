#Requires -Version 5.1
<#
.SYNOPSIS
    AI-SMC EA (AISMCReceiver.mq5) one-shot VPS re-deployment script.
.DESCRIPTION
    Run this on the TMGM VPS after any change to mql5/:
      1. Copy mql5\AISMCReceiver.mq5   → MT5 Experts folder
      2. Copy mql5\include\*.mqh       → MT5 Include folder (R7+: panel
         header + any future include dependencies)
      3. Compile via MetaEditor64.exe with a blocking Start-Process -Wait
      4. Verify the .ex5 artifact exists and is > 50 KB
      5. Extract `#property version` from the .mq5 source
      6. Send a Telegram notification via scripts/notify_ea_deploy.py
         so the user knows to F7 / re-attach the EA on XAUUSD + BTCUSD charts.

    Returns exit code 0 on success, non-zero on any hard failure.

.NOTES
    VPS context (2026-04-20):
      - Repo checkout:    C:\AI-SMC
      - MT5 terminal id:  7643C0B96C7AD5841307C9E1EB0B9252
      - Experts folder:   C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\<terminal_id>\MQL5\Experts\
      - Include folder:   ...\MQL5\Include\  (R7: required for aismc_panel.mqh)
      - MetaEditor path:  auto-detected via env + common install roots.

    Caller discovery (learned the hard way today):
      Plain pipe / & call returns before .ex5 is fully written. We MUST use
      ``Start-Process -Wait -NoNewWindow`` so the script blocks until
      MetaEditor actually exits, and only then inspect the artifact.

.EXAMPLE
    # On VPS PowerShell (run from anywhere):
    PS> & C:\AI-SMC\scripts\deploy_ea.ps1

    # Override locations for testing:
    PS> & C:\AI-SMC\scripts\deploy_ea.ps1 `
          -RepoRoot "C:\AI-SMC" `
          -TerminalId "7643C0B96C7AD5841307C9E1EB0B9252"
#>
[CmdletBinding()]
param(
    [string]$RepoRoot      = "C:\AI-SMC",
    [string]$TerminalId    = "7643C0B96C7AD5841307C9E1EB0B9252",
    [string]$MetaEditor    = "",             # empty -> auto-detect
    [int]   $MinEx5Bytes   = 50 * 1024,      # 50 KB — catch empty / stub compiles
    [int]   $CompileTimeoutSec = 120,
    [switch]$SkipTelegram                    # for dry-run testing
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ── Helpers ───────────────────────────────────────────────────────────────────

function Write-Step {
    param([string]$Message, [string]$Color = "Cyan")
    Write-Host ""
    Write-Host ("== " + $Message + " ==") -ForegroundColor $Color
}

function Fail {
    param([string]$Message, [int]$Code = 1)
    Write-Host ("FAIL: " + $Message) -ForegroundColor Red
    exit $Code
}

function Find-MetaEditor {
    param([string]$Explicit)
    if ($Explicit -and (Test-Path $Explicit)) {
        return (Resolve-Path $Explicit).Path
    }
    # 1) MT5 install root from env
    $envRoot = $env:MT5_TERMINAL_DIR
    if ($envRoot) {
        $candidate = Join-Path $envRoot "metaeditor64.exe"
        if (Test-Path $candidate) { return (Resolve-Path $candidate).Path }
    }
    # 2) Common installs (TMGM / Generic)
    $candidates = @(
        "C:\Program Files\TMGM MT5 Terminal\metaeditor64.exe",
        "C:\Program Files\MetaTrader 5\metaeditor64.exe",
        "C:\Program Files (x86)\MetaTrader 5\metaeditor64.exe"
    )
    foreach ($c in $candidates) {
        if (Test-Path $c) { return $c }
    }
    # 3) Fuzzy search under Program Files
    $fuzzy = Get-ChildItem -Path "C:\Program Files","C:\Program Files (x86)" `
        -Filter "metaeditor64.exe" -Recurse -ErrorAction SilentlyContinue `
        -ErrorVariable +searchErr | Select-Object -First 1
    if ($fuzzy) { return $fuzzy.FullName }
    return $null
}

function Extract-EaVersion {
    param([string]$Mq5Path)
    # #property version   "2.00"  -> "2.00"
    $lines = Get-Content $Mq5Path -TotalCount 80
    foreach ($line in $lines) {
        if ($line -match '^\s*#property\s+version\s+"([^"]+)"') {
            return $Matches[1]
        }
    }
    return "unknown"
}

# ── 1. Resolve paths ──────────────────────────────────────────────────────────

Write-Step "deploy_ea: resolving paths"

$sourceMq5 = Join-Path $RepoRoot "mql5\AISMCReceiver.mq5"
if (-not (Test-Path $sourceMq5)) {
    Fail ("Source .mq5 not found: " + $sourceMq5)
}

# R7: all .mqh dependencies live under mql5\include\ and must be mirrored
# into MQL5\Include\ before MetaEditor tries to resolve #include <...>.
$includeSrcDir = Join-Path $RepoRoot "mql5\include"

$mqlRoot       = Join-Path $env:APPDATA ("MetaQuotes\Terminal\" + $TerminalId + "\MQL5")
$expertsDir    = Join-Path $mqlRoot "Experts"
$includeDstDir = Join-Path $mqlRoot "Include"

if (-not (Test-Path $expertsDir)) {
    Write-Host ("  Creating Experts dir: " + $expertsDir)
    New-Item -ItemType Directory -Path $expertsDir -Force | Out-Null
}
if (-not (Test-Path $includeDstDir)) {
    Write-Host ("  Creating Include dir: " + $includeDstDir)
    New-Item -ItemType Directory -Path $includeDstDir -Force | Out-Null
}

$destMq5 = Join-Path $expertsDir "AISMCReceiver.mq5"
$destEx5 = Join-Path $expertsDir "AISMCReceiver.ex5"

Write-Host ("  Source  : " + $sourceMq5)
Write-Host ("  Dest    : " + $destMq5)
Write-Host ("  Include : " + $includeSrcDir + " -> " + $includeDstDir)

$eaVersion = Extract-EaVersion -Mq5Path $sourceMq5
Write-Host ("  Version: " + $eaVersion)

# ── 2. Copy .mq5 into Experts folder ──────────────────────────────────────────

Write-Step "deploy_ea: copying source"

try {
    Copy-Item -Path $sourceMq5 -Destination $destMq5 -Force
} catch {
    Fail ("Copy failed: " + $_.Exception.Message)
}
Write-Host ("  Copied {0:N0} bytes" -f (Get-Item $destMq5).Length)

# Remove stale .ex5 before compile so the existence check below is a true
# post-compile artifact (not a ghost from a previous version).
if (Test-Path $destEx5) {
    Remove-Item $destEx5 -Force
    Write-Host "  Removed stale .ex5"
}

# ── 2b. Sync include files ────────────────────────────────────────────────────
# R7: AISMCReceiver.mq5 now depends on `#include <aismc_panel.mqh>` (and
# potentially future .mqh siblings under mql5\include\).  We mirror the
# whole folder — globbing by *.mqh makes new additions auto-deploy with
# zero script edits.  Non-fatal if the repo ships no includes (e.g. on a
# rollback to a pre-R6 EA).

Write-Step "deploy_ea: syncing includes"

if (-not (Test-Path $includeSrcDir)) {
    Write-Host ("  No include dir in repo at " + $includeSrcDir + " — skipping (pre-R6 EA?)") -ForegroundColor Yellow
} else {
    $includeFiles = Get-ChildItem -Path $includeSrcDir -Filter "*.mqh" -File `
        -ErrorAction SilentlyContinue
    if (-not $includeFiles -or $includeFiles.Count -eq 0) {
        Write-Host "  No .mqh files found in include dir — skipping"
    } else {
        foreach ($f in $includeFiles) {
            $dst = Join-Path $includeDstDir $f.Name
            try {
                Copy-Item -Path $f.FullName -Destination $dst -Force
                Write-Host ("  Copied {0} ({1:N0} bytes)" -f $f.Name, $f.Length)
            } catch {
                Fail ("Include copy failed for " + $f.Name + ": " + $_.Exception.Message)
            }
        }
    }
}

# ── 3. Locate MetaEditor64.exe ────────────────────────────────────────────────

Write-Step "deploy_ea: locating MetaEditor64"

$editorPath = Find-MetaEditor -Explicit $MetaEditor
if (-not $editorPath) {
    Fail "MetaEditor64.exe not found (set -MetaEditor or `$env:MT5_TERMINAL_DIR)"
}
Write-Host ("  Editor: " + $editorPath)

# ── 4. Compile (blocking) ─────────────────────────────────────────────────────

Write-Step "deploy_ea: compiling (blocking, up to $CompileTimeoutSec s)"

# CRITICAL: `Start-Process -Wait -NoNewWindow` is the only reliable way to
# block until MetaEditor actually exits. Plain `&` returns before the
# compiler finishes writing .ex5 (observed 2026-04-20 18:00 on TMGM VPS).
$compileLog = Join-Path $env:TEMP ("ai_smc_ea_compile_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".log")
$compileArgs = @(
    "/compile:$destMq5",
    "/log:$compileLog"
)

$sw = [System.Diagnostics.Stopwatch]::StartNew()
try {
    $proc = Start-Process -FilePath $editorPath `
                          -ArgumentList $compileArgs `
                          -Wait -NoNewWindow `
                          -PassThru `
                          -ErrorAction Stop
} catch {
    Fail ("MetaEditor launch failed: " + $_.Exception.Message)
}
$sw.Stop()

Write-Host ("  Compile finished in {0:N1} s (exit {1})" -f $sw.Elapsed.TotalSeconds, $proc.ExitCode)

if ($sw.Elapsed.TotalSeconds -gt $CompileTimeoutSec) {
    Fail ("Compile exceeded timeout " + $CompileTimeoutSec + " s")
}

# MetaEditor's /compile returns non-zero on errors, 0 on warnings-only success.
if ($proc.ExitCode -ne 0) {
    Write-Host "  Compile exit code non-zero, dumping log tail:" -ForegroundColor Yellow
    if (Test-Path $compileLog) {
        Get-Content $compileLog -Tail 20 | ForEach-Object { Write-Host "    $_" }
    }
    Fail ("MetaEditor compile returned " + $proc.ExitCode)
}

# ── 5. Verify .ex5 artifact ──────────────────────────────────────────────────

Write-Step "deploy_ea: verifying .ex5"

if (-not (Test-Path $destEx5)) {
    if (Test-Path $compileLog) {
        Write-Host "  Compile log tail:" -ForegroundColor Yellow
        Get-Content $compileLog -Tail 20 | ForEach-Object { Write-Host "    $_" }
    }
    Fail (".ex5 not produced at " + $destEx5)
}

$ex5Size = (Get-Item $destEx5).Length
Write-Host ("  .ex5 size: {0:N0} bytes" -f $ex5Size)

if ($ex5Size -lt $MinEx5Bytes) {
    Fail (".ex5 suspiciously small: $ex5Size < $MinEx5Bytes bytes (compile likely produced a stub)")
}

# ── 6. Telegram notification ──────────────────────────────────────────────────

Write-Step "deploy_ea: sending Telegram notification"

if ($SkipTelegram) {
    Write-Host "  Skipped (--SkipTelegram flag)"
} else {
    $notifyScript = Join-Path $RepoRoot "scripts\notify_ea_deploy.py"
    if (-not (Test-Path $notifyScript)) {
        Write-Host ("  WARN notify_ea_deploy.py not found at " + $notifyScript + " — skipping") -ForegroundColor Yellow
    } else {
        $pyCandidates = @("python", "python.exe", "py")
        $pythonExe = $null
        foreach ($c in $pyCandidates) {
            $cmd = Get-Command $c -ErrorAction SilentlyContinue
            if ($cmd) { $pythonExe = $cmd.Source; break }
        }
        if (-not $pythonExe) {
            Write-Host "  WARN python not on PATH — Telegram notify skipped" -ForegroundColor Yellow
        } else {
            $ex5SizeKb = [math]::Round($ex5Size / 1024, 1)
            $notifyArgs = @(
                $notifyScript,
                "--version",      $eaVersion,
                "--ex5-size-kb",  $ex5SizeKb,
                "--compile-sec",  [math]::Round($sw.Elapsed.TotalSeconds, 1)
            )
            try {
                & $pythonExe @notifyArgs 2>&1 | ForEach-Object { Write-Host "    $_" }
                if ($LASTEXITCODE -ne 0) {
                    Write-Host ("  WARN Telegram notifier returned " + $LASTEXITCODE + " (non-fatal)") -ForegroundColor Yellow
                }
            } catch {
                Write-Host ("  WARN Telegram notify threw: " + $_.Exception.Message) -ForegroundColor Yellow
            }
        }
    }
}

# ── Done ──────────────────────────────────────────────────────────────────────

Write-Step "deploy_ea: SUCCESS" "Green"
Write-Host ("  EA v" + $eaVersion + " deployed to " + $destMq5)
Write-Host "  Action needed: in MT5 → detach EA → re-attach to XAUUSD + BTCUSD charts."
exit 0

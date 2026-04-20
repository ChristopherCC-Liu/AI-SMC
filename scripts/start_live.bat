@echo off
REM Round 4.6-N: AI-SMC live trading loop launcher (Windows VPS).
REM Deployed to C:\AI-SMC\start_live.bat and registered as Task Scheduler
REM `\AI-SMC-Live` task with ONSTART trigger only. live_demo.py owns its
REM own next_bar_close loop so Task Scheduler must NOT repeat every 5min.
REM
REM 4.6-L PID file at data\live_demo.pid prevents duplicate instances.

set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
REM Round 4.6-X (USER CATCH): enable real MT5 order_send (demo account safe).
set SMC_MT5_EXECUTE=1
REM Round 4 v5 Option B: explicit override to prevent inheriting user-level
REM SMC_MACRO_ENABLED=true set for treatment leg. Control MUST stay macro OFF
REM to maintain the A/B baseline. Suffix empty = control journal path.
set SMC_MACRO_ENABLED=false
set SMC_JOURNAL_SUFFIX=
REM Round 4 v5: enable 7-agent regime classifier on both legs (shared baseline).
REM Cuts researcher rounds 2->1 to halve Opus calls (bull+bear x1 instead of x2).
set SMC_AI_REGIME_ENABLED=true
set SMC_AI_DEBATE_ROUNDS=1
cd /d C:\AI-SMC
.venv\Scripts\python.exe scripts\live_demo.py >> logs\live_stdout.log 2>> logs\live_stderr.log

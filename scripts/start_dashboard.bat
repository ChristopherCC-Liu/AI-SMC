@echo off
REM Round 4.6-N: AI-SMC dashboard launcher (Windows VPS).
REM Deployed to C:\AI-SMC\start_dashboard.bat and registered as Task Scheduler
REM `\AI-SMC-Dashboard` task with ONSTART trigger only.

set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d C:\AI-SMC
.venv\Scripts\python.exe scripts\dashboard.py >> logs\dashboard_stdout.log 2>> logs\dashboard_stderr.log

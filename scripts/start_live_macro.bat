@echo off
REM Round 4 Alt-B W3: A/B parallel paper trading — treatment leg (macro overlay ON).
REM Deployed to C:\AI-SMC\scripts\start_live_macro.bat and registered as Task Scheduler
REM `\AI-SMC-Live-Macro` task with BootTrigger only.
REM
REM Process B (treatment): SMC_MACRO_ENABLED=true, SMC_JOURNAL_SUFFIX=_macro
REM  → journals go to data\XAUUSD\journal_macro\live_trades.jsonl
REM  → live state at data\XAUUSD\live_state_macro.json
REM
REM Process A (control) is start_live.bat / AI-SMC-Live — no env overrides.
REM After 30 days compare journal/ vs journal_macro/ via scripts\ab_compare.py.

set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set SMC_MACRO_ENABLED=true
set SMC_JOURNAL_SUFFIX=_macro
cd /d C:\AI-SMC
.venv\Scripts\python.exe scripts\live_demo.py >> logs\live_macro_stdout.log 2>> logs\live_macro_stderr.log

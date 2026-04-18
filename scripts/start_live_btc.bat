@echo off
REM Round 5 T2 (BTC): AI-SMC BTCUSD LIVE launcher.
REM Runs alongside start_live.bat (XAUUSD live). Registered as Task Scheduler
REM `\AI-SMC-Live-BTC` task with ONSTART trigger. live_demo.py owns its own
REM next_bar_close loop so Task Scheduler must NOT repeat every 5min.
REM
REM TMGM demo account $1000 — user authorized direct LIVE_EXEC (not paper)
REM since no real capital is at risk. margin_cap (40% of equity) + ConsecLossHalt
REM (per-symbol) + DrawdownGuard backstop are the safety nets.
REM
REM Per-symbol PID file lives at data\BTCUSD\live_demo.pid (separate from XAU).

set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set SMC_MT5_EXECUTE=1
REM SMC_TELEGRAM_BOT_TOKEN and SMC_TELEGRAM_CHAT_ID inherited from Machine env.
cd /d C:\AI-SMC
.venv\Scripts\python.exe scripts\live_demo.py --symbol BTCUSD >> logs\live_btc_stdout.log 2>> logs\live_btc_stderr.log

@echo off
REM Round 5 T2 (BTC): AI-SMC BTCUSD paper-mode launcher.
REM Runs alongside start_live.bat (XAUUSD live). Registered as Task Scheduler
REM `\AI-SMC-Live-BTC` task with ONSTART trigger. live_demo.py owns its own
REM next_bar_close loop so Task Scheduler must NOT repeat every 5min.
REM
REM Week 1 (2026-04-18 to 2026-04-26): --paper flag so no real MT5 order_send
REM fires; journal entries tagged paper=true for data-only observation.
REM Week 2 (2026-04-27+): remove --paper flag to enable LIVE_EXEC on BTC demo.
REM
REM Per-symbol PID file lives at data\BTCUSD\live_demo.pid (separate from XAU).

set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
REM SMC_MT5_EXECUTE intentionally NOT set — --paper overrides it either way
REM and Machine-level env SMC_MT5_EXECUTE=1 would be ignored by --paper flag.
REM SMC_TELEGRAM_BOT_TOKEN and SMC_TELEGRAM_CHAT_ID inherited from Machine env.
cd /d C:\AI-SMC
.venv\Scripts\python.exe scripts\live_demo.py --symbol BTCUSD --paper >> logs\live_btc_stdout.log 2>> logs\live_btc_stderr.log

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
REM Override Machine-level SMC_MT5_EXECUTE=1 so startup alert + journal
REM entries accurately reflect paper mode. --paper CLI flag is the actual
REM order-send gate; this env override is cosmetic but avoids the
REM misleading "mt5_execute=1" value in the system_startup Telegram alert.
set SMC_MT5_EXECUTE=0
REM SMC_TELEGRAM_BOT_TOKEN and SMC_TELEGRAM_CHAT_ID inherited from Machine env.
cd /d C:\AI-SMC
.venv\Scripts\python.exe scripts\live_demo.py --symbol BTCUSD --paper >> logs\live_btc_stdout.log 2>> logs\live_btc_stderr.log

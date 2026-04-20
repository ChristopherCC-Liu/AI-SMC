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
REM Round 4 v5 feature parity with XAUUSD legs.
set SMC_AI_REGIME_ENABLED=true
set SMC_AI_DEBATE_ROUNDS=1
set SMC_MAX_CONCURRENT_PER_SYMBOL=1
set SMC_ANTI_STACK_COOLDOWN_MINUTES=60
set SMC_RANGE_REVERSAL_CONFIRM_ENABLED=true
REM BTC single-leg: macro overlay OFF, no journal suffix.
set SMC_MACRO_ENABLED=false
set SMC_JOURNAL_SUFFIX=
REM SMC_TELEGRAM_BOT_TOKEN and SMC_TELEGRAM_CHAT_ID inherited from Machine env.
cd /d C:\AI-SMC
.venv\Scripts\python.exe scripts\live_demo.py --symbol BTCUSD >> logs\live_btc_stdout.log 2>> logs\live_btc_stderr.log

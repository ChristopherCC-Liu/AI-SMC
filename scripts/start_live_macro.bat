@echo off
REM Round 4 Alt-B W3 / audit-r4 v5 Option B: dual-magic A/B on same TMGM Demo.
REM Deployed to C:\AI-SMC\scripts\start_live_macro.bat and registered as Task Scheduler
REM `\AI-SMC-Live-Macro` task with BootTrigger only.
REM
REM Process B (treatment): SMC_MACRO_ENABLED=true, SMC_JOURNAL_SUFFIX=_macro
REM  → journals go to data\XAUUSD\journal_macro\live_trades.jsonl
REM  → live state at data\XAUUSD\live_state_macro.json
REM  → magic=19760428 (vs control 19760418) routes orders per-leg
REM  → virtual balance 50% of MT5 equity via SMCConfig.virtual_balance_split
REM
REM Process A (control) is start_live.bat / AI-SMC-Live — no env overrides.
REM Both legs write signals into strategy_server /signal array; single EA
REM instance on XAUUSD chart polls the array and OrderSends with each signal's
REM magic.  After 30 days compare journal/ vs journal_macro/ via
REM scripts\ab_compare.py.
REM
REM IMPORTANT — EA re-compile required: mql5\AISMCReceiver.mq5 v2.00 must be
REM compiled in MetaEditor and re-attached to the XAUUSD chart.

set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set SMC_MACRO_ENABLED=true
set SMC_JOURNAL_SUFFIX=_macro
REM Round 4 v5: enable 7-agent regime classifier on both legs (shared baseline).
REM Cuts researcher rounds 2->1 to halve Opus calls (bull+bear x1 instead of x2).
set SMC_AI_REGIME_ENABLED=true
set SMC_AI_DEBATE_ROUNDS=1
REM Round 4 v5 (Tasks #51/52/53): post-2026-04-20 02:46 stacked-SL guardrails.
REM max=1 enforces user's one-position-per-symbol hypothesis; 60min cooldown
REM guarantees two stacked BUYs in a single session cannot repeat.
set SMC_MAX_CONCURRENT_PER_SYMBOL=1
set SMC_ANTI_STACK_COOLDOWN_MINUTES=60
set SMC_RANGE_REVERSAL_CONFIRM_ENABLED=true
REM Round 7 P0: AI-aware mode_router (AIRegimeAssessment drives path selection).
REM Treatment leg only — Control leg (start_live.bat) stays on legacy mode_router
REM for live A/B comparison. Backtest 2020-2024 validated: 2023 Δ PF = 0.00
REM (regression eliminated), 2022 -0.86 is 2-trade sample noise.
set SMC_AI_MODE_ROUTER_ENABLED=true
set SMC_AI_REGIME_TRUST_THRESHOLD=0.6
REM Optional: override virtual balance split.  Default is 50/50.
REM set SMC_VIRTUAL_BALANCE_SPLIT={"": 0.5, "_macro": 0.5}
REM Optional: override treatment-leg magic (default 19760428).
REM set SMC_MACRO_MAGIC=19760428
cd /d C:\AI-SMC
.venv\Scripts\python.exe scripts\live_demo.py >> logs\live_macro_stdout.log 2>> logs\live_macro_stderr.log

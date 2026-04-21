"""AI-SMC Live Trading Loop — dual-mode (trending + ranging).

Runs every M15 bar close:
1. Fetch SYMBOL data from MT5
2. Run AI direction analysis (Claude debate or SMA fallback)
3. Detect range bounds on H1 (always, for display)
4. Route trading mode: trending (v1 5-gate) or ranging (mean-reversion)
5. Output BUY / SELL / RANGE BUY / RANGE SELL / HOLD signal
6. Save state to data/{SYMBOL}/live_state.json for dashboard

Usage:
    python scripts/live_demo.py                      # XAUUSD (default)
    python scripts/live_demo.py --symbol BTCUSD
    python scripts/live_demo.py --paper              # paper mode, no real MT5 orders
"""
import sys
import os
import time
import signal
import json
import atexit
import shutil

os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Round 4.6-L: single-instance guard (VPS found 2 live_demo.py processes running
# concurrently, causing journal append race and quota state file clobber).
# Atomic PID file — O_CREAT|O_EXCL fails if another instance already started.
# Round 4.6-M (skeptic H4 defense): realpath(__file__) independent of cwd, so
# rel-path vs abs-path invocations resolve to the same PID file path.
# Round 5 T2 (BTC): PID file is per-symbol so XAU and BTC processes don't
# deadlock each other. --symbol arg is parsed early (before heavy imports)
# just to derive the PID path.
def _early_symbol_arg() -> str:
    for i, arg in enumerate(sys.argv):
        if arg == "--symbol" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if arg.startswith("--symbol="):
            return arg.split("=", 1)[1]
    return "XAUUSD"


_SYMBOL_EARLY = _early_symbol_arg()
# audit-r4 v5 Option B: per-suffix PID file so control + treatment processes
# don't deadlock each other (both run on same host, same symbol, different
# journal_suffix via SMC_JOURNAL_SUFFIX env var).
_SUFFIX_EARLY = os.environ.get("SMC_JOURNAL_SUFFIX", "")
_PID_FILE = os.path.realpath(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "..",
        "data",
        _SYMBOL_EARLY,
        f"live_demo{_SUFFIX_EARLY}.pid",
    )
)


def _pid_alive(pid: int) -> bool:
    """Check if a process with the given PID is alive. Pure stdlib, cross-platform.

    POSIX: signal 0 raises ProcessLookupError if dead.
    Windows: OpenProcess + GetExitCodeProcess via ctypes.
    """
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return False
            return exit_code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _ensure_single_instance():
    os.makedirs(os.path.dirname(_PID_FILE), exist_ok=True)
    # Round 5-P0-4: stale PID self-heal. If an old PID file exists but the owning
    # process no longer runs (SIGKILL / OOM / hard reboot: atexit did not fire),
    # remove the stale file and retry instead of refusing to start.
    try:
        fd = os.open(_PID_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        stale_pid = -1
        try:
            with open(_PID_FILE, "r", encoding="utf-8") as f:
                stale_pid = int(f.read().strip() or "-1")
        except (OSError, ValueError):
            stale_pid = -1
        if stale_pid > 0 and _pid_alive(stale_pid):
            sys.stderr.write(
                f"[4.6-L] live_demo.py already running (PID {stale_pid}). Exiting.\n"
            )
            sys.exit(1)
        sys.stderr.write(
            f"[Round5-P0-4] stale PID file ({stale_pid}) — removing and restarting.\n"
        )
        try:
            os.remove(_PID_FILE)
        except OSError:
            pass
        try:
            fd = os.open(_PID_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            sys.stderr.write("[Round5-P0-4] race on PID create after stale remove. Exiting.\n")
            sys.exit(1)
    os.write(fd, str(os.getpid()).encode("utf-8"))
    os.close(fd)
    atexit.register(lambda: os.path.exists(_PID_FILE) and os.remove(_PID_FILE))


_ensure_single_instance()

import MetaTrader5 as mt5
import polars as pl
from datetime import datetime, timezone, timedelta
from pathlib import Path
from smc.data.schemas import Timeframe
from smc.smc_core.detector import SMCDetector
from smc.strategy.aggregator import MultiTimeframeAggregator
from smc.strategy.regime import classify_regime
from smc.strategy.range_trader import RangeTrader
from smc.strategy.breakout_detector import BreakoutDetector
from smc.strategy.mode_router import route_trading_mode
from smc.strategy.range_trader import (
    _ASIAN_SESSIONS,
    check_bounds_only_guards,
    check_range_guards,
)
from smc.strategy.range_quota import AsianRangeQuota
from smc.strategy.phase1a_circuit_breaker import Phase1aCircuitBreaker
from smc.strategy.htf_bias import compute_htf_bias, htf_bias_tier
from smc.monitor.timing import next_bar_close
from smc.monitor.structured_log import crit as log_crit, warn as log_warn, info as log_info
from smc.monitor.critical_alerter import alert_critical
# Round 5 stability R1: MT5 handle auto-heal watchdog — detects IPC handle rot
# (tick → None for N cycles) and reinitialises; gives up after 5 consecutive
# failures so Task Scheduler can respawn a fresh process with a fresh handle.
from smc.monitor import mt5_watchdog
# Round 5 stability R2: per-cycle health_probe event for uptime / SLA digest.
from smc.monitor import health_probe
from smc.monitor.state_io import atomic_write_json
from smc.monitor.reconcile_cursor import load_reconcile_cursor, save_reconcile_cursor
from smc.strategy.session import get_session_info
from smc.execution.mt5_send import send_with_retry, compute_dynamic_deviation
# Round 5 T1 F2+F3: real broker reconciliation + daily halt gating.
from smc.execution.mt5_positions_adapter import (
    fetch_broker_positions,
    fetch_closed_pnl_since,
)
from smc.risk.consec_loss_halt import ConsecLossHalt
from smc.risk.drawdown_guard import DrawdownGuard
from smc.risk.margin_cap import check_margin_cap
from smc.risk.live_position_sizer import compute_live_position_size
# Round 4 Alt-B W2: macro overlay (COT / TIPS / DXY)
from smc.ai.macro_layer import MacroLayer

# ---------------------------------------------------------------------------
# Per-symbol data paths (populated in main() after argparse).
# Top-level names kept as module-level vars so helper functions reference them;
# main() assigns the real Path objects before calling any helper.
# ---------------------------------------------------------------------------
JOURNAL_PATH: Path = Path("data/journal/live_trades.jsonl")  # overwritten in main
STATE_PATH: Path = Path("data/live_state.json")              # overwritten in main
AI_PATH: Path = Path("data/ai_analysis.json")                # overwritten in main
PAUSE_FLAG_PATH: Path = Path("data/trading_paused.flag")     # overwritten in main
MT5_POSITIONS_PATH: Path = Path("data/mt5_positions.json")   # overwritten in main


def _migrate_legacy_xau_paths(symbol: str) -> None:
    """Move pre-BTC-split data/*.json from data/ to data/XAUUSD/ once. Idempotent."""
    if symbol != "XAUUSD":
        return
    legacy = Path("data")
    target = legacy / "XAUUSD"
    target.mkdir(parents=True, exist_ok=True)
    (target / "journal").mkdir(parents=True, exist_ok=True)
    # Note: live_demo.pid intentionally NOT migrated — PID files are runtime
    # locks owned by the currently-running process. The per-symbol PID path
    # (data/XAUUSD/live_demo.pid) is created by _ensure_single_instance() at
    # startup; any stale legacy data/live_demo.pid is left for operator cleanup.
    for name in [
        "live_state.json",
        "ai_analysis.json",
        "asian_range_quota_state.json",
        "consec_loss_state.json",
        "user_config.json",
        "paused.flag",
        "trading_paused.flag",
        "range_cooldown_state.json",
        "phase1a_breaker_state.json",
        "mt5_positions.json",
        "execution_circuit_open.flag",
    ]:
        src = legacy / name
        dst = target / name
        if src.exists() and not dst.exists():
            shutil.move(str(src), str(dst))
    src_jnl = legacy / "journal" / "live_trades.jsonl"
    dst_jnl = target / "journal" / "live_trades.jsonl"
    if src_jnl.exists() and not dst_jnl.exists():
        shutil.move(str(src_jnl), str(dst_jnl))


def fetch_mt5_data(mt5_path: str = "XAUUSD"):
    """Fetch latest bars from MT5 for all timeframes.

    Parameters
    ----------
    mt5_path:
        MT5 symbol path, e.g. ``"XAUUSD"`` or ``"Bitcoin\\BTCUSD"``.
    """
    data = {}
    for label, mt5_tf, smc_tf in [
        ("D1", mt5.TIMEFRAME_D1, Timeframe.D1),
        ("H4", mt5.TIMEFRAME_H4, Timeframe.H4),
        ("H1", mt5.TIMEFRAME_H1, Timeframe.H1),
        ("M15", mt5.TIMEFRAME_M15, Timeframe.M15),
    ]:
        rates = mt5.copy_rates_from_pos(mt5_path, mt5_tf, 0, 500)
        if rates is not None and len(rates) > 0:
            df = pl.DataFrame({
                "ts": [datetime.fromtimestamp(r[0], tz=timezone.utc) for r in rates],
                "open": [float(r[1]) for r in rates],
                "high": [float(r[2]) for r in rates],
                "low": [float(r[3]) for r in rates],
                "close": [float(r[4]) for r in rates],
                "volume": [float(r[5]) for r in rates],
                "spread": [float(r[6]) for r in rates],
            }).with_columns(pl.col("ts").cast(pl.Datetime("ns", "UTC")))
            data[smc_tf] = df
    return data


def run_ai_analysis(data):
    """Run AI direction analysis and save result."""
    analysis = {"source": "technical", "assessed_at": datetime.now(timezone.utc).isoformat()}

    if Timeframe.D1 in data:
        closes = data[Timeframe.D1]["close"].to_list()
        if len(closes) >= 50:
            sma20 = sum(closes[-20:]) / 20
            sma50 = sum(closes[-50:]) / 50
            price = closes[-1]
            analysis["sma20"] = round(sma20, 2)
            analysis["sma50"] = round(sma50, 2)

            if price > sma20 > sma50:
                analysis["direction"] = "bullish"
                analysis["confidence"] = 0.7
            elif price < sma20 < sma50:
                analysis["direction"] = "bearish"
                analysis["confidence"] = 0.7
            else:
                analysis["direction"] = "neutral"
                analysis["confidence"] = 0.3
            analysis["reasoning"] = f"Price ${price:.0f} vs SMA20 ${sma20:.0f} vs SMA50 ${sma50:.0f}"

    # Try AI debate (Claude CLI) — Round 5 T5: 180s hard cap via ThreadPoolExecutor
    # Prevents worst-case 9-step debate (4 analysts + 2×bull/bear + judge) at
    # 120s/step from hanging an entire M15 cycle (18 min worst case → now ≤3 min).
    try:
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout
        from smc.ai.direction_engine import DirectionEngine
        engine = DirectionEngine(cache_ttl_hours=1)  # Round 4.6-Q (USER): 4h→1h
        h4_df = data.get(Timeframe.H4)
        # Round 5 R2: capture wall-clock elapsed for the AI call so the
        # per-cycle health_probe can report debate_elapsed_ms_last.
        _ai_start_mono = time.monotonic()
        with ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(engine.get_direction, h4_df=h4_df)
            ai_dir = fut.result(timeout=180)  # 3min hard cap
        analysis["ai_elapsed_ms"] = int((time.monotonic() - _ai_start_mono) * 1000)
        if ai_dir.source != "neutral_default":
            analysis["ai_direction"] = ai_dir.direction
            analysis["ai_confidence"] = round(ai_dir.confidence, 3)
            analysis["ai_reasoning"] = ai_dir.reasoning
            analysis["ai_source"] = ai_dir.source
            analysis["ai_key_drivers"] = list(ai_dir.key_drivers) if ai_dir.key_drivers else []
            analysis["source"] = f"technical + {ai_dir.source}"
    except FutTimeout:
        analysis["ai_error"] = "ai_timeout_180s"
        analysis["ai_elapsed_ms"] = 180_000
        log_warn("ai_timeout", cycle_hint=analysis.get("assessed_at", "?"))
    except Exception as e:
        analysis["ai_error"] = str(e)[:100]

    # Volatility
    if Timeframe.D1 in data:
        closes = data[Timeframe.D1]["close"].to_list()
        highs = data[Timeframe.D1]["high"].to_list()
        lows = data[Timeframe.D1]["low"].to_list()
        if len(closes) >= 15:
            trs = []
            for i in range(1, min(15, len(closes))):
                tr = max(highs[-i] - lows[-i], abs(highs[-i] - closes[-(i+1)]), abs(lows[-i] - closes[-(i+1)]))
                trs.append(tr)
            atr = sum(trs) / len(trs)
            analysis["atr_pct"] = round(atr / closes[-1] * 100, 2)
            analysis["volatility"] = "HIGH" if analysis["atr_pct"] > 1.4 else ("LOW" if analysis["atr_pct"] < 1.0 else "NORMAL")

    # Save for dashboard
    AI_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(AI_PATH, "w") as f:
        json.dump(analysis, f, indent=2, default=str)

    return analysis


def _determine_trending(setups, ai_dir, ai_conf, effective_conf, session):
    """V1 trending mode: 5-gate AI filter (unchanged from original logic).

    Returns (action, reason, best_setup).
    """
    # GATE 1: AI confidence after session penalty
    if effective_conf < 0.5:
        return "HOLD", f"LOW CONFIDENCE: AI {ai_dir.upper()} {ai_conf:.0%} → {effective_conf:.0%} (need 50%+)", None

    # GATE 3: AI must not be neutral
    if ai_dir == "neutral":
        return "HOLD", f"AI NEUTRAL ({ai_conf:.0%}) — no trade without directional clarity", None

    # GATE 4: Must have SMC setups
    if not setups:
        return "HOLD", f"AI {ai_dir.upper()} ({effective_conf:.0%}) but no SMC setups found", None

    best = max(setups, key=lambda s: s.confluence_score)
    e = best.entry_signal

    # GATE 5: Direction alignment (AI must agree with SMC)
    ai_trade_dir = "long" if ai_dir == "bullish" else "short"
    if ai_trade_dir == e.direction:
        action = "BUY" if e.direction == "long" else "SELL"
        lot = "0.01" if effective_conf >= 0.8 else ("0.005" if effective_conf >= 0.6 else "0.0025")
        reason = (f"AI {ai_dir.upper()} ({effective_conf:.0%}) + SMC {e.direction.upper()} "
                  f"| {e.trigger_type} | conf {best.confluence_score:.2f} | lot {lot} | {session}")
        return action, reason, best
    else:
        if ai_conf < 0.65:
            action = f"{'BUY' if e.direction == 'long' else 'SELL'} (override)"
            reason = (f"SOFT OVERRIDE: AI {ai_dir.upper()} ({ai_conf:.0%}) vs SMC {e.direction.upper()} "
                      f"| AI uncertain → SMC wins at 1/4 lot | {e.trigger_type}")
            return action, reason, best
        else:
            return "HOLD", f"AI BLOCKED: AI {ai_dir.upper()} ({ai_conf:.0%}) vs SMC {e.direction.upper()} | AI confident → no trade", None


def _determine_v1_passthrough(setups, session):
    """V1 passthrough: AI is unsure, let v1 HTF bias decide direction.

    Setups already passed compute_htf_bias (D1+H4 BOS/CHoCH) and confluence
    scoring in MultiTimeframeAggregator.generate_setups(). We trust v1's own
    direction filter and only apply session + setup availability gates.

    Returns (action, reason, best_setup).
    """
    if not setups:
        return "HOLD", "[AI_NEUTRAL_FALLBACK] V1 PASSTHROUGH: no SMC setups from HTF bias", None

    best = max(setups, key=lambda s: s.confluence_score)
    e = best.entry_signal
    action = "BUY" if e.direction == "long" else "SELL"
    reason = (f"[AI_NEUTRAL_FALLBACK] V1 PASSTHROUGH: AI unsure → HTF bias {e.direction.upper()} "
              f"| {e.trigger_type} | conf {best.confluence_score:.2f} "
              f"| lot 0.0025 (reduced) | {session}")
    return action, reason, best


def _determine_ranging(price, range_bounds, h1_snapshot, m15_snapshot, h1_atr,
                       range_trader, breakout_detector, session="", h1_df=None,
                       htf_bias=None, cfg=None, m15_df=None):
    """Ranging mode: breakout guard + mean-reversion setups at range boundaries.

    Returns (action, reason, best_setup). Round 4.6-F: session kwarg so
    generate_range_setups can apply Asian-wide boundary_pct (30%).
    Round 4.6-K: h1_df enables setup-level check_range_guards enforcement
    (RR>=1.2, touches>=2) — closing the 4.6-E "deferred" TODO.
    Round 5 T0 (P0-9): htf_bias enables Guard 6 HTF alignment — rejects range
    setups that oppose a confident HTF bias (confidence >= 0.5).
    """
    # Breakout invalidation — if price breaks the range, hold and wait
    breakout = breakout_detector.check_breakout(price, range_bounds, h1_atr)
    if breakout != "none":
        return "HOLD", f"BREAKOUT: {breakout} — range invalidated", None

    range_setups = range_trader.generate_range_setups(
        h1_snapshot, m15_snapshot, price, range_bounds, h1_atr,
        session=session,
        m15_df=m15_df,
    )

    # Round 4.6-K: setup-level guards (RR>=1.2, touches>=2) enforcement.
    # Round 5 T0 (P0-9): Guard 6 HTF alignment — pass htf_bias into each check.
    if range_setups and h1_df is not None:
        range_setups = tuple(
            s for s in range_setups
            if check_range_guards(range_bounds, s, session, h1_df, htf_bias=htf_bias, cfg=cfg)
        )

    if range_setups:
        best = max(range_setups, key=lambda s: s.confidence)
        action = "RANGE BUY" if best.direction == "long" else "RANGE SELL"
        reason = (f"RANGE {best.trigger} | "
                  f"${range_bounds.lower:.0f}-${range_bounds.upper:.0f}")
        return action, reason, best

    return (
        "HOLD",
        f"RANGING but no boundary setups | "
        f"Range ${range_bounds.lower:.0f}-${range_bounds.upper:.0f}",
        None,
    )


def determine_action(setups, ai_analysis, regime, *,
                     h1_df=None, h1_snapshot=None, m15_snapshot=None,
                     h1_atr=0.0, price=0.0,
                     range_trader=None, breakout_detector=None,
                     asian_range_quota=None, phase1a_breaker=None,
                     htf_bias=None, cfg=None, m15_df=None):
    """Dual-mode action router: trending (v1 5-gate) or ranging (mean-reversion).

    Always detects range for display. Mode router decides which path runs.
    Returns (action, reason, best_setup, mode_decision).
    Round 5 T0 (P0-9): htf_bias piped to _determine_ranging → check_range_guards
    Guard 6 (HTF alignment). None is safe (backward-compat default).
    """
    session, session_penalty = get_session_info(cfg=cfg)
    ai_dir = ai_analysis.get("ai_direction", ai_analysis.get("direction", "neutral"))
    ai_conf = ai_analysis.get("ai_confidence", ai_analysis.get("confidence", 0.3))
    effective_conf = ai_conf - session_penalty

    # Always detect range (for display even in trending mode)
    range_bounds = None
    if range_trader is not None and h1_snapshot is not None:
        range_bounds = range_trader.detect_range(h1_df, h1_snapshot)

    # Round 4.6-E: bounds-level guards precheck so mode_router Priority 2
    # (range_bounds + guards_passed + session) can actually fire.
    # Previously guards_passed defaulted to False and ranging never activated.
    guards_passed = False
    if range_bounds is not None and h1_df is not None:
        guards_passed = check_bounds_only_guards(range_bounds, session, h1_df, cfg=cfg)

    # Route trading mode
    mode = route_trading_mode(
        ai_direction=ai_dir,
        ai_confidence=effective_conf,
        regime=regime,
        session=session,
        range_bounds=range_bounds,
        guards_passed=guards_passed,
        current_price=price,
        cfg=cfg,
    )

    if mode.mode == "trending":
        action, reason, best = _determine_trending(
            setups, ai_dir, ai_conf, effective_conf, session,
        )
        return action, reason, best, mode

    if mode.mode == "ranging" and mode.range_bounds is not None:
        # Round 4.5 hotfix: CircuitBreaker 扩展到全 Asian (UTC 0-8)
        # 用户激活 ASIAN_CORE ranging 但接受风险 → 需要 breaker 保险
        if session in _ASIAN_SESSIONS and asian_range_quota is not None:
            if phase1a_breaker is not None and phase1a_breaker.is_tripped():
                return "HOLD", f"[ASIAN_BREAKER_TRIPPED] Asian ranging disabled | {session}", None, mode
            if asian_range_quota.is_exhausted_today(datetime.now(tz=timezone.utc)):
                return "HOLD", f"[COOLDOWN] Asian ranging already used today | {session}", None, mode
        action, reason, best = _determine_ranging(
            price, mode.range_bounds, h1_snapshot, m15_snapshot, h1_atr,
            range_trader, breakout_detector, session=session, h1_df=h1_df,
            htf_bias=htf_bias, cfg=cfg, m15_df=m15_df,
        )
        return action, reason, best, mode

    if mode.mode == "v1_passthrough":
        # AI unsure — let v1 pipeline decide using its own HTF bias.
        # v1 setups already passed compute_htf_bias (D1+H4 BOS/CHoCH)
        # and confluence scoring — they have their own direction filter.
        action, reason, best = _determine_v1_passthrough(setups, session)
        return action, reason, best, mode

    # Fallback: router returned no actionable mode
    return "HOLD", mode.reason, None, mode


def save_state(cycle, price, action, reason, ai_analysis, regime, setups,
               best_setup, *, mode_decision=None, range_trader=None, aggregator=None,
               blocked_reason: "str | None" = None,
               cfg=None, balance_usd: "float | None" = None, risk_pct: float = 1.0,
               position_size_lots: "float | None" = None,
               htf_bias=None):
    """Save live state for dashboard display (dual-mode aware).

    audit-r2 R1 (rev2 per ops-sustain):
      - position_size_lots can be supplied pre-computed (single source of
        truth with pre_write_gate) or recomputed inline via
        compute_live_position_size.  Supplying it is the preferred path —
        guarantees gate and EA see the same lot.
      - balance_usd is fail-closed: None or non-positive → lot = 0.0.  Never
        default to a literal ($10k) that can be magnitudes higher than demo
        equity.
    """
    state = {
        "cycle": cycle,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "price": price,
        "action": action,
        "reason": reason,
        "regime": regime,
        "ai_direction": ai_analysis.get("ai_direction", ai_analysis.get("direction", "?")),
        "ai_confidence": ai_analysis.get("ai_confidence", ai_analysis.get("confidence", 0)),
        "setups_count": len(setups),
        "volatility": ai_analysis.get("volatility", "?"),
        "trading_mode": mode_decision.mode if mode_decision else "trending",
        # Round 5 T5 audit r1: null when no gate tripped; str reason when blocked.
        "blocked_reason": blocked_reason,
    }

    # Range bounds (always populated when detected, even in trending mode)
    rb = mode_decision.range_bounds if mode_decision else None
    if rb is not None:
        state["range_bounds"] = {
            "upper": rb.upper,
            "lower": rb.lower,
            "width": rb.upper - rb.lower,
        }
    else:
        state["range_bounds"] = None

    # Round 4.6-C2 (measure-first): end-to-end 3-stage diagnostic
    # (detect / guards / setups) so we can locate the failing branch
    # without blind hotfixes.
    if range_trader is not None:
        from smc.strategy.range_trader import get_last_guards_diagnostic
        state["range_diagnostic"] = {
            "detect": dict(range_trader._last_diagnostic or {}),
            "guards": get_last_guards_diagnostic(),
            "setups": dict(range_trader._last_setups_diagnostic or {}),
        }

    # Round 4.6-S-diag: expose aggregator (v1 pipeline) stage-by-stage rejection
    # so dashboard/live_state 能 surface "为什么 v1_passthrough 0 setups".
    if aggregator is not None and hasattr(aggregator, "_last_setup_diagnostic"):
        state["smc_diagnostic"] = dict(aggregator._last_setup_diagnostic or {})

    # audit-r2 R1 (rev2): use caller-provided lots when given (single source
    # of truth with pre_write_gate); otherwise fall through to helper.
    if position_size_lots is None:
        if cfg is not None:
            position_size_lots = compute_live_position_size(
                best_setup,
                cfg=cfg,
                balance_usd=balance_usd,
                risk_pct=risk_pct,
                blocked_reason=blocked_reason,
            )
        else:
            position_size_lots = 0.0
    state["position_size"] = position_size_lots

    # audit-r2 ops #18 (Guard 6 debate monitor): expose HTF bias confidence
    # + tier bucket so decision-reviewer can accumulate session-level
    # distribution data to size Round 3 S2 two-stage soft-multiplier decision.
    # None-safe: halt path / missing snapshots → neutral tier, conf 0.0.
    _hb_conf = 0.0
    if htf_bias is not None:
        _hb_conf = float(getattr(htf_bias, "confidence", 0.0) or 0.0)
    state["htf_bias_conf"] = round(_hb_conf, 3)
    state["htf_bias_tier"] = htf_bias_tier(_hb_conf)

    if best_setup:
        # Range setups use .direction/.trigger/.confidence directly (RangeSetup)
        # Trending setups use .entry_signal (TradeSetup)
        # audit-r2 ops #18 (TP1 debate monitor):
        #   planned_rr_ratio: build-time nominal RR (what Guard 2 validated
        #     against the aggressive TP — take_profit_ext on RangeSetup,
        #     take_profit_1 on TradeSetup since trending has no separate
        #     aggressive target today).
        #   exec_rr_ratio: the RR MT5 actually sees on the order (midpoint
        #     for Range, tp1 for Trend — identical to planned for Trend).
        # Round 3 will compare hit-rate distribution planned vs exec to
        # decide Option A/B/C for TP1 policy.
        if hasattr(best_setup, "entry_signal"):
            e = best_setup.entry_signal
            # Trending: planned == exec (no separate aggressive target today)
            _risk_pts = abs(e.entry_price - e.stop_loss)
            _tp1_pts = abs(e.take_profit_1 - e.entry_price)
            _exec_rr = round(_tp1_pts / _risk_pts, 3) if _risk_pts > 0 else 0.0
            state["best_setup"] = {
                "direction": e.direction,
                "entry": e.entry_price,
                "sl": e.stop_loss,
                "tp1": e.take_profit_1,
                "trigger": e.trigger_type,
                "confluence": best_setup.confluence_score,
                # audit-r2 R1: strategy_server reads this → /signal.lot → EA.
                "position_size_lots": position_size_lots,
                # audit-r2 ops #18 (TP1 debate):
                "planned_rr_ratio": _exec_rr,  # trending has no nominal-vs-exec divergence
                "exec_rr_ratio": _exec_rr,
            }
        else:
            # RangeSetup: planned (aggressive TP=tp_ext) vs exec (midpoint TP)
            _entry = getattr(best_setup, "entry_price", price)
            _sl = getattr(best_setup, "stop_loss", 0)
            _tp_exec = getattr(best_setup, "take_profit", 0)
            _tp_planned = getattr(best_setup, "take_profit_ext", _tp_exec)
            _risk_pts = abs(_entry - _sl)
            _planned_rr = round(abs(_tp_planned - _entry) / _risk_pts, 3) if _risk_pts > 0 else 0.0
            _exec_rr = round(abs(_tp_exec - _entry) / _risk_pts, 3) if _risk_pts > 0 else 0.0
            state["best_setup"] = {
                "direction": best_setup.direction,
                "entry": _entry,
                "sl": _sl,
                "tp1": _tp_exec,
                "trigger": best_setup.trigger,
                "confluence": best_setup.confidence,
                # audit-r2 R1: strategy_server reads this → /signal.lot → EA.
                "position_size_lots": position_size_lots,
                # audit-r2 ops #18 (TP1 debate):
                "planned_rr_ratio": _planned_rr,
                "exec_rr_ratio": _exec_rr,
            }

    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2, default=str)

    return state


def main():
    import argparse

    parser = argparse.ArgumentParser(description="AI-SMC Live Trading Loop")
    parser.add_argument(
        "--symbol",
        choices=["XAUUSD", "BTCUSD"],
        default="XAUUSD",
        help="Trading symbol (default: XAUUSD)",
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Paper mode — log signals but send no real MT5 orders",
    )
    parser.add_argument(
        "--no-execute",
        action="store_true",
        help=(
            "Signal-only mode — compute strategy + write live_state.json for "
            "external executor (e.g. AISMCReceiver.mq5 EA) but do NOT call "
            "mt5.order_send from Python. Avoids MT5 multi-client IPC issues."
        ),
    )
    args = parser.parse_args()
    SYMBOL = args.symbol
    PAPER_MODE = args.paper or args.no_execute  # both short-circuit order_send
    NO_EXECUTE = args.no_execute

    from smc.instruments import get_instrument_config
    cfg = get_instrument_config(SYMBOL)

    # -----------------------------------------------------------------------
    # One-time legacy migration: move pre-BTC-split flat data/ files into
    # data/XAUUSD/ so new per-symbol layout takes effect without data loss.
    # Idempotent — second run is a no-op.
    # -----------------------------------------------------------------------
    _migrate_legacy_xau_paths(SYMBOL)

    # -----------------------------------------------------------------------
    # Per-symbol data paths — must be assigned before any helper that
    # references the module-level path vars (fetch_mt5_data, save_state, …).
    # -----------------------------------------------------------------------
    global JOURNAL_PATH, STATE_PATH, AI_PATH, PAUSE_FLAG_PATH, MT5_POSITIONS_PATH
    DATA_ROOT = Path("data") / SYMBOL
    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    # Round 4 Alt-B W3: A/B suffix so process A (control) and process B
    # (treatment, SMC_MACRO_ENABLED=true) write to separate journals and
    # live_state files without interfering with each other.
    from smc.config import SMCConfig as _SMCConfigPaths
    _path_cfg = _SMCConfigPaths()
    _journal_suffix: str = _path_cfg.journal_suffix  # "" or e.g. "_macro"

    # audit-r4 v5 Option B: resolve effective magic for this leg.
    # Control (suffix="") → cfg.magic (XAU=19760418, BTC=19760419).
    # Treatment (suffix="_macro") → cfg.macro_magic (default 19760428).
    # Same TMGM Demo account; different magic lets broker reconcile split legs.
    _effective_magic: int = _path_cfg.magic_for(cfg.magic, _journal_suffix)

    _journal_dir = DATA_ROOT / f"journal{_journal_suffix}"
    _journal_dir.mkdir(parents=True, exist_ok=True)
    JOURNAL_PATH = _journal_dir / "live_trades.jsonl"
    STATE_PATH = DATA_ROOT / f"live_state{_journal_suffix}.json"
    AI_PATH = DATA_ROOT / "ai_analysis.json"
    # audit-r4 v5 Option B: per-suffix pause flag so control leg and treatment
    # leg can be paused independently via the dashboard kill switch.
    PAUSE_FLAG_PATH = DATA_ROOT / f"trading_paused{_journal_suffix}.flag"
    MT5_POSITIONS_PATH = DATA_ROOT / f"mt5_positions{_journal_suffix}.json"
    # audit-r4 v5 Option B: per-suffix risk state files so control + treatment
    # legs track halt / quota / breaker independently on the same TMGM Demo
    # account.  Control (suffix="") writes to <name>.json; treatment
    # (suffix="_macro") writes to <name>_macro.json.  Backward-compat: control
    # leg paths are byte-identical to pre-Option-B behaviour.
    CONSEC_LOSS_PATH = DATA_ROOT / f"consec_loss_state{_journal_suffix}.json"
    ASIAN_QUOTA_PATH = DATA_ROOT / f"asian_range_quota_state{_journal_suffix}.json"
    # Round 5 T2 Stage 5: per-symbol circuit breaker flag so XAU and BTC
    # processes each own their execution circuit independently.
    CIRCUIT_FLAG_PATH = DATA_ROOT / f"execution_circuit_open{_journal_suffix}.flag"
    # audit-r2 ops #4: reconcile cursor — prevents double-counting of
    # closed deals across restarts (silent bug surfaced by ops-sustain).
    RECONCILE_TS_PATH = DATA_ROOT / f"last_reconcile_ts{_journal_suffix}.json"
    PHASE1A_BREAKER_PATH = DATA_ROOT / f"phase1a_breaker_state{_journal_suffix}.json"

    print(f"[{datetime.now()}] AI-SMC Live Trading Loop Starting...")
    print("=" * 60)
    print(f"  Mode:       {'PAPER (no real orders)' if PAPER_MODE else 'LIVE'}")
    print(f"  Instrument: {SYMBOL}")
    print("  AI:         Claude Debate + SMA Fallback")
    print("  Strategy:   DUAL-MODE (trending v1 + ranging)")
    print(f"  Data root:  {DATA_ROOT}")
    print("=" * 60)
    print()

    # Round 5 T4 band-aid: retry mt5.initialize() up to 10 times with 5s backoff
    # (10 × 5s = 50s covers MT5 cold-start 30s window).
    # Proper fix (Week 2) is an MQL5 EA signal receiver that removes Python ↔
    # MT5 IPC coupling entirely. See docs/MILESTONE_20260418.md Round 5 T4.
    # Round 5 T5: extended from 3 → 10 attempts.
    _mt5_init_ok = False
    for _attempt in range(1, 11):
        if mt5.initialize():
            _mt5_init_ok = True
            if _attempt > 1:
                log_info("mt5_init_retry_success", attempt=_attempt)
            break
        err = mt5.last_error()
        log_warn("mt5_init_attempt_failed", attempt=_attempt, error=str(err))
        print(f"MT5 init attempt {_attempt}/10 failed: {err}")
        if _attempt < 10:
            time.sleep(5)
    if not _mt5_init_ok:
        alert_critical(
            "mt5_init_exhausted",
            attempts=10,
            last_error=str(mt5.last_error()),
            send_telegram=True,
        )
        sys.exit(1)

    # Round 5 stability R1: initialise MT5 handle watchdog state.  The
    # monotonic timestamp we stamp here is the reference point for
    # `handle_age_sec` in the per-cycle health_probe event (R2).
    mt5_wd = mt5_watchdog.mark_initialized(mt5_watchdog.new_state())

    info = mt5.account_info()
    print(f"MT5 Connected: {info.login} @ {info.server}")
    print(f"Balance: ${info.balance}")
    print()
    # Round 5 T1 F1: Telegram startup alert. send_telegram=True pushes if env
    # SMC_TELEGRAM_BOT_TOKEN / _CHAT_ID are set; otherwise we just get the
    # [CRIT] stderr line for the VPS `tail -f | grep CRIT` flow.
    alert_critical(
        "system_startup",
        account=info.login,
        server=info.server,
        balance=float(info.balance),
        mt5_execute=os.environ.get("SMC_MT5_EXECUTE", "0"),
        send_telegram=True,
    )

    detector = SMCDetector(swing_length=10)
    aggregator = MultiTimeframeAggregator(
        detector=detector,
        ai_regime_enabled=_path_cfg.ai_regime_enabled,
    )

    # Round 4 Alt-B W2+W3: _path_cfg (SMCConfig) already loaded above for
    # journal_suffix; reuse the same instance for macro overlay settings.
    _smc_cfg = _path_cfg
    _macro_flag: bool = _smc_cfg.macro_enabled
    _fred_key_val: str = _smc_cfg.fred_api_key.get_secret_value()

    # Round 4 Alt-B W2: initialise MacroLayer once per process.
    # Cache is per-symbol so XAUUSD and BTCUSD processes don't race on
    # the same Parquet files.
    macro_layer = MacroLayer(
        cache_dir=DATA_ROOT / "macro",
        fred_api_key=_fred_key_val or None,
        cache_ttl_hours=_smc_cfg.macro_cache_ttl_hours,
    )
    log_info(
        "macro_layer_init",
        macro_enabled=_macro_flag,
        cache_dir=str(DATA_ROOT / "macro"),
    )
    log_info(
        "ai_regime_init",
        ai_regime_enabled=_path_cfg.ai_regime_enabled,
        ai_regime_min_confidence=_path_cfg.ai_regime_min_confidence,
    )

    # Round 5 T3 (dual-symbol-audit P0): inject cfg + per-symbol cooldown path
    # so BTC doesn't silently run XAU params (Donchian 48 vs 24, width 200pts vs
    # 2% pct, boundary 0.30 vs 0.25, guards 800/400 vs 1500, RR 1.2 vs 1.5).
    range_trader = RangeTrader(
        cfg=cfg,
        cooldown_state_path=DATA_ROOT / "range_cooldown_state.json",
        reversal_confirm_enabled=_path_cfg.range_reversal_confirm_enabled,
    )
    breakout_det = BreakoutDetector()

    JOURNAL_PATH.parent.mkdir(parents=True, exist_ok=True)

    running = True
    def stop(sig, frame):
        nonlocal running
        print(f"\n[{datetime.now()}] Shutdown signal received.")
        running = False
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    cycle = 0
    ai_analysis = {}
    last_ai_update = 0
    # Round 4.6-H2: load persisted quota across restarts (防进程重启静默重复开仓)
    # Stage 3: pass per-symbol path so XAUUSD/BTCUSD states are isolated.
    # cfg.use_asian_quota=False (BTC) → quota object still created but
    # is_exhausted_today() always returns False because record_open never fires
    # (guarded by session in _ASIAN_SESSIONS, which is empty for BTC).
    asian_range_quota = AsianRangeQuota.load(state_path=ASIAN_QUOTA_PATH)
    # Round 5 T3 (dual-symbol-audit P0): per-symbol breaker state file so
    # XAU and BTC processes don't clobber each other's Phase 1a state.
    # audit-r4 v5 Option B: also per-suffix so control and treatment don't
    # clobber on same TMGM Demo account.
    phase1a_breaker = Phase1aCircuitBreaker(
        state_path=PHASE1A_BREAKER_PATH,
    )
    # Round 5 T1 F3: consecutive-loss halt (3 losses in a row → halt rest of
    # UTC day; WIN resets streak). DrawdownGuard is a %-based backstop.
    # audit-r3 R4: consec_limit from cfg.consec_loss_limit (per-symbol).
    # XAU=3, BTC=3 today but interface open for future per-instrument tuning.
    consec_halt = ConsecLossHalt(
        state_path=CONSEC_LOSS_PATH,
        consec_limit=cfg.consec_loss_limit,
    )
    drawdown_guard = DrawdownGuard(max_daily_loss_pct=3.0, max_drawdown_pct=10.0)
    # peak_balance and daily_pnl are tracked across cycles from reconciled
    # closes. Initialized from the current MT5 balance at startup so the
    # guard reasons about post-restart P&L rather than session-zero.
    _initial_balance = float(info.balance)
    peak_balance = _initial_balance
    daily_pnl = 0.0
    # audit-r3 R3: track the UTC date for which peak_balance was last reset.
    # On UTC 00:00 rollover we reset peak_balance to current balance so the
    # drawdown guard reasons about *today's* peak, not historical all-time
    # high.  Without reset, a bad week would bake a "ghost peak" into the
    # guard indefinitely, producing spurious drawdown_guard trips even after
    # the account recovers intra-day.  Mirrors Phase1aCircuitBreaker /
    # ConsecLossHalt daily-reset semantics (UTC 0:00 boundary).
    from datetime import date as _date
    peak_balance_reset_date: _date = datetime.now(timezone.utc).date()
    # audit-r2 ops #4: reload persisted reconcile cursor so closed deals
    # processed pre-restart are NOT re-played into consec_halt /
    # phase1a_breaker / daily_pnl on startup.  Missing or corrupt cursor
    # falls back to the original "now - 12h" first-boot behaviour.
    last_reconcile_ts = load_reconcile_cursor(RECONCILE_TS_PATH)

    while running:
        cycle += 1
        now = datetime.now(timezone.utc)

        nxt = next_bar_close(Timeframe.M15, now)
        wait = (nxt - now).total_seconds()

        if wait > 0:
            print(f"[{now.strftime('%H:%M:%S')} UTC] Cycle {cycle}: next M15 at {nxt.strftime('%H:%M')} ({wait:.0f}s)")
            while wait > 0 and running:
                time.sleep(min(10, wait))
                wait -= 10

        if not running:
            break

        now = datetime.now(timezone.utc)
        print(f"\n{'='*60}")
        print(f"[{now.strftime('%H:%M:%S')} UTC] CYCLE {cycle}")
        print(f"{'='*60}")

        # audit-r3 R3: daily peak_balance reset at UTC 0:00 boundary.
        # Keeps drawdown_guard.total_drawdown_pct reasoning about *today's*
        # peak, preventing historical highs from spuriously tripping the
        # %-based drawdown backstop after recovery.  Reset to current live
        # balance (fall back to peak_balance if account_info fails).
        #
        # NOTE (decision-reviewer micro-suggestion): `peak_balance_reset_date`
        # advances EVEN IF the balance probe fails.  This prevents a retry
        # storm: without date advance, every cycle during an MT5 outage would
        # re-enter this branch, call account_info() repeatedly, and flood
        # structured logs / Telegram alerts.  Analog: k8s liveness probes
        # have backoff, not hot-loop retry.  If the reset was imperfect today,
        # tomorrow's boundary will try again cleanly.
        _today_utc = now.date()
        if _today_utc != peak_balance_reset_date:
            _balance_info_success = False
            try:
                _reset_info = mt5.account_info()
                if _reset_info is not None and float(_reset_info.balance) > 0:
                    _reset_balance = float(_reset_info.balance)
                    _balance_info_success = True
                else:
                    _reset_balance = peak_balance
            except Exception:
                _reset_balance = peak_balance
            log_info(
                "peak_balance_daily_reset",
                cycle=cycle,
                prev_peak=round(peak_balance, 2),
                new_peak=round(_reset_balance, 2),
                from_date=peak_balance_reset_date.isoformat(),
                to_date=_today_utc.isoformat(),
                balance_info_success=_balance_info_success,
            )
            peak_balance = _reset_balance
            peak_balance_reset_date = _today_utc

        # audit-r2 ops #3: halt paths fall through to save_state so the
        # dashboard's "为什么不开仓" card surfaces the real reason (symmetry
        # with pre_write_gate added in Round 1).  Before this fix, halt
        # `continue` skipped save_state entirely — live_state.json went
        # stale and watchdog_smart's Axis-3 freshness check could mistake
        # a legitimate halt for a process crash.
        _halt_blocked_reason: str | None = None
        _halt_label: str | None = None
        _halt_display_reason: str | None = None

        # Round 5-P0-2a: honor dashboard KillSwitch. If the pause flag is present,
        # skip all trading logic for this cycle. dashboard_server.py creates the
        # flag on POST /api/toggle_trading; live_demo was previously ignoring it.
        if PAUSE_FLAG_PATH.exists():
            log_warn("trading_paused", cycle=cycle, flag=str(PAUSE_FLAG_PATH))
            print(f"  [PAUSED] trading_paused.flag present — skipping cycle {cycle}")
            _halt_blocked_reason = "kill_switch:dashboard_paused"
            _halt_label = "PAUSED"
            _halt_display_reason = "Dashboard kill-switch active (trading_paused.flag)"

        # Round 5 T1 F3: daily-halt gating. Consecutive-loss halt is the
        # user-visible "亏 3 单自动停" rule; DrawdownGuard is the %-based
        # backstop (3% daily loss). Either trip blocks new opens until the
        # next UTC 00:00 rollover (Phase1a/ConsecLoss) or a manual reset.
        if _halt_blocked_reason is None and consec_halt.is_tripped():
            snap = consec_halt.snapshot()
            log_warn(
                "consec_loss_halt_active",
                cycle=cycle,
                consec_losses=snap.consec_losses,
                tripped_at=snap.tripped_at,
            )
            print(f"  [HALT] consec-loss halt ({snap.consec_losses} losses) — skipping cycle {cycle}")
            _halt_blocked_reason = f"consec_loss_halt:losses={snap.consec_losses}"
            _halt_label = "HALT"
            _halt_display_reason = f"连亏 {snap.consec_losses} 单触发每日保险"

        if _halt_blocked_reason is None:
            budget = drawdown_guard.check_budget(
                balance=float(info.balance),
                peak_balance=peak_balance,
                daily_pnl=daily_pnl,
            )
            if not budget.can_trade:
                log_warn(
                    "drawdown_guard_block",
                    cycle=cycle,
                    reason=budget.rejection_reason,
                    daily_loss_pct=budget.daily_loss_pct,
                    total_drawdown_pct=budget.total_drawdown_pct,
                )
                print(f"  [HALT] drawdown guard: {budget.rejection_reason}")
                _halt_blocked_reason = f"drawdown_guard:{budget.rejection_reason}"
                _halt_label = "HALT"
                _halt_display_reason = f"回撤保护: {budget.rejection_reason}"

        if _halt_blocked_reason is not None:
            # audit-r2 ops #3: write HOLD state so dashboard + EA /signal
            # both see the halt reason explicitly instead of stale data.
            try:
                save_state(
                    cycle=cycle,
                    price=0.0,
                    action="HOLD",
                    reason=f"[{_halt_label}] {_halt_display_reason}",
                    ai_analysis={},
                    regime="unknown",
                    setups=(),
                    best_setup=None,
                    mode_decision=None,
                    range_trader=None,
                    aggregator=None,
                    blocked_reason=_halt_blocked_reason,
                    cfg=cfg,
                    balance_usd=None,  # halt path: no sizer work needed → fail-closed to 0 lots
                    risk_pct=1.0,
                    position_size_lots=0.0,
                )
            except Exception as _halt_save_exc:
                log_warn("halt_save_state_fail", exc=str(_halt_save_exc)[:120])
            continue

        try:
            # 1. Price
            tick = mt5.symbol_info_tick(cfg.mt5_path)
            tick_ok = tick is not None
            # Round 5 stability R1: feed tick outcome into the watchdog so
            # consecutive tick_none streaks trigger handle reinit and, after
            # 5 failures, process exit for Task Scheduler respawn.
            mt5_wd = mt5_watchdog.record_tick_result(mt5_wd, tick_ok=tick_ok)
            if not tick_ok:
                log_warn("tick_unavailable", cycle=cycle)
                print("  WARN: no tick data")
                if mt5_watchdog.should_giveup(mt5_wd):
                    alert_critical(
                        "mt5_handle_reset_giveup",
                        streak=mt5_wd.consecutive_tick_none,
                        reset_attempts=mt5_wd.reset_attempts,
                        send_telegram=True,
                    )
                    log_crit(
                        "mt5_handle_reset_giveup",
                        streak=mt5_wd.consecutive_tick_none,
                        reset_attempts=mt5_wd.reset_attempts,
                    )
                    print(
                        f"  CRIT: {mt5_wd.consecutive_tick_none} consecutive "
                        f"tick_none — exiting for Task Scheduler respawn."
                    )
                    sys.exit(1)
                if mt5_watchdog.should_reset(mt5_wd):
                    mt5_wd = mt5_watchdog.try_reset_handle(mt5, mt5_wd)
                continue
            price = tick.bid
            spread = tick.ask - tick.bid
            print(f"  {SYMBOL}: ${price:.2f} (spread ${spread:.2f})")

            # 2. Fetch data
            data = fetch_mt5_data(cfg.mt5_path)
            bars_info = ", ".join(f"{k}: {len(v)}" for k, v in data.items())
            print(f"  Data: {{{bars_info}}}")

            # Round 5 stability R2: emit one health_probe per cycle for the
            # daily SLA digest + ops dashboard P&L card.  A single fresh
            # mt5.account_info() call gives balance/equity/floating — we
            # accept the one IPC roundtrip per cycle because the dashboard
            # needs a live floating-pnl number (cached `info.balance` only
            # updates on reconcile, which would stale the card).
            _probe_balance: float | None = None
            _probe_equity: float | None = None
            _probe_floating: float | None = None
            try:
                _probe_acc = mt5.account_info()
                if _probe_acc is not None:
                    _probe_balance = float(_probe_acc.balance)
                    _probe_equity = float(_probe_acc.equity)
                    _probe_floating = round(_probe_equity - _probe_balance, 2)
            except Exception:
                pass
            try:
                _macro_fresh = bool(macro_layer.is_cache_fresh()) if _macro_flag else False
            except Exception:
                _macro_fresh = False
            _probe = health_probe.build_probe(
                cycle=cycle,
                cycle_ts_iso=now.isoformat(),
                leg=_journal_suffix,
                tick_ok=True,  # we would have `continue`d above if not
                data=data,
                handle_age_sec=mt5_watchdog.handle_age_sec(mt5_wd),
                handle_reset_count=mt5_wd.reset_attempts,
                debate_elapsed_ms_last=ai_analysis.get("ai_elapsed_ms"),
                macro_bias_fresh=_macro_fresh,
                balance_usd=_probe_balance,
                equity_usd=_probe_equity,
                floating_usd=_probe_floating,
            )
            health_probe.emit(_probe)

            # 3. AI Analysis (every H4 = every 16 M15 cycles, or first run)
            if cycle == 1 or (time.time() - last_ai_update) > 14400:
                print(f"  Running AI analysis...")
                ai_analysis = run_ai_analysis(data)
                last_ai_update = time.time()
                ai_dir = ai_analysis.get("ai_direction", ai_analysis.get("direction", "?"))
                ai_conf = ai_analysis.get("ai_confidence", ai_analysis.get("confidence", 0))
                print(f"  AI Direction: {ai_dir.upper()} ({ai_conf:.0%} confidence)")
            else:
                ai_dir = ai_analysis.get("ai_direction", ai_analysis.get("direction", "?"))
                print(f"  AI Direction: {ai_dir.upper()} (cached)")

            # 4. Regime
            regime = classify_regime(data.get(Timeframe.D1), cfg=cfg)
            print(f"  Regime: {regime}")

            # 4b. Detect SMC snapshots for HTF bias + range detection
            h1_df = data.get(Timeframe.H1)
            h1_snapshot = None
            if h1_df is not None and len(h1_df) > 0:
                h1_snapshot = detector.detect(h1_df, Timeframe.H1)
            m15_snapshot = None
            m15_df = data.get(Timeframe.M15)
            if m15_df is not None and len(m15_df) > 0:
                m15_snapshot = detector.detect(m15_df, Timeframe.M15)
            # Round 5 T0 (P0-9): D1 + H4 snapshots for HTF bias → Guard 6 alignment
            # in range_trader.check_range_guards. Previously only h1/m15 detected;
            # Guard 6 would pass-through with htf_bias=None (backward-compat default).
            d1_df = data.get(Timeframe.D1)
            d1_snapshot = detector.detect(d1_df, Timeframe.D1) if d1_df is not None and len(d1_df) > 0 else None
            h4_df = data.get(Timeframe.H4)
            h4_snapshot = detector.detect(h4_df, Timeframe.H4) if h4_df is not None and len(h4_df) > 0 else None
            htf_bias = compute_htf_bias(d1_snapshot, h4_snapshot)
            h1_atr = aggregator._compute_h1_atr(h1_df)

            # 5a. Round 4 Alt-B W2: compute macro overlay bias (config-gated).
            # Runs BEFORE generate_setups so macro_bias is available for scoring.
            # Failure is non-fatal: log warning and proceed with 0.0 (baseline mode).
            _macro_enabled = _macro_flag
            macro_bias_value: float = 0.0
            _macro_components: dict = {"dxy": 0.0, "cot": 0.0, "yield": 0.0}
            if _macro_enabled:
                try:
                    _mb = macro_layer.compute_macro_bias(instrument=cfg.symbol)
                    macro_bias_value = _mb.total_bias
                    _macro_components = {
                        "dxy": _mb.dxy_bias,
                        "cot": _mb.cot_bias,
                        "yield": _mb.yield_bias,
                    }
                    log_info(
                        "macro_bias_computed",
                        cycle=cycle,
                        total_bias=round(macro_bias_value, 4),
                        direction=_mb.direction,
                        sources_available=_mb.sources_available,
                        cot=_mb.cot_bias,
                        yield_b=_mb.yield_bias,
                        dxy=_mb.dxy_bias,
                    )
                    print(
                        f"  Macro bias: {macro_bias_value:+.4f} "
                        f"(dir={_mb.direction}, sources={_mb.sources_available})"
                    )
                except Exception as _macro_exc:
                    log_warn(
                        "macro_bias_fetch_failed",
                        cycle=cycle,
                        exc=str(_macro_exc)[:120],
                    )
                    macro_bias_value = 0.0

            # 5b. Thread macro_bias into aggregator for this cycle.
            aggregator.set_macro_bias(macro_bias_value)

            # 5. Strategy (v1 trending setups — always generated)
            setups = aggregator.generate_setups(data, price)
            print(f"  SMC Setups: {len(setups)}")

            # 6. Dual-mode action routing
            session, _ = get_session_info(cfg=cfg)
            action, reason, best, mode = determine_action(
                setups, ai_analysis, regime,
                h1_df=h1_df,
                h1_snapshot=h1_snapshot,
                m15_snapshot=m15_snapshot,
                h1_atr=h1_atr,
                price=price,
                range_trader=range_trader,
                breakout_detector=breakout_det,
                asian_range_quota=asian_range_quota,
                phase1a_breaker=phase1a_breaker,
                htf_bias=htf_bias,
                cfg=cfg,
                m15_df=m15_df,
            )
            # Round 5 T0 (P0-2b): quota record_open moved *after* successful
            # order_send below (inside LIVE_EXEC branch). MT5 failures no longer
            # consume Asian 1/day quota.
            # TODO: When paper/live trade-close tracking is implemented, call
            # phase1a_breaker.record_trade_close(pnl_usd) after each
            # ASIAN_LONDON_TRANSITION ranging trade closes.

            # Round 5 T5 audit r1 (P0-risk): Pre-write risk gate for EA architecture.
            # In EA mode Python does NOT call order_send, so margin_cap and
            # asian_range_quota gates inside the execution block never fire.
            # We intercept here — before save_state writes live_state.json — so
            # strategy_server /signal reads HOLD and EA naturally skips opening.
            blocked_reason: str | None = None

            # audit-r2 R1 (rev2 per ops-sustain #1 + #2): compute lot size BEFORE
            # margin_cap so the gate evaluates the actual order volume the EA
            # will receive.  Previous revision read a non-existent attribute on
            # the TradeSetup/RangeSetup instance and silently fell back to
            # 0.01 lot, disabling margin_cap for any lot > 0.01.
            # balance_usd is fail-closed: unknown equity → 0 lots, not a
            # $10k fantasy that would magnify risk on a $1k demo account.
            _live_balance_usd: float | None = None
            try:
                _acc_info = mt5.account_info()
                if _acc_info is not None and float(_acc_info.balance) > 0:
                    _live_balance_usd = float(_acc_info.balance)
            except Exception as _bal_exc:
                log_warn("balance_probe_error", exc=str(_bal_exc)[:120])

            # audit-r4 v5 Option B: virtual balance split prevents treatment
            # leg from over-sizing using the full shared account equity.
            # Control (suffix="") sees 50% of mt5 balance; treatment
            # (suffix="_macro") sees the other 50% (configurable via
            # SMC_VIRTUAL_BALANCE_SPLIT).  When split=1.0 for a given suffix,
            # this matches pre-Option-B behaviour exactly (single-leg mode).
            _virtual_balance_usd: float | None
            if _live_balance_usd is None:
                _virtual_balance_usd = None
            else:
                _virtual_balance_usd = _path_cfg.virtual_balance_for(
                    _journal_suffix, _live_balance_usd,
                )
                log_info(
                    "virtual_balance_applied",
                    cycle=cycle,
                    suffix=_journal_suffix,
                    mt5_balance=round(_live_balance_usd, 2),
                    virtual_balance=round(_virtual_balance_usd, 2),
                    split=_path_cfg.virtual_balance_split.get(_journal_suffix, 0.5),
                )

            planned_lots = compute_live_position_size(
                best,
                cfg=cfg,
                balance_usd=_virtual_balance_usd,
                risk_pct=1.0,
                blocked_reason=None,  # gate will set blocked_reason below
            )

            if best is not None:
                # Gate 1: margin_cap — requires live MT5 account_info
                try:
                    _gate_tick = mt5.symbol_info_tick(cfg.mt5_path)
                    _gate_price = (
                        float(getattr(_gate_tick, "ask" if getattr(best, "direction", "long") == "long" else "bid", price))
                        if _gate_tick is not None
                        else price
                    )
                    _gate_order_type = (
                        mt5.ORDER_TYPE_BUY if getattr(best, "direction", "long") == "long"
                        else mt5.ORDER_TYPE_SELL
                    )
                    # audit-r2 R1 (rev2): use the SAME lot size save_state /
                    # strategy_server / EA will see.  If sizer returned 0.0
                    # (unknown balance / bad SL), gate falls back to min_lot so
                    # margin_cap still sanity-checks the smallest possible
                    # tradeable order — preserves old behaviour for the
                    # degenerate case.
                    _gate_lots = planned_lots if planned_lots > 0 else cfg.min_lot
                    _margin_result = check_margin_cap(
                        mt5,
                        symbol=cfg.mt5_path,
                        action=_gate_order_type,
                        volume=_gate_lots,
                        price=_gate_price,
                        max_pct=0.40,
                    )
                    if not _margin_result.can_trade:
                        blocked_reason = f"margin_cap:{_margin_result.reason}"
                except Exception as _gate_exc:
                    log_warn("pre_write_margin_cap_error", exc=str(_gate_exc)[:120])

                # Gate 2: asian_range_quota — only during Asian sessions
                if not blocked_reason and session in _ASIAN_SESSIONS:
                    try:
                        if asian_range_quota.is_exhausted_today(datetime.now(tz=timezone.utc)):
                            blocked_reason = "asian_quota:exhausted_today"
                    except Exception as _quota_exc:
                        log_warn("pre_write_quota_error", exc=str(_quota_exc)[:120])

                # Gate 3 (Round 4 v5): max_concurrent_per_symbol hard cap.
                # Prevents 2026-04-20 02:46-style disaster where 5 BUYs
                # stacked in a declining window and all hit SL together.
                if not blocked_reason:
                    try:
                        from smc.risk.concurrent_gates import check_concurrent_cap
                        _open_positions = mt5.positions_get(symbol=cfg.mt5_path) or []
                        _cap_result = check_concurrent_cap(
                            _open_positions,
                            magic=_effective_magic,
                            max_concurrent=_path_cfg.max_concurrent_per_symbol,
                        )
                        if not _cap_result.can_trade:
                            blocked_reason = f"{_cap_result.reason}:{_cap_result.detail}"
                    except Exception as _cap_exc:
                        log_warn("pre_write_concurrent_cap_error", exc=str(_cap_exc)[:120])

                # Gate 4 (Round 4 v5): anti-stacking cooldown — even below
                # the hard cap, require N minutes between same-direction
                # entries on (symbol, magic).
                if not blocked_reason and _path_cfg.anti_stack_cooldown_minutes > 0:
                    try:
                        from smc.risk.concurrent_gates import check_anti_stack_cooldown
                        _want_dir = getattr(best, "direction", None)
                        _now_ts = datetime.now(tz=timezone.utc)
                        _cooldown = _path_cfg.anti_stack_cooldown_minutes
                        _lookback = _now_ts - timedelta(minutes=_cooldown + 5)
                        _recent_deals = mt5.history_deals_get(_lookback, _now_ts) or []
                        _stack_result = check_anti_stack_cooldown(
                            _recent_deals,
                            symbol=cfg.mt5_path,
                            magic=_effective_magic,
                            direction=_want_dir or "",
                            now=_now_ts,
                            cooldown_minutes=_cooldown,
                        )
                        if not _stack_result.can_trade:
                            blocked_reason = f"{_stack_result.reason}:{_stack_result.detail}"
                    except Exception as _stack_exc:
                        log_warn("pre_write_anti_stack_error", exc=str(_stack_exc)[:120])

            if blocked_reason:
                action = "HOLD"
                reason = f"[PRE_WRITE_GATE] {blocked_reason}"
                best = None
                planned_lots = 0.0  # audit-r2 R1 rev2: gate trip → lot=0 for save_state
                log_warn("pre_write_gate_blocked", blocked_reason=blocked_reason)
                print(f"  [PRE_WRITE_GATE] Blocked: {blocked_reason}")

            # Display action prominently
            action_colors = {
                "BUY": "+++", "SELL": "---", "HOLD": "===",
                "RANGE": "~~~", "V1": ">>>",
            }
            marker = action_colors.get(action.split()[0], "???")
            print()
            print(f"  {marker} ACTION: {action} [{mode.mode.upper()}] {marker}")
            print(f"  {reason}")

            # Display range bounds when detected (even in trending mode)
            if mode.range_bounds is not None:
                rb = mode.range_bounds
                print(f"  Range: ${rb.lower:.0f}-${rb.upper:.0f} | Width: ${rb.upper - rb.lower:.0f}")

            if best and hasattr(best, "entry_signal"):
                e = best.entry_signal
                print(f"  Entry: ${e.entry_price:.2f} | SL: ${e.stop_loss:.2f} | TP: ${e.take_profit_1:.2f}")
            elif best and hasattr(best, "entry_price"):
                print(f"  Entry: ${best.entry_price:.2f} | SL: ${best.stop_loss:.2f} | TP: ${best.take_profit:.2f}")

            # 7. Journal
            #    Round 4.6-H1: range ENTER 也要写 journal. 原代码只遍历 trending
            #    `setups` (TradeSetup tuple) 通过 s.entry_signal 取字段, 但 range
            #    path 的 `best` 是 RangeSetup 不在 setups 里 → journal 漏记.
            if action.startswith("RANGE") and best is not None:
                # Round 4.6-X (USER CRITICAL CATCH): MT5 order_send execution layer.
                # 之前 Lead 严重失职 — journal 只写 "PAPER" 日志, 没 call MT5 API.
                # audit-r3 V1 (HIGH): Python order_send is permanently
                # DISABLED.  The EA (mql5/AISMCReceiver.mq5) is the sole
                # production execution path since audit-r1 Round 5 T5
                # self-healing architecture refactor — it polls
                # strategy_server /signal and executes via MT5's native
                # CTrade.  If Python ALSO called order_send here, the
                # account would receive **two** orders per cycle (Python
                # + EA), doubling risk and corrupting consec_halt /
                # phase1a_breaker reconcile math.
                #
                # SMC_MT5_EXECUTE env var used to gate the legacy live
                # path; now kept only as an operator-misconfig detector.
                # If someone exports it = 1, we emit a deprecation warning
                # to structured logs so ops can catch the mistake.  The
                # send_with_retry code path below is RETAINED (dormant)
                # so a future roll-back from EA arch is a one-line flip.
                _mt5_execute = False
                if os.environ.get("SMC_MT5_EXECUTE", "0") == "1":
                    log_warn(
                        "smc_mt5_execute_deprecated",
                        cycle=cycle,
                        set_by_user=True,
                        note=(
                            "SMC_MT5_EXECUTE env var is deprecated; "
                            "EA (AISMCReceiver.mq5) is the sole order path. "
                            "Unset this var to silence this warning."
                        ),
                    )
                _mt5_ticket: int | None = None
                _mt5_send_retcode: int | None = None
                _mt5_mode_tag = "PAPER"
                _margin_gated = False  # set True when margin_cap gate blocks real order
                # Asian ranging 用 0.3x multiplier (Phase1a 降档协议), 其他 1.0x.
                # margin 公式分两步避免 bug:
                #   notional_usd = entry × contract_size × lots
                #   margin_usd   = notional / leverage
                # PAPER mode; live execution 时 MT5 按账户重算.
                # Round 4.6-J-fix: 用 _ASIAN_SESSIONS frozenset 避免 inline tuple drift.
                lot_multiplier = 0.3 if session in _ASIAN_SESSIONS else 1.0
                base_lot = cfg.min_lot
                position_size_lots = round(base_lot * lot_multiplier, 4)
                # Round 4.6-Y (USER CATCH): MT5 min lot = cfg.min_lot broker-wide.
                # Clamp Asian-reduced lot back to min. Asian risk management 改由
                # quota (1/day) 和 CircuitBreaker 承担, 不靠 fractional lot.
                if position_size_lots < cfg.min_lot:
                    position_size_lots = cfg.min_lot
                notional_value_usd = (
                    best.entry_price * cfg.contract_size * position_size_lots
                )
                margin_used_estimate_usd = round(
                    notional_value_usd / cfg.leverage_ratio, 2
                )

                if PAPER_MODE:
                    log_info(
                        "paper_mode_signal",
                        cycle=cycle,
                        symbol=SYMBOL,
                        direction=best.direction,
                        entry=round(best.entry_price, 2),
                        sl=round(best.stop_loss, 2),
                        tp=round(best.take_profit, 2),
                        lots=position_size_lots,
                        session=session,
                    )
                    print(
                        f"  [PAPER] Would submit: {best.direction.upper()} {position_size_lots} lots "
                        f"@ {best.entry_price:.2f} SL={best.stop_loss:.2f} TP={best.take_profit:.2f}"
                    )

                # Round 4.6-X + Round 5 T0 (P0-3): MT5 order_send via rugged wrapper
                # with retry / backoff / dynamic deviation / circuit breaker.
                # send_with_retry refreshes tick.ask/bid before each attempt and
                # opens a persistent circuit flag after 3 consecutive REQUOTE/EXC.
                _mt5_send_attempts = 0
                if _mt5_execute:
                    dyn_deviation = compute_dynamic_deviation(mt5, cfg.mt5_path, fallback=100)
                    order_type = mt5.ORDER_TYPE_BUY if best.direction == "long" else mt5.ORDER_TYPE_SELL
                    tick_for_margin = mt5.symbol_info_tick(cfg.mt5_path)
                    margin_price = (
                        float(getattr(tick_for_margin, "ask" if best.direction == "long" else "bid", best.entry_price))
                        if tick_for_margin is not None
                        else best.entry_price
                    )
                    # Round 5 T2 Stage 5: global margin cap gate before any real order.
                    # Checks that total margin (existing + proposed) stays within 40% of
                    # equity — prevents dual-symbol XAU+BTC from blowing the $1000 demo.
                    margin_check = check_margin_cap(
                        mt5,
                        symbol=cfg.mt5_path,
                        action=order_type,
                        volume=position_size_lots,
                        price=margin_price,
                        max_pct=0.40,
                    )
                    if not margin_check.can_trade:
                        log_warn(
                            "margin_cap_gate",
                            symbol=cfg.symbol,
                            reason=margin_check.reason,
                            margin_used=margin_check.current_margin_used,
                            equity=margin_check.current_equity,
                            proposed=margin_check.proposed_margin,
                            total_after=margin_check.total_after,
                            cap_ratio=margin_check.cap_ratio,
                        )
                        print(
                            f"  [MARGIN_CAP] Gate blocked: {margin_check.reason} "
                            f"(margin {margin_check.total_after:.2f} / equity {margin_check.current_equity:.2f})"
                        )
                        log_entry = {
                            "time": now.isoformat(),
                            "cycle": cycle,
                            "price": price,
                            "action": action,
                            "direction": best.direction,
                            "entry": best.entry_price,
                            "sl": best.stop_loss,
                            "tp1": best.take_profit,
                            "trigger": best.trigger,
                            "mode": "MARGIN_GATED",
                            "margin_gated": True,
                            "margin_reason": margin_check.reason,
                            "trading_mode": mode.mode,
                            "session": session,
                        }
                        with open(JOURNAL_PATH, "a") as f:
                            f.write(json.dumps(log_entry) + "\n")
                        # Skip real order; do not consume Asian quota.
                        _mt5_execute = False  # prevents the send_with_retry block below
                        _margin_gated = True
                    request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": cfg.mt5_path,
                        "volume": position_size_lots,
                        "type": order_type,
                        "price": 0.0,  # refreshed per attempt inside send_with_retry
                        "sl": best.stop_loss,
                        "tp": best.take_profit,
                        "deviation": dyn_deviation,
                        "magic": _effective_magic,
                        "comment": f"AI-SMC {best.trigger[:15]}",
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_IOC,
                    }
                if _mt5_execute:
                    # Round 5 stability R1: mark critical section so the
                    # handle watchdog will not call mt5.shutdown() in
                    # parallel if symbol_info_tick transiently returns None
                    # during send_with_retry's internal tick refresh loop.
                    # (Cycle is single-threaded today, but the flag keeps
                    # the contract explicit for future executor threading.)
                    mt5_wd = mt5_watchdog.enter_critical(mt5_wd)
                    send_result = send_with_retry(mt5, request, circuit_flag_path=CIRCUIT_FLAG_PATH)
                    mt5_wd = mt5_watchdog.exit_critical(mt5_wd)
                    _mt5_send_retcode = send_result.retcode
                    _mt5_ticket = send_result.ticket
                    _mt5_send_attempts = send_result.attempts
                    if send_result.success:
                        _mt5_mode_tag = "LIVE_EXEC"
                        print(
                            f"  [MT5] Order sent ✓ ticket={_mt5_ticket} "
                            f"attempts={_mt5_send_attempts} dev={dyn_deviation}"
                        )
                        log_info(
                            "mt5_order_sent",
                            cycle=cycle,
                            ticket=_mt5_ticket,
                            attempts=_mt5_send_attempts,
                            deviation=dyn_deviation,
                            direction=best.direction,
                            volume=position_size_lots,
                        )
                        # Round 5 T1 F1: Telegram push on successful open so user
                        # sees "开单" on phone without watching the dashboard.
                        # This is the happy-path counterpart to the mt5_order_fail
                        # alert_critical a few lines below.
                        alert_critical(
                            "mt5_order_opened",
                            cycle=cycle,
                            ticket=_mt5_ticket,
                            direction=best.direction,
                            entry=round(best.entry_price, 2),
                            sl=round(best.stop_loss, 2),
                            tp=round(best.take_profit, 2),
                            lots=position_size_lots,
                            session=session,
                            send_telegram=True,
                        )
                    else:
                        _mt5_mode_tag = f"MT5_FAIL_{_mt5_send_retcode}"
                        print(
                            f"  [MT5] Order FAIL retcode={_mt5_send_retcode} "
                            f"attempts={_mt5_send_attempts} msg={send_result.message}"
                        )
                        alert_critical(
                            "mt5_order_fail",
                            cycle=cycle,
                            retcode=_mt5_send_retcode,
                            attempts=_mt5_send_attempts,
                            message=send_result.message,
                            direction=best.direction,
                        )

                # Round 5 T0 (P0-2b): Asian quota only consumed on PAPER mode
                # (audit trail) or successful LIVE_EXEC. MT5 failures preserve
                # the 1/day quota so subsequent cycles can retry.
                # Stage 3: pass per-symbol state path so XAU/BTC are isolated.
                if session in _ASIAN_SESSIONS and _mt5_mode_tag in ("PAPER", "LIVE_EXEC"):
                    asian_range_quota = asian_range_quota.record_open(
                        datetime.now(tz=timezone.utc),
                        state_path=ASIAN_QUOTA_PATH,
                    )

                if not _margin_gated:
                    log_entry = {
                        "time": now.isoformat(),
                        "cycle": cycle,
                        "price": price,
                        "action": action,
                        "direction": best.direction,
                        "entry": best.entry_price,
                        "sl": best.stop_loss,
                        "tp1": best.take_profit,
                        "tp_ext": best.take_profit_ext,
                        "trigger": best.trigger,
                        "rr_ratio": best.rr_ratio,
                        "risk_points": best.risk_points,
                        "reward_points": best.reward_points,
                        "confidence": best.confidence,
                        "grade": best.grade,
                        "range_source": best.range_bounds.source,
                        "range_lower": best.range_bounds.lower,
                        "range_upper": best.range_bounds.upper,
                        "regime": regime,
                        "ai_direction": ai_analysis.get("ai_direction", ai_analysis.get("direction", "?")),
                        "mode": _mt5_mode_tag,  # 4.6-X: PAPER / LIVE_EXEC / MT5_FAIL_xxx
                        "mt5_ticket": _mt5_ticket,
                        "mt5_retcode": _mt5_send_retcode,
                        "mt5_send_attempts": _mt5_send_attempts,  # Round 5 T0 P0-3
                        "trading_mode": mode.mode,
                        "session": session,
                        # Round 4.6-J position sizing fields (4.6-J-fix: notional separated)
                        "lot_multiplier": lot_multiplier,
                        "position_size_lots": position_size_lots,
                        "notional_value_usd": round(notional_value_usd, 2),
                        "margin_used_estimate_usd": margin_used_estimate_usd,
                        "leverage_ratio": cfg.leverage_ratio,
                        # audit-r2 ops #18 (TP1 debate + Guard 6 monitor):
                        # planned = TP_ext (aggressive, Guard 2 validator);
                        # exec    = TP midpoint (what MT5 receives).
                        # Round 3 uses hit-rate of both to decide TP1 policy.
                        "planned_rr_ratio": round(
                            abs(best.take_profit_ext - best.entry_price)
                            / abs(best.entry_price - best.stop_loss),
                            3,
                        ) if abs(best.entry_price - best.stop_loss) > 0 else 0.0,
                        "exec_rr_ratio": round(
                            abs(best.take_profit - best.entry_price)
                            / abs(best.entry_price - best.stop_loss),
                            3,
                        ) if abs(best.entry_price - best.stop_loss) > 0 else 0.0,
                        "htf_bias_conf": round(float(getattr(htf_bias, "confidence", 0.0) or 0.0), 3),
                        "htf_bias_tier": htf_bias_tier(float(getattr(htf_bias, "confidence", 0.0) or 0.0)),
                        # audit-r4 v5 Option B: dual-magic audit fields for A/B analysis.
                        "journal_suffix": _journal_suffix,
                        "magic": _effective_magic,
                        "account_balance_usd": round(_live_balance_usd, 2) if _live_balance_usd is not None else None,
                        "virtual_balance_usd": round(_virtual_balance_usd, 2) if _virtual_balance_usd is not None else None,
                    }
                    with open(JOURNAL_PATH, "a") as f:
                        f.write(json.dumps(log_entry) + "\n")
            else:
                # Trending (v1 5-gate) path: iterate setups as before
                for s in setups:
                    e = s.entry_signal
                    # audit-r2 R1 rev2 (ops-sustain #4): surface the R1-computed
                    # lot in journal for --no-execute audit trail.  Trending
                    # setups currently do NOT use per-setup sizing (EA uses
                    # /signal.lot based on best_setup only), so we log the
                    # planned_lots for the best setup = the EA will execute,
                    # or 0.0 if this setup is not the one written to /signal.
                    _journal_lots = planned_lots if (best is s) else 0.0
                    # audit-r2 ops #18: trending has no separate aggressive TP;
                    # planned == exec.  Formula mirrors RangeSetup block above
                    # so journal comparisons across modes are apples-to-apples.
                    _t_risk = abs(e.entry_price - e.stop_loss)
                    _t_rr = round(abs(e.take_profit_1 - e.entry_price) / _t_risk, 3) if _t_risk > 0 else 0.0
                    log_entry = {
                        "time": now.isoformat(),
                        "cycle": cycle,
                        "price": price,
                        "action": action,
                        "direction": e.direction,
                        "entry": e.entry_price,
                        "sl": e.stop_loss,
                        "tp1": e.take_profit_1,
                        "trigger": e.trigger_type,
                        "confluence": round(s.confluence_score, 3),
                        "regime": regime,
                        "ai_direction": ai_analysis.get("ai_direction", ai_analysis.get("direction", "?")),
                        "mode": "PAPER",
                        "trading_mode": mode.mode,
                        # audit-r2 ops #18 (TP1 debate + Guard 6 monitor)
                        "planned_rr_ratio": _t_rr,  # trending: planned == exec
                        "exec_rr_ratio": _t_rr,
                        "htf_bias_conf": round(float(getattr(htf_bias, "confidence", 0.0) or 0.0), 3),
                        "htf_bias_tier": htf_bias_tier(float(getattr(htf_bias, "confidence", 0.0) or 0.0)),
                        "position_size_lots": _journal_lots,
                        # Round 4 Alt-B W2: macro overlay fields for A/B analysis
                        "macro_bias_value": round(macro_bias_value, 4),
                        "macro_enabled": _macro_enabled,
                        "macro_components": dict(_macro_components),
                        # audit-r4 v5 Option B: dual-magic audit fields for A/B analysis.
                        "journal_suffix": _journal_suffix,
                        "magic": _effective_magic,
                        "account_balance_usd": round(_live_balance_usd, 2) if _live_balance_usd is not None else None,
                        "virtual_balance_usd": round(_virtual_balance_usd, 2) if _virtual_balance_usd is not None else None,
                    }
                    with open(JOURNAL_PATH, "a") as f:
                        f.write(json.dumps(log_entry) + "\n")

            # Round 5 T1 F2: reconcile with real broker state + feed halt.
            # Writes data/mt5_positions.json for the dashboard Hero card and
            # records closed-trade P&L into the consec-loss halt + Phase1a
            # breaker + daily_pnl running total for DrawdownGuard backstop.
            try:
                broker_positions = fetch_broker_positions(mt5, symbol=cfg.mt5_path, magic=_effective_magic)
                atomic_write_json(
                    MT5_POSITIONS_PATH,
                    {
                        "ts": now.isoformat(),
                        "positions": [p.model_dump(mode="json") for p in broker_positions],
                    },
                )
                closed_deals = fetch_closed_pnl_since(mt5, last_reconcile_ts, magic=_effective_magic)
                for deal in closed_deals:
                    pnl = float(deal.get("pnl_usd", 0.0))
                    daily_pnl += pnl
                    consec_halt.record(pnl)
                    phase1a_breaker.record_trade_close(pnl)
                    log_info(
                        "trade_reconciled",
                        cycle=cycle,
                        ticket=deal.get("ticket"),
                        pnl_usd=pnl,
                        running_daily_pnl=round(daily_pnl, 2),
                    )
                    # Round 5 O3: enrich the trade_closed Telegram push with
                    # entry/exit/rr/regime/trigger so老板 can judge from the
                    # phone whether the close matched intent. Best-effort —
                    # enrichment never blocks the Telegram send.
                    from smc.monitor.trade_close_enrichment import (
                        build_trade_close_context,
                        format_trade_close_telegram,
                    )
                    try:
                        # Scan *both* per-symbol legs so a ticket from either
                        # control or treatment resolves. We pass the current
                        # JOURNAL_PATH first (fast hit on same-leg tickets).
                        _other_suffix = "_macro" if _journal_suffix == "" else ""
                        _journal_candidates = [
                            JOURNAL_PATH,
                            DATA_ROOT / f"journal{_other_suffix}" / "live_trades.jsonl",
                        ]
                        _ctx = build_trade_close_context(
                            ticket=int(deal.get("ticket", 0) or 0),
                            pnl_usd=pnl,
                            exit_price=deal.get("exit_price"),
                            close_time=deal.get("close_time"),
                            journal_paths=_journal_candidates,
                            structured_log_path=Path("logs/structured.jsonl"),
                        )
                        _telegram_body = format_trade_close_telegram(_ctx)
                    except Exception as _enrich_exc:
                        log_warn(
                            "trade_close_enrich_fail",
                            cycle=cycle,
                            ticket=deal.get("ticket"),
                            exception_type=type(_enrich_exc).__name__,
                            exception_msg=str(_enrich_exc)[:200],
                        )
                        _telegram_body = None

                    # Log the structured event (audit trail) — keep legacy fields
                    # so grep/parsers that depend on `pnl_usd` + `running_daily_pnl`
                    # continue to work. Only the Telegram message shape changes.
                    alert_critical(
                        "trade_closed",
                        cycle=cycle,
                        ticket=deal.get("ticket"),
                        pnl_usd=pnl,
                        running_daily_pnl=round(daily_pnl, 2),
                        rr_realized=_ctx.rr_realized if _telegram_body else None,
                        regime_at_entry=_ctx.regime_at_entry if _telegram_body else None,
                        leg=_ctx.leg_label if _telegram_body else None,
                        send_telegram=False,  # Telegram sent separately with enriched body
                    )
                    if _telegram_body:
                        try:
                            # Send the enriched multi-line body directly via
                            # the raw alerter so newlines survive (bypasses
                            # alert_critical's k=v flattening).
                            import asyncio as _asyncio
                            import os as _os
                            from smc.monitor.alerter import TelegramAlerter
                            _tok = _os.getenv("SMC_TELEGRAM_BOT_TOKEN") or _os.getenv("TELEGRAM_BOT_TOKEN")
                            _cid = _os.getenv("SMC_TELEGRAM_CHAT_ID") or _os.getenv("TELEGRAM_CHAT_ID")
                            if _tok and _cid:
                                _alerter = TelegramAlerter(bot_token=_tok, chat_id=_cid)
                                try:
                                    _asyncio.run(_alerter.send_text(_telegram_body))
                                except RuntimeError:
                                    # Running loop; fire-and-forget.
                                    try:
                                        _loop = _asyncio.get_running_loop()
                                        _loop.create_task(_alerter.send_text(_telegram_body))
                                    except Exception:
                                        pass
                        except Exception:
                            pass  # best-effort — never break trading loop
                # Update peak_balance from live account equity for next cycle.
                try:
                    info = mt5.account_info()
                    peak_balance = max(peak_balance, float(info.balance))
                except Exception:
                    pass
                last_reconcile_ts = now
                # audit-r2 ops #4: persist cursor immediately after updating
                # in-memory state so next restart picks up from here, not
                # from now-12h.  Wrapped in try/except to keep trading loop
                # alive even if disk write fails (next cycle will retry).
                try:
                    save_reconcile_cursor(RECONCILE_TS_PATH, last_reconcile_ts)
                except Exception as _cursor_exc:
                    log_warn("reconcile_cursor_save_fail", exc=str(_cursor_exc)[:120])
                # If the halt just tripped this cycle, fire a loud alert so the
                # user knows the system went to sleep for the day.
                if consec_halt.is_tripped() and closed_deals:
                    snap = consec_halt.snapshot()
                    alert_critical(
                        "daily_halt_triggered",
                        cycle=cycle,
                        consec_losses=snap.consec_losses,
                        tripped_at=snap.tripped_at,
                        send_telegram=True,
                    )
            except Exception as rec_exc:
                log_warn(
                    "reconcile_failed",
                    cycle=cycle,
                    exception_type=type(rec_exc).__name__,
                    exception_msg=str(rec_exc)[:200],
                )

            # 8. Save state for dashboard (dual-mode aware)
            # NOTE: canonical save_state call moved to finally block below so
            # dashboard always reflects the most recent attempt even on crash.

        except Exception as exc:
            import traceback
            tb_tail = traceback.format_exc(limit=4)
            print(f"  ERROR: {exc}")
            traceback.print_exc()
            # Round 5 T0 (P0-7 + P0-8): structured critical alert with bounded
            # traceback for grep-ability and optional Telegram push.
            alert_critical(
                "cycle_exception",
                cycle=cycle,
                exception_type=type(exc).__name__,
                exception_msg=str(exc)[:200],
                traceback_tail=tb_tail[-800:],
            )
        finally:
            # Round 5 T5: defensive save_state — runs even if cycle body crashes
            # mid-way so dashboard/watchdog always see the latest attempt.
            # Variables that may be unset if an exception fired early are
            # retrieved via locals() with safe defaults.
            _locs = locals()
            # audit-r2 R1 rev2 (ops-sustain #2): fail-closed balance.  Prefer
            # the _live_balance_usd already probed in the main body; fall
            # through to peak_balance; NEVER default to a literal ($10k) —
            # oversizing on a $1k demo would trigger one-tick liquidation.
            # audit-r4 v5 Option B: prefer the pre-computed _virtual_balance_usd
            # so save_state sees the same per-leg budget the pre-write gate used.
            _balance_for_sizing = _locs.get("_virtual_balance_usd", None)
            if _balance_for_sizing is None:
                _mt5_balance_fallback = _locs.get("_live_balance_usd", None)
                if _mt5_balance_fallback is None and peak_balance:
                    _mt5_balance_fallback = float(peak_balance)
                if _mt5_balance_fallback is not None:
                    _balance_for_sizing = _path_cfg.virtual_balance_for(
                        _journal_suffix, _mt5_balance_fallback,
                    )
            try:
                save_state(
                    _locs.get("cycle", cycle),
                    _locs.get("price", 0.0),
                    _locs.get("action", "UNKNOWN"),
                    _locs.get("reason", "cycle_error"),
                    _locs.get("ai_analysis", ai_analysis),
                    _locs.get("regime", "unknown"),
                    _locs.get("setups", ()),
                    _locs.get("best", None),
                    mode_decision=_locs.get("mode", None),
                    range_trader=range_trader,
                    aggregator=aggregator,
                    blocked_reason=_locs.get("blocked_reason", None),
                    cfg=cfg,
                    balance_usd=_balance_for_sizing,
                    risk_pct=1.0,  # TODO: pull from cfg when risk_pct added to InstrumentConfig
                    position_size_lots=_locs.get("planned_lots", None),  # single source of truth
                    htf_bias=_locs.get("htf_bias", None),  # audit-r2 ops #18 Guard 6 monitor
                )
            except Exception as save_exc:
                log_warn("save_state_fail", exc=str(save_exc)[:100])

    mt5.shutdown()
    print(f"\n[{datetime.now()}] AI-SMC Live Trading Loop stopped.")


if __name__ == "__main__":
    main()

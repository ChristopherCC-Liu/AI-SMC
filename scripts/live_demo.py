"""AI-SMC Live Trading Loop — dual-mode (trending + ranging).

Runs every M15 bar close:
1. Fetch XAUUSD data from MT5
2. Run AI direction analysis (Claude debate or SMA fallback)
3. Detect range bounds on H1 (always, for display)
4. Route trading mode: trending (v1 5-gate) or ranging (mean-reversion)
5. Output BUY / SELL / RANGE BUY / RANGE SELL / HOLD signal
6. Save state to data/live_state.json for dashboard

Usage:
    python scripts/live_demo.py
"""
import sys
import os
import time
import signal
import json
import atexit

os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Round 4.6-L: single-instance guard (VPS found 2 live_demo.py processes running
# concurrently, causing journal append race and quota state file clobber).
# Atomic PID file — O_CREAT|O_EXCL fails if another instance already started.
# Round 4.6-M (skeptic H4 defense): realpath(__file__) independent of cwd, so
# rel-path vs abs-path invocations resolve to the same PID file path.
_PID_FILE = os.path.realpath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data", "live_demo.pid")
)


def _ensure_single_instance():
    os.makedirs(os.path.dirname(_PID_FILE), exist_ok=True)
    try:
        fd = os.open(_PID_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode("utf-8"))
        os.close(fd)
        atexit.register(lambda: os.path.exists(_PID_FILE) and os.remove(_PID_FILE))
    except FileExistsError:
        try:
            with open(_PID_FILE, "r", encoding="utf-8") as f:
                existing = f.read().strip()
        except OSError:
            existing = "unknown"
        sys.stderr.write(
            f"[4.6-L] live_demo.py already running (PID {existing}). "
            f"If stale, delete {_PID_FILE} and restart. Exiting.\n"
        )
        sys.exit(1)


_ensure_single_instance()

import MetaTrader5 as mt5
import polars as pl
from datetime import datetime, timezone
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
from smc.monitor.timing import next_bar_close

JOURNAL_PATH = Path("data/journal/live_trades.jsonl")
STATE_PATH = Path("data/live_state.json")
AI_PATH = Path("data/ai_analysis.json")


def fetch_mt5_data():
    """Fetch latest XAUUSD bars from MT5 for all timeframes."""
    data = {}
    for label, mt5_tf, smc_tf in [
        ("D1", mt5.TIMEFRAME_D1, Timeframe.D1),
        ("H4", mt5.TIMEFRAME_H4, Timeframe.H4),
        ("H1", mt5.TIMEFRAME_H1, Timeframe.H1),
        ("M15", mt5.TIMEFRAME_M15, Timeframe.M15),
    ]:
        rates = mt5.copy_rates_from_pos("XAUUSD", mt5_tf, 0, 500)
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

    # Try AI debate (Claude CLI)
    try:
        from smc.ai.direction_engine import DirectionEngine
        engine = DirectionEngine(cache_ttl_hours=1)  # Round 4.6-Q (USER): 4h→1h
        h4_df = data.get(Timeframe.H4)
        ai_dir = engine.get_direction(h4_df=h4_df)
        if ai_dir.source != "neutral_default":
            analysis["ai_direction"] = ai_dir.direction
            analysis["ai_confidence"] = round(ai_dir.confidence, 3)
            analysis["ai_reasoning"] = ai_dir.reasoning
            analysis["ai_source"] = ai_dir.source
            analysis["ai_key_drivers"] = list(ai_dir.key_drivers) if ai_dir.key_drivers else []
            analysis["source"] = f"technical + {ai_dir.source}"
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


def get_session_info():
    """Return (session_name, confidence_penalty) based on current UTC hour."""
    hour = datetime.now(timezone.utc).hour
    if 0 <= hour < 6:
        return "ASIAN_CORE", 0.2
    elif 6 <= hour < 8:
        return "ASIAN_LONDON_TRANSITION", 0.15  # Asian-London transition, gentler penalty
    elif 8 <= hour < 13:
        return "LONDON", 0.0
    elif 13 <= hour < 16:
        return "LONDON/NY OVERLAP", 0.0
    elif 16 <= hour < 21:
        return "NEW YORK", 0.0
    elif 21 <= hour < 24:
        return "LATE NY", 0.1
    else:  # defensive, should not reach
        return "ASIAN_CORE", 0.2


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
                       range_trader, breakout_detector, session="", h1_df=None):
    """Ranging mode: breakout guard + mean-reversion setups at range boundaries.

    Returns (action, reason, best_setup). Round 4.6-F: session kwarg so
    generate_range_setups can apply Asian-wide boundary_pct (30%).
    Round 4.6-K: h1_df enables setup-level check_range_guards enforcement
    (RR>=1.2, touches>=2) — closing the 4.6-E "deferred" TODO.
    """
    # Breakout invalidation — if price breaks the range, hold and wait
    breakout = breakout_detector.check_breakout(price, range_bounds, h1_atr)
    if breakout != "none":
        return "HOLD", f"BREAKOUT: {breakout} — range invalidated", None

    range_setups = range_trader.generate_range_setups(
        h1_snapshot, m15_snapshot, price, range_bounds, h1_atr,
        session=session,
    )

    # Round 4.6-K: setup-level guards (RR>=1.2, touches>=2) enforcement.
    # Was "deferred" in 4.6-E but never wired into production path.
    if range_setups and h1_df is not None:
        range_setups = tuple(
            s for s in range_setups
            if check_range_guards(range_bounds, s, session, h1_df)
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
                     asian_range_quota=None, phase1a_breaker=None):
    """Dual-mode action router: trending (v1 5-gate) or ranging (mean-reversion).

    Always detects range for display. Mode router decides which path runs.
    Returns (action, reason, best_setup, mode_decision).
    """
    session, session_penalty = get_session_info()
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
        guards_passed = check_bounds_only_guards(range_bounds, session, h1_df)

    # Route trading mode
    mode = route_trading_mode(
        ai_direction=ai_dir,
        ai_confidence=effective_conf,
        regime=regime,
        session=session,
        range_bounds=range_bounds,
        guards_passed=guards_passed,
        current_price=price,
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
               best_setup, *, mode_decision=None, range_trader=None, aggregator=None):
    """Save live state for dashboard display (dual-mode aware)."""
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

    if best_setup:
        # Range setups use .direction/.trigger/.confidence directly (RangeSetup)
        # Trending setups use .entry_signal (TradeSetup)
        if hasattr(best_setup, "entry_signal"):
            e = best_setup.entry_signal
            state["best_setup"] = {
                "direction": e.direction,
                "entry": e.entry_price,
                "sl": e.stop_loss,
                "tp1": e.take_profit_1,
                "trigger": e.trigger_type,
                "confluence": best_setup.confluence_score,
            }
        else:
            # RangeSetup from range_trader
            state["best_setup"] = {
                "direction": best_setup.direction,
                "entry": getattr(best_setup, "entry_price", price),
                "sl": getattr(best_setup, "stop_loss", 0),
                "tp1": getattr(best_setup, "take_profit", 0),
                "trigger": best_setup.trigger,
                "confluence": best_setup.confidence,
            }

    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2, default=str)

    return state


def main():
    print(f"[{datetime.now()}] AI-SMC Live Trading Loop Starting...")
    print("=" * 60)
    print("  Mode:       DEMO (paper, log only)")
    print("  Instrument: XAUUSD")
    print("  AI:         Claude Debate + SMA Fallback")
    print("  Strategy:   DUAL-MODE (trending v1 + ranging)")
    print("=" * 60)
    print()

    if not mt5.initialize():
        print(f"MT5 init failed: {mt5.last_error()}")
        sys.exit(1)

    info = mt5.account_info()
    print(f"MT5 Connected: {info.login} @ {info.server}")
    print(f"Balance: ${info.balance}")
    print()

    detector = SMCDetector(swing_length=10)
    aggregator = MultiTimeframeAggregator(detector=detector, ai_regime_enabled=False)
    range_trader = RangeTrader()
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
    asian_range_quota = AsianRangeQuota.load()
    phase1a_breaker = Phase1aCircuitBreaker()

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

        try:
            # 1. Price
            tick = mt5.symbol_info_tick("XAUUSD")
            if tick is None:
                print("  WARN: no tick data")
                continue
            price = tick.bid
            spread = tick.ask - tick.bid
            print(f"  XAUUSD: ${price:.2f} (spread ${spread:.2f})")

            # 2. Fetch data
            data = fetch_mt5_data()
            bars_info = ", ".join(f"{k}: {len(v)}" for k, v in data.items())
            print(f"  Data: {{{bars_info}}}")

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
            regime = classify_regime(data.get(Timeframe.D1))
            print(f"  Regime: {regime}")

            # 4b. Detect H1 SMC snapshot + ATR for range detection
            h1_df = data.get(Timeframe.H1)
            h1_snapshot = None
            if h1_df is not None and len(h1_df) > 0:
                h1_snapshot = detector.detect(h1_df, Timeframe.H1)
            m15_snapshot = None
            m15_df = data.get(Timeframe.M15)
            if m15_df is not None and len(m15_df) > 0:
                m15_snapshot = detector.detect(m15_df, Timeframe.M15)
            h1_atr = aggregator._compute_h1_atr(h1_df)

            # 5. Strategy (v1 trending setups — always generated)
            setups = aggregator.generate_setups(data, price)
            print(f"  SMC Setups: {len(setups)}")

            # 6. Dual-mode action routing
            session, _ = get_session_info()
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
            )
            if action.startswith("RANGE") and session in _ASIAN_SESSIONS:
                asian_range_quota = asian_range_quota.record_open(datetime.now(tz=timezone.utc))
            # TODO: When paper/live trade-close tracking is implemented, call
            # phase1a_breaker.record_trade_close(pnl_usd) after each
            # ASIAN_LONDON_TRANSITION ranging trade closes.

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
                # 用户期望 "装 MT5 = 真实交易", 实际 MT5 0 positions. Fix:
                # SMC_MT5_EXECUTE=1 env var 开启真实 order_send (TMGM Demo 安全).
                _mt5_execute = os.environ.get("SMC_MT5_EXECUTE", "0") == "1"
                _mt5_ticket: int | None = None
                _mt5_send_retcode: int | None = None
                _mt5_mode_tag = "PAPER"
                # Asian ranging 用 0.3x multiplier (Phase1a 降档协议), 其他 1.0x.
                # base_lot=0.01 XAUUSD minimum. margin 公式分两步避免 bug:
                #   notional_usd = entry × contract_size × lots  (1 XAUUSD lot = 100 oz)
                #   margin_usd   = notional / leverage           (100:1 simplification)
                # 旧写法 entry*lots*100/100 数值正确但语义混淆 (100 既是 contract
                # size 又似 leverage). PAPER mode, live execution 时 MT5 按账户重算.
                # Round 4.6-J-fix: 用 _ASIAN_SESSIONS frozenset 避免 inline tuple drift.
                XAUUSD_CONTRACT_SIZE_OZ = 100
                PAPER_LEVERAGE_RATIO = 100  # 100:1 typical retail XAUUSD
                lot_multiplier = 0.3 if session in _ASIAN_SESSIONS else 1.0
                base_lot = 0.01
                position_size_lots = round(base_lot * lot_multiplier, 4)
                # Round 4.6-Y (USER CATCH): MT5 min lot = 0.01 broker-wide, 0.003
                # 会被 reject (retcode 10014 Invalid volume). Clamp Asian-reduced
                # lot back to MT5 min. Asian risk management 改由 quota (1/day)
                # 和 CircuitBreaker 承担, 不靠 fractional lot.
                _MT5_MIN_LOT = 0.01
                if position_size_lots < _MT5_MIN_LOT:
                    position_size_lots = _MT5_MIN_LOT
                notional_value_usd = (
                    best.entry_price * XAUUSD_CONTRACT_SIZE_OZ * position_size_lots
                )
                margin_used_estimate_usd = round(
                    notional_value_usd / PAPER_LEVERAGE_RATIO, 2
                )

                # Round 4.6-X: actual MT5 order_send when SMC_MT5_EXECUTE=1
                if _mt5_execute:
                    try:
                        tick = mt5.symbol_info_tick("XAUUSD")
                        current_tick_price = tick.ask if best.direction == "long" else tick.bid
                        request = {
                            "action": mt5.TRADE_ACTION_DEAL,
                            "symbol": "XAUUSD",
                            "volume": position_size_lots,
                            "type": mt5.ORDER_TYPE_BUY if best.direction == "long" else mt5.ORDER_TYPE_SELL,
                            "price": current_tick_price,
                            "sl": best.stop_loss,
                            "tp": best.take_profit,
                            "deviation": 20,
                            "magic": 19760418,
                            "comment": f"AI-SMC {best.trigger[:15]}",
                            "type_time": mt5.ORDER_TIME_GTC,
                            "type_filling": mt5.ORDER_FILLING_IOC,
                        }
                        result = mt5.order_send(request)
                        _mt5_send_retcode = result.retcode if result else None
                        _mt5_ticket = result.order if result else None
                        if _mt5_send_retcode == mt5.TRADE_RETCODE_DONE:
                            _mt5_mode_tag = "LIVE_EXEC"
                            print(f"  [MT5] Order sent ✓ ticket={_mt5_ticket}")
                        else:
                            _mt5_mode_tag = f"MT5_FAIL_{_mt5_send_retcode}"
                            print(f"  [MT5] Order FAIL retcode={_mt5_send_retcode}")
                    except Exception as exc:
                        _mt5_mode_tag = f"MT5_EXC_{type(exc).__name__}"
                        print(f"  [MT5] Exception: {exc}")

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
                    "trading_mode": mode.mode,
                    "session": session,
                    # Round 4.6-J position sizing fields (4.6-J-fix: notional separated)
                    "lot_multiplier": lot_multiplier,
                    "position_size_lots": position_size_lots,
                    "notional_value_usd": round(notional_value_usd, 2),
                    "margin_used_estimate_usd": margin_used_estimate_usd,
                    "paper_leverage_ratio": PAPER_LEVERAGE_RATIO,
                }
                with open(JOURNAL_PATH, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")
            else:
                # Trending (v1 5-gate) path: iterate setups as before
                for s in setups:
                    e = s.entry_signal
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
                    }
                    with open(JOURNAL_PATH, "a") as f:
                        f.write(json.dumps(log_entry) + "\n")

            # 8. Save state for dashboard (dual-mode aware)
            save_state(cycle, price, action, reason, ai_analysis, regime, setups,
                       best, mode_decision=mode, range_trader=range_trader,
                       aggregator=aggregator)

        except Exception as exc:
            print(f"  ERROR: {exc}")
            import traceback
            traceback.print_exc()

    mt5.shutdown()
    print(f"\n[{datetime.now()}] AI-SMC Live Trading Loop stopped.")


if __name__ == "__main__":
    main()

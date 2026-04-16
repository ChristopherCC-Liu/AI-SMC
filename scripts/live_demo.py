"""AI-SMC Live Trading Loop — with AI analysis integration.

Runs every M15 bar close:
1. Fetch XAUUSD data from MT5
2. Run AI direction analysis (Claude debate or SMA fallback)
3. Run SMC strategy pipeline
4. Output clear BUY / SELL / HOLD signal
5. Save state to data/live_state.json for dashboard

Usage:
    python scripts/live_demo.py
"""
import sys
import os
import time
import signal
import json

os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import MetaTrader5 as mt5
import polars as pl
from datetime import datetime, timezone
from pathlib import Path
from smc.data.schemas import Timeframe
from smc.smc_core.detector import SMCDetector
from smc.strategy.aggregator import MultiTimeframeAggregator
from smc.strategy.regime import classify_regime
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
        engine = DirectionEngine()
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
    if 8 <= hour < 13:
        return "LONDON", 0.0
    elif 13 <= hour < 16:
        return "LONDON/NY OVERLAP", 0.0
    elif 16 <= hour < 21:
        return "NEW YORK", 0.0
    elif 21 <= hour < 24:
        return "LATE NY", 0.1
    else:
        return "ASIAN", 0.2


def determine_action(setups, ai_analysis, regime):
    """AI HARD GATE: determine BUY / SELL / HOLD.

    AI is NOT decoration — it controls whether trades happen:
    - Session filter: Asian session = BLOCKED (PF 0.69 proven loser)
    - Direction filter: AI must agree with SMC direction
    - Confidence gate: AI confidence < 0.5 after session penalty = BLOCKED
    - AI NEUTRAL = HOLD (no trade without AI clarity)
    """
    session, session_penalty = get_session_info()
    ai_dir = ai_analysis.get("ai_direction", ai_analysis.get("direction", "neutral"))
    ai_conf = ai_analysis.get("ai_confidence", ai_analysis.get("confidence", 0.3))
    effective_conf = ai_conf - session_penalty

    # GATE 1: Session filter (HARD — Asian session proven PF 0.69)
    if session == "ASIAN":
        return "HOLD", f"SESSION BLOCKED: {session} (PF 0.69 historically) | AI conf {ai_conf:.0%} - {session_penalty:.0%} = {effective_conf:.0%}", None

    # GATE 2: AI confidence after session penalty
    if effective_conf < 0.5:
        return "HOLD", f"LOW CONFIDENCE: AI {ai_dir.upper()} {ai_conf:.0%} - {session_penalty:.0%} penalty = {effective_conf:.0%} (need 50%+)", None

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
        # ALIGNED — execute
        action = "BUY" if e.direction == "long" else "SELL"
        lot = "0.01" if effective_conf >= 0.8 else ("0.005" if effective_conf >= 0.6 else "0.0025")
        reason = (f"AI {ai_dir.upper()} ({effective_conf:.0%}) + SMC {e.direction.upper()} "
                  f"| {e.trigger_type} | conf {best.confluence_score:.2f} | lot {lot} | {session}")
        return action, reason, best
    else:
        # CONFLICT — check soft override
        if ai_conf < 0.65:
            # Low-confidence conflict: AI uncertain, let SMC speak (quarter lot)
            action = f"{'BUY' if e.direction == 'long' else 'SELL'} (override)"
            reason = (f"SOFT OVERRIDE: AI {ai_dir.upper()} ({ai_conf:.0%}) vs SMC {e.direction.upper()} "
                      f"| AI uncertain → SMC wins at 1/4 lot | {e.trigger_type}")
            return action, reason, best
        else:
            # High-confidence conflict: AI wins
            return "HOLD", f"AI BLOCKED: AI {ai_dir.upper()} ({ai_conf:.0%}) vs SMC {e.direction.upper()} | AI confident → no trade", None


def save_state(cycle, price, action, reason, ai_analysis, regime, setups, best_setup):
    """Save live state for dashboard display."""
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
    }
    if best_setup:
        e = best_setup.entry_signal
        state["best_setup"] = {
            "direction": e.direction,
            "entry": e.entry_price,
            "sl": e.stop_loss,
            "tp1": e.take_profit_1,
            "trigger": e.trigger_type,
            "confluence": best_setup.confluence_score,
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
    print("  Strategy:   v1 (PF 1.66, validated)")
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

            # 5. Strategy
            setups = aggregator.generate_setups(data, price)
            print(f"  SMC Setups: {len(setups)}")

            # 6. Action
            action, reason, best = determine_action(setups, ai_analysis, regime)

            # Display action prominently
            action_colors = {"BUY": "+++", "SELL": "---", "HOLD": "==="}
            marker = action_colors.get(action.split()[0], "???")
            print()
            print(f"  {marker} ACTION: {action} {marker}")
            print(f"  {reason}")

            if best:
                e = best.entry_signal
                print(f"  Entry: ${e.entry_price:.2f} | SL: ${e.stop_loss:.2f} | TP: ${e.take_profit_1:.2f}")

            # 7. Journal
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
                }
                with open(JOURNAL_PATH, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")

            # 8. Save state for dashboard
            save_state(cycle, price, action, reason, ai_analysis, regime, setups, best)

        except Exception as exc:
            print(f"  ERROR: {exc}")
            import traceback
            traceback.print_exc()

    mt5.shutdown()
    print(f"\n[{datetime.now()}] AI-SMC Live Trading Loop stopped.")


if __name__ == "__main__":
    main()

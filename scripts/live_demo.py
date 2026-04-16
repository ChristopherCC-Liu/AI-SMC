"""AI-SMC Live Demo Loop — simplified standalone launcher.

Usage on VPS:
    cd C:\AI-SMC
    .venv\Scripts\python.exe scripts/live_demo.py

Logs setups to data/journal/live_trades.jsonl (PAPER mode, no real orders).
"""
import sys
import os
import time
import signal
import json

os.environ["NO_COLOR"] = "1"
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


def main():
    print(f"[{datetime.now()}] AI-SMC Live Demo Starting...")
    print("Mode: DEMO (paper, log only)")
    print("Instrument: XAUUSD")
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

    journal_path = Path("data/journal/live_trades.jsonl")
    journal_path.parent.mkdir(parents=True, exist_ok=True)

    running = True

    def stop(sig, frame):
        nonlocal running
        print(f"\n[{datetime.now()}] Shutdown signal received.")
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    cycle = 0
    while running:
        cycle += 1
        now = datetime.now(timezone.utc)

        nxt = next_bar_close(Timeframe.M15, now)
        wait = (nxt - now).total_seconds()

        if wait > 0:
            print(f"[{now.strftime('%H:%M:%S')} UTC] Cycle {cycle}: waiting {wait:.0f}s -> next M15 close {nxt.strftime('%H:%M')}")
            while wait > 0 and running:
                time.sleep(min(10, wait))
                wait -= 10

        if not running:
            break

        now = datetime.now(timezone.utc)
        print(f"[{now.strftime('%H:%M:%S')} UTC] Cycle {cycle}: processing...")

        try:
            tick = mt5.symbol_info_tick("XAUUSD")
            if tick is None:
                print("  WARN: no tick")
                continue
            price = tick.bid

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

            bars_info = ", ".join(f"{k}: {len(v)}" for k, v in data.items())
            print(f"  Data: {{{bars_info}}}")
            print(f"  Price: ${price:.2f}")

            setups = aggregator.generate_setups(data, price)
            regime = classify_regime(data.get(Timeframe.D1))
            print(f"  Regime: {regime} | Setups: {len(setups)}")

            for s in setups:
                e = s.entry_signal
                log_entry = {
                    "time": now.isoformat(),
                    "cycle": cycle,
                    "price": price,
                    "direction": e.direction,
                    "entry": e.entry_price,
                    "sl": e.stop_loss,
                    "tp1": e.take_profit_1,
                    "trigger": e.trigger_type,
                    "confluence": round(s.confluence_score, 3),
                    "regime": regime,
                    "mode": "PAPER",
                }
                with open(journal_path, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")
                print(f"  >>> SETUP: {e.direction} @ {e.entry_price:.2f} | SL {e.stop_loss:.2f} | TP {e.take_profit_1:.2f} | {e.trigger_type} | conf {s.confluence_score:.3f}")

            if not setups:
                print("  No setups.")

        except Exception as exc:
            print(f"  ERROR: {exc}")

    mt5.shutdown()
    print(f"[{datetime.now()}] AI-SMC Live Demo stopped.")


if __name__ == "__main__":
    main()

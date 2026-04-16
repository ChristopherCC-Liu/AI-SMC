"""AI-SMC Market Analysis — runs AI direction debate and saves result.

Usage:
    python scripts/ai_analyze.py              # uses Claude CLI or API
    python scripts/ai_analyze.py --fallback   # SMA50 fallback only (no LLM)

Saves result to data/ai_analysis.json for dashboard display.
"""
import sys
import os
import json
from datetime import datetime, timezone
from pathlib import Path

os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import MetaTrader5 as mt5
import polars as pl

OUTPUT_PATH = Path("data/ai_analysis.json")


def fetch_data():
    """Fetch latest H4 + D1 data from MT5."""
    mt5.initialize()

    result = {}
    for label, mt5_tf in [("D1", mt5.TIMEFRAME_D1), ("H4", mt5.TIMEFRAME_H4)]:
        rates = mt5.copy_rates_from_pos("XAUUSD", mt5_tf, 0, 100)
        if rates is not None and len(rates) > 0:
            df = pl.DataFrame({
                "ts": [datetime.fromtimestamp(r[0], tz=timezone.utc) for r in rates],
                "open": [float(r[1]) for r in rates],
                "high": [float(r[2]) for r in rates],
                "low": [float(r[3]) for r in rates],
                "close": [float(r[4]) for r in rates],
                "volume": [float(r[5]) for r in rates],
            }).with_columns(pl.col("ts").cast(pl.Datetime("ns", "UTC")))
            result[label] = df

    tick = mt5.symbol_info_tick("XAUUSD")
    result["price"] = tick.bid if tick else 0
    result["spread"] = (tick.ask - tick.bid) if tick else 0

    return result


def compute_sma_analysis(data):
    """SMA-based technical analysis (always available, no LLM)."""
    analysis = {"source": "technical", "assessed_at": datetime.now(timezone.utc).isoformat()}

    if "D1" in data:
        closes = data["D1"]["close"].to_list()

        # SMA 20/50
        if len(closes) >= 50:
            sma20 = sum(closes[-20:]) / 20
            sma50 = sum(closes[-50:]) / 50
            price = closes[-1]

            analysis["sma20"] = round(sma20, 2)
            analysis["sma50"] = round(sma50, 2)
            analysis["price_vs_sma20"] = "above" if price > sma20 else "below"
            analysis["price_vs_sma50"] = "above" if price > sma50 else "below"
            analysis["sma20_vs_sma50"] = "golden_cross" if sma20 > sma50 else "death_cross"

            # Trend strength
            pct_above_sma50 = (price - sma50) / sma50 * 100
            analysis["trend_strength_pct"] = round(pct_above_sma50, 2)

            if price > sma20 > sma50:
                analysis["direction"] = "bullish"
                analysis["confidence"] = min(0.8, 0.5 + abs(pct_above_sma50) / 20)
                analysis["reasoning"] = f"Price ${price:.0f} above SMA20 (${sma20:.0f}) above SMA50 (${sma50:.0f}). Strong uptrend. Distance from SMA50: {pct_above_sma50:+.1f}%"
            elif price < sma20 < sma50:
                analysis["direction"] = "bearish"
                analysis["confidence"] = min(0.8, 0.5 + abs(pct_above_sma50) / 20)
                analysis["reasoning"] = f"Price ${price:.0f} below SMA20 (${sma20:.0f}) below SMA50 (${sma50:.0f}). Strong downtrend. Distance from SMA50: {pct_above_sma50:+.1f}%"
            else:
                analysis["direction"] = "neutral"
                analysis["confidence"] = 0.3
                analysis["reasoning"] = f"Mixed signals. Price ${price:.0f}, SMA20=${sma20:.0f}, SMA50=${sma50:.0f}. No clear trend."

        # ATR volatility
        if len(closes) >= 15:
            highs = data["D1"]["high"].to_list()
            lows = data["D1"]["low"].to_list()
            trs = []
            for i in range(1, min(15, len(closes))):
                tr = max(highs[-(i)] - lows[-(i)],
                         abs(highs[-(i)] - closes[-(i+1)]),
                         abs(lows[-(i)] - closes[-(i+1)]))
                trs.append(tr)
            atr14 = sum(trs) / len(trs)
            atr_pct = atr14 / closes[-1] * 100
            analysis["atr14"] = round(atr14, 2)
            analysis["atr_pct"] = round(atr_pct, 2)
            analysis["volatility"] = "high" if atr_pct > 1.4 else ("low" if atr_pct < 1.0 else "normal")

        # Recent price action
        if len(closes) >= 5:
            change_5d = (closes[-1] - closes[-5]) / closes[-5] * 100
            analysis["change_5d_pct"] = round(change_5d, 2)
            analysis["momentum"] = "up" if change_5d > 0.5 else ("down" if change_5d < -0.5 else "flat")

    # H4 structure
    if "H4" in data:
        h4_closes = data["H4"]["close"].to_list()
        if len(h4_closes) >= 20:
            h4_sma20 = sum(h4_closes[-20:]) / 20
            analysis["h4_trend"] = "up" if h4_closes[-1] > h4_sma20 else "down"
            analysis["h4_price"] = round(h4_closes[-1], 2)
            analysis["h4_sma20"] = round(h4_sma20, 2)

    # Key levels
    if "D1" in data and len(data["D1"]) >= 20:
        highs = data["D1"]["high"].to_list()[-20:]
        lows = data["D1"]["low"].to_list()[-20:]
        analysis["resistance_20d"] = round(max(highs), 2)
        analysis["support_20d"] = round(min(lows), 2)

    analysis["key_factors"] = []
    if analysis.get("direction") == "bullish":
        analysis["key_factors"].append("SMA20 > SMA50 (golden cross)")
        if analysis.get("momentum") == "up":
            analysis["key_factors"].append(f"5D momentum +{analysis.get('change_5d_pct', 0):.1f}%")
        if analysis.get("h4_trend") == "up":
            analysis["key_factors"].append("H4 trend aligned (up)")
    elif analysis.get("direction") == "bearish":
        analysis["key_factors"].append("SMA20 < SMA50 (death cross)")
        if analysis.get("momentum") == "down":
            analysis["key_factors"].append(f"5D momentum {analysis.get('change_5d_pct', 0):.1f}%")
        if analysis.get("h4_trend") == "down":
            analysis["key_factors"].append("H4 trend aligned (down)")

    return analysis


def try_ai_debate(data, analysis):
    """Try to enhance analysis with AI debate (Claude CLI or API)."""
    try:
        from smc.ai.direction_engine import DirectionEngine
        engine = DirectionEngine()
        h4_df = data.get("H4")
        ai_dir = engine.get_direction(h4_df=h4_df)

        if ai_dir.source != "neutral_default":
            analysis["ai_direction"] = ai_dir.direction
            analysis["ai_confidence"] = round(ai_dir.confidence, 3)
            analysis["ai_reasoning"] = ai_dir.reasoning
            analysis["ai_source"] = ai_dir.source
            analysis["ai_key_drivers"] = list(ai_dir.key_drivers) if ai_dir.key_drivers else []
            analysis["source"] = f"technical + {ai_dir.source}"
    except Exception as e:
        analysis["ai_error"] = str(e)

    return analysis


def main():
    fallback_only = "--fallback" in sys.argv

    data = fetch_data()
    analysis = compute_sma_analysis(data)

    if not fallback_only:
        analysis = try_ai_debate(data, analysis)

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(analysis, f, indent=2, default=str)

    # Print summary
    d = analysis.get("direction", "unknown")
    c = analysis.get("confidence", 0)
    r = analysis.get("reasoning", "")
    print(f"Direction: {d.upper()} (confidence {c:.0%})")
    print(f"Reasoning: {r}")
    if analysis.get("ai_direction"):
        print(f"AI Override: {analysis['ai_direction'].upper()} ({analysis['ai_source']})")
    print(f"Saved to {OUTPUT_PATH}")

    mt5.shutdown()


if __name__ == "__main__":
    main()

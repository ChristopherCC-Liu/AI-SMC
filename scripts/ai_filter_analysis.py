"""
Retroactive AI Filter Analysis
===============================
Applies SMA50 direction filter (proxy for AI directional bias) to v1's
historical trades and quantifies the impact.

Logic:
  - For each trade, compute SMA50 at the trade open date using D1 close data.
  - SMA direction = bullish if current SMA50 > SMA50 from 5 days ago, else bearish.
  - ALIGNED: long trade + bullish SMA, or short trade + bearish SMA.
  - CONFLICT: long trade + bearish SMA, or short trade + bullish SMA.

Output: data/ai_filter_analysis.json
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

# ──────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────
PROJECT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT / "data"
D1_DIR = DATA_DIR / "parquet" / "XAUUSD" / "D1"
TRADES_FILE = DATA_DIR / "gate_v10_v1.jsonl"
OUTPUT_FILE = DATA_DIR / "ai_filter_analysis.json"


def load_d1_data() -> pd.DataFrame:
    """Load all D1 parquet files into a single DataFrame sorted by timestamp."""
    frames: list[pd.DataFrame] = []
    for year_dir in sorted(D1_DIR.iterdir()):
        if not year_dir.is_dir():
            continue
        for pq_file in sorted(year_dir.glob("*.parquet")):
            frames.append(pd.read_parquet(pq_file, columns=["ts", "close"]))
    df = pd.concat(frames, ignore_index=True).sort_values("ts").reset_index(drop=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def compute_sma(df: pd.DataFrame, period: int = 50) -> pd.DataFrame:
    """Add SMA column and SMA direction (bullish/bearish) to D1 data."""
    result = df.copy()
    result["sma50"] = result["close"].rolling(window=period, min_periods=period).mean()
    # Direction: compare current SMA to SMA 5 bars ago
    result["sma50_prev"] = result["sma50"].shift(5)
    result["sma_direction"] = "neutral"
    result.loc[result["sma50"] > result["sma50_prev"], "sma_direction"] = "bullish"
    result.loc[result["sma50"] < result["sma50_prev"], "sma_direction"] = "bearish"
    return result


def get_sma_direction_at(d1: pd.DataFrame, trade_ts: str) -> str:
    """Find the SMA direction on or before the trade open timestamp."""
    ts = pd.Timestamp(trade_ts)
    # Get the most recent D1 bar on or before the trade open
    mask = d1["ts"] <= ts
    if not mask.any():
        return "unknown"
    row = d1.loc[mask].iloc[-1]
    return str(row["sma_direction"])


def classify_session(trade_ts: str) -> str:
    """Classify trade into session: Asian, London, NewYork based on UTC hour."""
    hour = pd.Timestamp(trade_ts).hour
    if 0 <= hour < 8:
        return "Asian"
    elif 8 <= hour < 14:
        return "London"
    else:
        return "NewYork"


def safe_pf(wins_pnl: float, losses_pnl: float) -> float:
    """Compute profit factor safely (avoid division by zero)."""
    if losses_pnl == 0:
        return float("inf") if wins_pnl > 0 else 0.0
    return abs(wins_pnl / losses_pnl)


def compute_group_stats(trades: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute WR, PF, total PnL for a group of trades."""
    n = len(trades)
    if n == 0:
        return {"n": 0, "wr": 0.0, "pf": 0.0, "pnl": 0.0, "avg_pnl": 0.0}
    wins = [t for t in trades if t["pnl_usd"] > 0]
    losses = [t for t in trades if t["pnl_usd"] <= 0]
    total_pnl = sum(t["pnl_usd"] for t in trades)
    win_pnl = sum(t["pnl_usd"] for t in wins)
    loss_pnl = sum(t["pnl_usd"] for t in losses)
    wr = len(wins) / n
    pf = safe_pf(win_pnl, loss_pnl)

    return {
        "n": n,
        "wr": round(wr, 4),
        "pf": round(pf, 4) if not math.isinf(pf) else "Infinity",
        "pnl": round(total_pnl, 2),
        "avg_pnl": round(total_pnl / n, 2),
    }


def main() -> None:
    print("Loading D1 data...")
    d1 = load_d1_data()
    print(f"  D1 bars: {len(d1)} ({d1['ts'].min()} to {d1['ts'].max()})")

    d1 = compute_sma(d1, period=50)
    print(f"  SMA50 available from: {d1.dropna(subset=['sma50']).iloc[0]['ts']}")

    # ── Load trades ──────────────────────────────────────────
    print("\nLoading v1 trades...")
    all_trades: list[dict[str, Any]] = []
    with TRADES_FILE.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            window = json.loads(line)
            for trade in window.get("trade_details", []):
                trade["window"] = window["window"]
                trade["test_start"] = window["test_start"]
                trade["test_end"] = window["test_end"]
                all_trades.append(trade)

    print(f"  Total trades extracted: {len(all_trades)}")

    # ── Classify each trade ──────────────────────────────────
    aligned_trades: list[dict[str, Any]] = []
    conflict_trades: list[dict[str, Any]] = []
    unknown_trades: list[dict[str, Any]] = []

    for trade in all_trades:
        sma_dir = get_sma_direction_at(d1, trade["open_ts"])
        trade["sma_direction"] = sma_dir
        trade["session"] = classify_session(trade["open_ts"])

        direction = trade["direction"]
        if sma_dir == "unknown" or sma_dir == "neutral":
            unknown_trades.append(trade)
        elif (direction == "long" and sma_dir == "bullish") or \
             (direction == "short" and sma_dir == "bearish"):
            trade["alignment"] = "aligned"
            aligned_trades.append(trade)
        else:
            trade["alignment"] = "conflict"
            conflict_trades.append(trade)

    print(f"\n  Aligned:  {len(aligned_trades)}")
    print(f"  Conflict: {len(conflict_trades)}")
    print(f"  Unknown:  {len(unknown_trades)}")

    # ── Compute stats ────────────────────────────────────────
    all_stats = compute_group_stats(all_trades)
    aligned_stats = compute_group_stats(aligned_trades)
    conflict_stats = compute_group_stats(conflict_trades)

    # ── Filter improvement ───────────────────────────────────
    pf_all = all_stats["pf"] if all_stats["pf"] != "Infinity" else 999
    pf_aligned = aligned_stats["pf"] if aligned_stats["pf"] != "Infinity" else 999
    pf_delta = round(pf_aligned - pf_all, 4) if not isinstance(pf_aligned, str) and not isinstance(pf_all, str) else "N/A"

    wr_delta = round(aligned_stats["wr"] - all_stats["wr"], 4)
    trades_removed = conflict_stats["n"]

    # ── Verdict ──────────────────────────────────────────────
    if isinstance(pf_delta, (int, float)) and pf_delta > 0.1 and wr_delta > 0.02:
        verdict = "AI_FILTER_HELPS"
    elif isinstance(pf_delta, (int, float)) and pf_delta < -0.1:
        verdict = "AI_FILTER_HURTS"
    else:
        verdict = "AI_FILTER_NEUTRAL"

    # ── Losing trade selectivity ─────────────────────────────
    all_losers = [t for t in all_trades if t["pnl_usd"] <= 0]
    conflict_losers = [t for t in conflict_trades if t["pnl_usd"] <= 0]
    loser_selectivity = round(len(conflict_losers) / len(all_losers), 4) if all_losers else 0.0

    # ── Trade count per year ─────────────────────────────────
    # Approximate: v1 covers ~3.5 years (2021-01 to 2024-04)
    years_by_trade: dict[str, int] = {}
    years_by_aligned: dict[str, int] = {}
    for t in all_trades:
        yr = pd.Timestamp(t["open_ts"]).year
        years_by_trade[str(yr)] = years_by_trade.get(str(yr), 0) + 1
    for t in aligned_trades:
        yr = pd.Timestamp(t["open_ts"]).year
        years_by_aligned[str(yr)] = years_by_aligned.get(str(yr), 0) + 1

    total_years = len(set(pd.Timestamp(t["open_ts"]).year for t in all_trades))
    trades_per_year_all = round(len(all_trades) / max(total_years, 1), 1)
    trades_per_year_filtered = round(len(aligned_trades) / max(total_years, 1), 1)

    # ── Session analysis ─────────────────────────────────────
    session_stats: dict[str, Any] = {}
    for session_name in ["Asian", "London", "NewYork"]:
        session_trades = [t for t in all_trades if t["session"] == session_name]
        session_stats[session_name] = compute_group_stats(session_trades)

    # ── Print results ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("AI FILTER RETROACTIVE ANALYSIS")
    print("=" * 60)

    print(f"\n{'Group':<12} {'N':>4}  {'WR':>6}  {'PF':>8}  {'PnL':>8}  {'Avg':>7}")
    print("-" * 50)
    for label, stats in [("All", all_stats), ("Aligned", aligned_stats), ("Conflict", conflict_stats)]:
        pf_str = f"{stats['pf']:.2f}" if isinstance(stats['pf'], (int, float)) else stats['pf']
        print(f"{label:<12} {stats['n']:>4}  {stats['wr']:>6.1%}  {pf_str:>8}  {stats['pnl']:>8.2f}  {stats['avg_pnl']:>7.2f}")

    print(f"\nFilter Improvement:")
    print(f"  PF delta:        {pf_delta}")
    print(f"  WR delta:        {wr_delta:+.2%}")
    print(f"  Trades removed:  {trades_removed}")
    print(f"  Loser selectivity (% of losers in conflict): {loser_selectivity:.1%}")

    print(f"\nTrades per year:")
    print(f"  Without filter: {trades_per_year_all}")
    print(f"  With filter:    {trades_per_year_filtered}")
    print(f"  By year (all):  {years_by_trade}")
    print(f"  By year (filt): {years_by_aligned}")

    print(f"\nSession Analysis:")
    for s, st in session_stats.items():
        pf_str = f"{st['pf']:.2f}" if isinstance(st['pf'], (int, float)) else st['pf']
        print(f"  {s:<10} N={st['n']:>3}  WR={st['wr']:>6.1%}  PF={pf_str:>6}  PnL={st['pnl']:>8.2f}")

    print(f"\nVERDICT: {verdict}")

    # ── Detailed trade-level breakdown (for debugging) ───────
    print("\n" + "-" * 60)
    print("Trade-level detail:")
    print(f"{'Open Time':<28} {'Dir':<6} {'SMA':<9} {'Align':<9} {'PnL':>8} {'Session':<10} {'W'}")
    print("-" * 90)
    for t in sorted(all_trades, key=lambda x: x["open_ts"]):
        alignment = t.get("alignment", "unknown")
        print(f"{t['open_ts']:<28} {t['direction']:<6} {t.get('sma_direction','?'):<9} "
              f"{alignment:<9} {t['pnl_usd']:>8.2f} {t['session']:<10} {t['window']}")

    # ── Save JSON output ─────────────────────────────────────
    output = {
        "all_trades": all_stats,
        "aligned": aligned_stats,
        "conflict": conflict_stats,
        "filter_improvement": {
            "pf_delta": pf_delta,
            "wr_delta": wr_delta,
            "trades_removed": trades_removed,
            "loser_selectivity": loser_selectivity,
        },
        "trades_per_year": {
            "without_filter": trades_per_year_all,
            "with_filter": trades_per_year_filtered,
            "by_year_all": years_by_trade,
            "by_year_filtered": years_by_aligned,
        },
        "session_analysis": session_stats,
        "verdict": verdict,
        "methodology": {
            "sma_period": 50,
            "sma_direction_lookback": 5,
            "aligned_def": "long+bullish OR short+bearish",
            "conflict_def": "long+bearish OR short+bullish",
            "data_source": "D1 XAUUSD parquet, SMA50 slope over 5-bar lookback",
        },
        "trade_details": [
            {
                "open_ts": t["open_ts"],
                "direction": t["direction"],
                "sma_direction": t.get("sma_direction", "unknown"),
                "alignment": t.get("alignment", "unknown"),
                "pnl_usd": t["pnl_usd"],
                "close_reason": t["close_reason"],
                "session": t["session"],
                "window": t["window"],
            }
            for t in sorted(all_trades, key=lambda x: x["open_ts"])
        ],
    }

    OUTPUT_FILE.write_text(json.dumps(output, indent=2, ensure_ascii=False, default=str))
    print(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

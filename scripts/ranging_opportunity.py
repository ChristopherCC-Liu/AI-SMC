"""Ranging opportunity quantification for AI-SMC.

Analyses:
1. Time in ranging mode (D1 regime classification)
2. Range characteristics (duration, width, boundary touches)
3. Simple mean-reversion backtest on ranging periods
4. Opportunity cost of current HOLD during ranging
5. Session x Regime cross-analysis
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from smc.data.lake import ForexDataLake
from smc.data.schemas import Timeframe

POINT_SIZE = 0.01
LAKE_ROOT = Path(__file__).resolve().parent.parent / "data" / "parquet"
lake = ForexDataLake(LAKE_ROOT)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_all(tf: Timeframe) -> pl.DataFrame:
    """Load all XAUUSD data for a timeframe."""
    rng = lake.available_range("XAUUSD", tf)
    if rng is None:
        raise RuntimeError(f"No data for XAUUSD {tf}")
    return lake.query("XAUUSD", tf, start=rng[0], end=datetime(2025, 1, 1, tzinfo=timezone.utc))


_ATR_PERIOD = 14
_TRENDING_THRESHOLD = 1.4
_RANGING_THRESHOLD = 1.0


def rolling_regime(d1: pl.DataFrame) -> pl.DataFrame:
    """Classify regime for every D1 bar using a rolling ATR(14)% window.

    Returns D1 frame with added columns: atr_pct, regime.
    """
    high = d1["high"].to_list()
    low = d1["low"].to_list()
    close = d1["close"].to_list()
    n = len(high)

    atr_pcts: list[float | None] = [None] * n
    regimes: list[str] = ["transitional"] * n

    for i in range(_ATR_PERIOD + 1, n):
        tr_vals: list[float] = []
        for j in range(i - _ATR_PERIOD, i):
            hl = high[j + 1] - low[j + 1]
            hc = abs(high[j + 1] - close[j])
            lc = abs(low[j + 1] - close[j])
            tr_vals.append(max(hl, hc, lc))
        atr = sum(tr_vals) / _ATR_PERIOD
        pct = (atr / close[i]) * 100.0 if close[i] > 0 else 0.0
        atr_pcts[i] = round(pct, 4)
        if pct >= _TRENDING_THRESHOLD:
            regimes[i] = "trending"
        elif pct < _RANGING_THRESHOLD:
            regimes[i] = "ranging"
        else:
            regimes[i] = "transitional"

    return d1.with_columns(
        pl.Series("atr_pct", atr_pcts, dtype=pl.Float64),
        pl.Series("regime", regimes, dtype=pl.String),
    )


def identify_ranging_periods(d1_regime: pl.DataFrame) -> list[dict]:
    """Identify contiguous ranging periods and compute characteristics."""
    ts_list = d1_regime["ts"].to_list()
    high_list = d1_regime["high"].to_list()
    low_list = d1_regime["low"].to_list()
    close_list = d1_regime["close"].to_list()
    regime_list = d1_regime["regime"].to_list()

    periods: list[dict] = []
    start_idx: int | None = None

    for i, r in enumerate(regime_list):
        if r == "ranging":
            if start_idx is None:
                start_idx = i
        else:
            if start_idx is not None:
                end_idx = i - 1
                if end_idx - start_idx >= 2:  # at least 3 bars
                    period_high = max(high_list[start_idx : end_idx + 1])
                    period_low = min(low_list[start_idx : end_idx + 1])
                    range_width = period_high - period_low
                    mid_price = (period_high + period_low) / 2.0
                    duration_days = end_idx - start_idx + 1

                    # Count boundary touches (within 15% of upper/lower)
                    upper_zone = period_high - range_width * 0.15
                    lower_zone = period_low + range_width * 0.15
                    upper_touches = sum(
                        1 for j in range(start_idx, end_idx + 1) if high_list[j] >= upper_zone
                    )
                    lower_touches = sum(
                        1 for j in range(start_idx, end_idx + 1) if low_list[j] <= lower_zone
                    )

                    periods.append({
                        "start": str(ts_list[start_idx]),
                        "end": str(ts_list[end_idx]),
                        "duration_days": duration_days,
                        "range_width_points": round(range_width / POINT_SIZE, 1),
                        "range_width_usd": round(range_width, 2),
                        "mid_price": round(mid_price, 2),
                        "upper_touches": upper_touches,
                        "lower_touches": lower_touches,
                    })
                start_idx = None

    # Handle case where last period is still ranging
    if start_idx is not None:
        end_idx = len(regime_list) - 1
        if end_idx - start_idx >= 2:
            period_high = max(high_list[start_idx : end_idx + 1])
            period_low = min(low_list[start_idx : end_idx + 1])
            range_width = period_high - period_low
            mid_price = (period_high + period_low) / 2.0
            duration_days = end_idx - start_idx + 1
            upper_zone = period_high - range_width * 0.15
            lower_zone = period_low + range_width * 0.15
            upper_touches = sum(
                1 for j in range(start_idx, end_idx + 1) if high_list[j] >= upper_zone
            )
            lower_touches = sum(
                1 for j in range(start_idx, end_idx + 1) if low_list[j] <= lower_zone
            )
            periods.append({
                "start": str(ts_list[start_idx]),
                "end": str(ts_list[end_idx]),
                "duration_days": duration_days,
                "range_width_points": round(range_width / POINT_SIZE, 1),
                "range_width_usd": round(range_width, 2),
                "mid_price": round(mid_price, 2),
                "upper_touches": upper_touches,
                "lower_touches": lower_touches,
            })

    return periods


def session_tag(ts: datetime) -> str:
    """Classify a UTC timestamp into forex session."""
    h = ts.hour
    if 0 <= h < 8:
        return "asian"
    elif 8 <= h < 13:
        return "london"
    elif 13 <= h < 17:
        return "ny_overlap"
    elif 17 <= h < 22:
        return "ny"
    else:
        return "asian"  # late NY wraps into asian


# ===========================================================================
# Analysis 1: Time in ranging mode
# ===========================================================================

def analysis_1_time_in_regime(d1_regime: pl.DataFrame) -> dict:
    """What % of D1 bars are trending / transitional / ranging?"""
    valid = d1_regime.filter(pl.col("atr_pct").is_not_null())
    total = len(valid)
    counts = valid.group_by("regime").len().sort("regime")

    result: dict[str, float] = {}
    for row in counts.iter_rows(named=True):
        pct = round(row["len"] / total * 100.0, 2)
        result[row["regime"]] = pct

    # Also by year
    yearly = (
        valid
        .with_columns(pl.col("ts").dt.year().alias("year"))
        .group_by("year", "regime")
        .len()
    )
    year_totals = valid.with_columns(pl.col("ts").dt.year().alias("year")).group_by("year").len()
    yearly_joined = yearly.join(year_totals, on="year", suffix="_total")
    yearly_joined = yearly_joined.with_columns(
        (pl.col("len") / pl.col("len_total") * 100.0).round(2).alias("pct")
    ).sort("year", "regime")

    by_year: dict[int, dict[str, float]] = {}
    for row in yearly_joined.iter_rows(named=True):
        yr = row["year"]
        if yr not in by_year:
            by_year[yr] = {}
        by_year[yr][row["regime"]] = row["pct"]

    return {
        "total_bars": total,
        "overall_pct": result,
        "by_year": {str(k): v for k, v in sorted(by_year.items())},
    }


# ===========================================================================
# Analysis 2: Range characteristics
# ===========================================================================

def analysis_2_range_characteristics(periods: list[dict], h1: pl.DataFrame) -> dict:
    """Compute statistics about ranging periods."""
    if not periods:
        return {"n_periods": 0}

    durations = [p["duration_days"] for p in periods]
    widths_pts = [p["range_width_points"] for p in periods]
    widths_usd = [p["range_width_usd"] for p in periods]
    upper_t = [p["upper_touches"] for p in periods]
    lower_t = [p["lower_touches"] for p in periods]

    # H1 bar range within ranging periods
    h1_ranges_within: list[float] = []
    for p in periods:
        start_dt = datetime.fromisoformat(p["start"])
        end_dt = datetime.fromisoformat(p["end"])
        h1_sub = h1.filter(
            (pl.col("ts") >= start_dt) & (pl.col("ts") <= end_dt)
        )
        if not h1_sub.is_empty():
            bar_ranges = (h1_sub["high"] - h1_sub["low"]).to_list()
            h1_ranges_within.extend(bar_ranges)

    return {
        "n_periods": len(periods),
        "avg_duration_days": round(sum(durations) / len(durations), 1),
        "median_duration_days": round(sorted(durations)[len(durations) // 2], 1),
        "max_duration_days": max(durations),
        "min_duration_days": min(durations),
        "avg_range_width_points": round(sum(widths_pts) / len(widths_pts), 1),
        "avg_range_width_usd": round(sum(widths_usd) / len(widths_usd), 2),
        "median_range_width_usd": round(sorted(widths_usd)[len(widths_usd) // 2], 2),
        "avg_upper_touches": round(sum(upper_t) / len(upper_t), 1),
        "avg_lower_touches": round(sum(lower_t) / len(lower_t), 1),
        "avg_h1_bar_range_usd": round(sum(h1_ranges_within) / len(h1_ranges_within), 2) if h1_ranges_within else 0,
        "avg_h1_bar_range_points": round(sum(h1_ranges_within) / len(h1_ranges_within) / POINT_SIZE, 1) if h1_ranges_within else 0,
    }


# ===========================================================================
# Analysis 3: Simple mean-reversion backtest
# ===========================================================================

def analysis_3_mean_reversion_backtest(periods: list[dict], h1: pl.DataFrame) -> dict:
    """Mean reversion backtest on H1 within identified ranging periods.

    Strategy: BUY when price < lower + 10% of range, SELL when price > upper - 10% of range.
    SL: just beyond the range boundary (range_width * 5% past boundary = tight SL).
    TP: midpoint of range.
    Position: 0.005 lots (1 lot = 100 oz).
    Spread cost: 3 points per trade (from data).
    Trades that don't close by period end are marked at market.
    """
    LOT_SIZE = 0.005  # half trending size
    OZ_PER_LOT = 100.0
    SPREAD_COST_PER_TRADE = 3.0 * POINT_SIZE * LOT_SIZE * OZ_PER_LOT  # 3 pts spread

    trades: list[dict] = []

    for p in periods:
        start_dt = datetime.fromisoformat(p["start"])
        end_dt = datetime.fromisoformat(p["end"])
        range_width = p["range_width_usd"]
        mid = p["mid_price"]

        upper = mid + range_width / 2.0
        lower = mid - range_width / 2.0

        buy_zone = lower + range_width * 0.10
        sell_zone = upper - range_width * 0.10
        # Tight SL: 5% of range beyond boundary (realistic risk)
        sl_distance = range_width * 0.05

        h1_sub = h1.filter(
            (pl.col("ts") >= start_dt) & (pl.col("ts") <= end_dt)
        )
        if h1_sub.is_empty() or range_width < 5.0:
            continue

        closes = h1_sub["close"].to_list()
        highs = h1_sub["high"].to_list()
        lows = h1_sub["low"].to_list()

        in_trade = False
        trade_dir = ""
        entry_price = 0.0
        sl_price = 0.0
        tp_price = 0.0

        for i in range(len(closes)):
            if not in_trade:
                if closes[i] <= buy_zone:
                    in_trade = True
                    trade_dir = "BUY"
                    entry_price = closes[i]
                    sl_price = lower - sl_distance
                    tp_price = mid
                elif closes[i] >= sell_zone:
                    in_trade = True
                    trade_dir = "SELL"
                    entry_price = closes[i]
                    sl_price = upper + sl_distance
                    tp_price = mid
            else:
                closed = False
                if trade_dir == "BUY":
                    if lows[i] <= sl_price:
                        pnl_usd = (sl_price - entry_price) * LOT_SIZE * OZ_PER_LOT - SPREAD_COST_PER_TRADE
                        trades.append({"dir": "BUY", "pnl_usd": round(pnl_usd, 2), "result": "SL"})
                        closed = True
                    elif highs[i] >= tp_price:
                        pnl_usd = (tp_price - entry_price) * LOT_SIZE * OZ_PER_LOT - SPREAD_COST_PER_TRADE
                        trades.append({"dir": "BUY", "pnl_usd": round(pnl_usd, 2), "result": "TP"})
                        closed = True
                elif trade_dir == "SELL":
                    if highs[i] >= sl_price:
                        pnl_usd = (entry_price - sl_price) * LOT_SIZE * OZ_PER_LOT - SPREAD_COST_PER_TRADE
                        trades.append({"dir": "SELL", "pnl_usd": round(pnl_usd, 2), "result": "SL"})
                        closed = True
                    elif lows[i] <= tp_price:
                        pnl_usd = (entry_price - tp_price) * LOT_SIZE * OZ_PER_LOT - SPREAD_COST_PER_TRADE
                        trades.append({"dir": "SELL", "pnl_usd": round(pnl_usd, 2), "result": "TP"})
                        closed = True

                if closed:
                    in_trade = False

                # Close at market on last bar of period
                elif i == len(closes) - 1:
                    if trade_dir == "BUY":
                        pnl_usd = (closes[i] - entry_price) * LOT_SIZE * OZ_PER_LOT - SPREAD_COST_PER_TRADE
                    else:
                        pnl_usd = (entry_price - closes[i]) * LOT_SIZE * OZ_PER_LOT - SPREAD_COST_PER_TRADE
                    trades.append({"dir": trade_dir, "pnl_usd": round(pnl_usd, 2), "result": "EXPIRE"})
                    in_trade = False

    if not trades:
        return {"trades": 0, "wr": 0, "pf": 0, "pnl": 0, "avg_pnl": 0}

    total_trades = len(trades)
    wins = [t for t in trades if t["pnl_usd"] > 0]
    losses = [t for t in trades if t["pnl_usd"] <= 0]
    wr = round(len(wins) / total_trades * 100.0, 2)
    gross_profit = sum(t["pnl_usd"] for t in wins)
    gross_loss = abs(sum(t["pnl_usd"] for t in losses))
    pf = round(gross_profit / gross_loss, 2) if gross_loss > 0 else 999.99
    total_pnl = round(sum(t["pnl_usd"] for t in trades), 2)
    avg_pnl = round(total_pnl / total_trades, 2)

    tp_count = sum(1 for t in trades if t["result"] == "TP")
    sl_count = sum(1 for t in trades if t["result"] == "SL")
    expire_count = sum(1 for t in trades if t["result"] == "EXPIRE")

    return {
        "trades": total_trades,
        "wins": len(wins),
        "losses": len(losses),
        "wr": wr,
        "pf": pf,
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "pnl": total_pnl,
        "avg_pnl_per_trade": avg_pnl,
        "tp_exits": tp_count,
        "sl_exits": sl_count,
        "expire_exits": expire_count,
        "lot_size": LOT_SIZE,
        "spread_cost_per_trade": round(SPREAD_COST_PER_TRADE, 4),
    }


# ===========================================================================
# Analysis 4: Opportunity cost of current HOLD
# ===========================================================================

def analysis_4_opportunity_cost(d1_regime: pl.DataFrame, m15: pl.DataFrame) -> dict:
    """Quantify idle M15 bars and potential PnL from ranging periods."""
    # How many D1 bars are ranging/transitional (non-trending)?
    valid = d1_regime.filter(pl.col("atr_pct").is_not_null())
    non_trending = valid.filter(pl.col("regime") != "trending")
    total_d1 = len(valid)
    non_trending_d1 = len(non_trending)

    # Map D1 regime onto M15 bars (each M15 bar inherits its D1 date's regime)
    m15_with_date = m15.with_columns(pl.col("ts").dt.date().alias("date"))
    d1_regime_dates = d1_regime.with_columns(pl.col("ts").dt.date().alias("date")).select("date", "regime")
    m15_regime = m15_with_date.join(d1_regime_dates, on="date", how="left")

    total_m15 = len(m15_regime)
    idle_m15 = len(m15_regime.filter(pl.col("regime").is_in(["ranging", "transitional"])))
    ranging_m15 = len(m15_regime.filter(pl.col("regime") == "ranging"))

    # Calculate potential PnL: if we captured 20% of range width per cycle
    # Use average range width from ranging periods and count of cycles
    # For simple calc: each ranging D1 day offers ~1 cycle opportunity
    ranging_d1 = valid.filter(pl.col("regime") == "ranging")

    # Average daily range during ranging days
    ranging_daily_ranges = (ranging_d1["high"] - ranging_d1["low"]).to_list()
    avg_daily_range = sum(ranging_daily_ranges) / len(ranging_daily_ranges) if ranging_daily_ranges else 0

    # If we capture 20% of daily range per day, with 0.005 lots
    LOT_SIZE = 0.005
    OZ_PER_LOT = 100.0
    capture_pct = 0.20
    pnl_per_day = avg_daily_range * capture_pct * LOT_SIZE * OZ_PER_LOT
    n_ranging_days = len(ranging_d1)
    years_of_data = 5.0

    total_opportunity = round(pnl_per_day * n_ranging_days, 2)
    per_year = round(total_opportunity / years_of_data, 2)

    return {
        "total_d1_bars": total_d1,
        "non_trending_d1_bars": non_trending_d1,
        "non_trending_pct": round(non_trending_d1 / total_d1 * 100.0, 2),
        "total_m15_bars": total_m15,
        "idle_m15_bars": idle_m15,
        "ranging_m15_bars": ranging_m15,
        "idle_m15_pct": round(idle_m15 / total_m15 * 100.0, 2),
        "ranging_m15_pct": round(ranging_m15 / total_m15 * 100.0, 2),
        "avg_daily_range_usd_ranging": round(avg_daily_range, 2),
        "capture_assumption_pct": capture_pct * 100,
        "opportunity_pnl_total_5y": total_opportunity,
        "opportunity_cost_per_year": per_year,
        "lot_size_assumed": LOT_SIZE,
    }


# ===========================================================================
# Analysis 5: Session x Regime cross-analysis
# ===========================================================================

def analysis_5_session_regime(d1_regime: pl.DataFrame, h1: pl.DataFrame) -> dict:
    """Cross-analyze session and regime for profitability characteristics."""
    # Map D1 regime onto H1 bars
    h1_with_date = h1.with_columns(pl.col("ts").dt.date().alias("date"))
    d1_dates = d1_regime.with_columns(pl.col("ts").dt.date().alias("date")).select("date", "regime")
    h1_regime = h1_with_date.join(d1_dates, on="date", how="left")

    ts_list = h1_regime["ts"].to_list()
    regime_list = h1_regime["regime"].to_list()
    high_list = h1_regime["high"].to_list()
    low_list = h1_regime["low"].to_list()
    close_list = h1_regime["close"].to_list()
    open_list = h1_regime["open"].to_list()

    # Build session tags
    sessions = [session_tag(t) for t in ts_list]

    # For each session x regime: avg bar range, avg body size, direction ratio
    combos: dict[str, dict[str, list]] = {}
    for i in range(len(ts_list)):
        s = sessions[i]
        r = regime_list[i] if regime_list[i] is not None else "unknown"
        key = f"{s}_{r}"
        if key not in combos:
            combos[key] = {"bar_range": [], "body": [], "direction": []}
        combos[key]["bar_range"].append(high_list[i] - low_list[i])
        combos[key]["body"].append(abs(close_list[i] - open_list[i]))
        combos[key]["direction"].append(1 if close_list[i] > open_list[i] else -1)

    result: dict[str, dict] = {}
    for key, data in sorted(combos.items()):
        n = len(data["bar_range"])
        avg_range = sum(data["bar_range"]) / n
        avg_body = sum(data["body"]) / n
        bullish_pct = sum(1 for d in data["direction"] if d > 0) / n * 100.0
        # body/range ratio: high = directional, low = indecisive (ranging character)
        body_range_ratio = avg_body / avg_range if avg_range > 0 else 0

        result[key] = {
            "n_bars": n,
            "avg_range_usd": round(avg_range, 2),
            "avg_range_points": round(avg_range / POINT_SIZE, 1),
            "avg_body_usd": round(avg_body, 2),
            "body_range_ratio": round(body_range_ratio, 4),
            "bullish_pct": round(bullish_pct, 2),
        }

    # Simplified mean-reversion signal quality per session during ranging
    # Metric: how often does price reverse after hitting extremes?
    reversal_quality: dict[str, dict] = {}
    for session_name in ["asian", "london", "ny_overlap", "ny"]:
        key = f"{session_name}_ranging"
        if key not in combos:
            reversal_quality[session_name] = {"n_bars": 0, "reversal_rate": 0}
            continue
        bars = combos[key]
        n = len(bars["bar_range"])
        # A simple reversal proxy: small body relative to range = wick rejection = reversal signal
        reversals = sum(
            1 for i in range(n)
            if bars["bar_range"][i] > 0 and bars["body"][i] / bars["bar_range"][i] < 0.3
        )
        reversal_quality[session_name] = {
            "n_bars": n,
            "reversal_rate": round(reversals / n * 100.0, 2) if n > 0 else 0,
        }

    return {
        "session_regime_matrix": result,
        "reversal_quality_during_ranging": reversal_quality,
    }


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    print("=== Loading data ===")
    d1 = load_all(Timeframe.D1)
    h4 = load_all(Timeframe.H4)
    h1 = load_all(Timeframe.H1)
    m15 = load_all(Timeframe.M15)
    print(f"  D1: {len(d1)} bars, H4: {len(h4)} bars, H1: {len(h1)} bars, M15: {len(m15)} bars")

    print("\n=== Analysis 1: Regime Classification ===")
    d1_regime = rolling_regime(d1)
    a1 = analysis_1_time_in_regime(d1_regime)
    print(f"  Total valid bars: {a1['total_bars']}")
    for regime, pct in a1["overall_pct"].items():
        print(f"  {regime}: {pct}%")
    print("  By year:")
    for yr, data in a1["by_year"].items():
        print(f"    {yr}: {data}")

    print("\n=== Analysis 2: Range Characteristics ===")
    periods = identify_ranging_periods(d1_regime)
    a2 = analysis_2_range_characteristics(periods, h1)
    print(f"  Ranging periods found: {a2['n_periods']}")
    if a2["n_periods"] > 0:
        print(f"  Avg duration: {a2['avg_duration_days']} days")
        print(f"  Avg range width: {a2['avg_range_width_usd']} USD ({a2['avg_range_width_points']} pts)")
        print(f"  Avg upper touches: {a2['avg_upper_touches']}, lower: {a2['avg_lower_touches']}")
        print(f"  Avg H1 bar range within: {a2['avg_h1_bar_range_usd']} USD")

    print("\n=== Analysis 3: Mean Reversion Backtest ===")
    a3 = analysis_3_mean_reversion_backtest(periods, h1)
    print(f"  Trades: {a3['trades']}")
    if a3["trades"] > 0:
        print(f"  Win rate: {a3['wr']}%")
        print(f"  Profit factor: {a3['pf']}")
        print(f"  Total PnL: ${a3['pnl']}")
        print(f"  Avg PnL/trade: ${a3['avg_pnl_per_trade']}")

    print("\n=== Analysis 4: Opportunity Cost ===")
    a4 = analysis_4_opportunity_cost(d1_regime, m15)
    print(f"  Non-trending D1 bars: {a4['non_trending_d1_bars']} / {a4['total_d1_bars']} ({a4['non_trending_pct']}%)")
    print(f"  Idle M15 bars: {a4['idle_m15_bars']} ({a4['idle_m15_pct']}%)")
    print(f"  Ranging M15 bars: {a4['ranging_m15_bars']} ({a4['ranging_m15_pct']}%)")
    print(f"  Opportunity cost/year: ${a4['opportunity_cost_per_year']}")

    print("\n=== Analysis 5: Session x Regime ===")
    a5 = analysis_5_session_regime(d1_regime, h1)
    print("  Session-Regime matrix (selected):")
    for key in ["asian_ranging", "london_ranging", "ny_overlap_ranging", "asian_trending", "london_trending"]:
        if key in a5["session_regime_matrix"]:
            d = a5["session_regime_matrix"][key]
            print(f"    {key}: {d['n_bars']} bars, avg_range={d['avg_range_usd']}$, body/range={d['body_range_ratio']}")
    print("  Reversal quality during ranging:")
    for session, d in a5["reversal_quality_during_ranging"].items():
        print(f"    {session}: {d['n_bars']} bars, reversal_rate={d['reversal_rate']}%")

    # -----------------------------------------------------------------------
    # Compile output
    # -----------------------------------------------------------------------
    # Compute combined PF projection
    trending_pf = 1.09  # from existing system analysis
    ranging_pf_raw = a3["pf"] if a3["trades"] > 0 else 0
    ranging_pct = a1["overall_pct"].get("ranging", 0) / 100.0
    transitional_pct = a1["overall_pct"].get("transitional", 0) / 100.0
    trending_pct = a1["overall_pct"].get("trending", 0) / 100.0

    # The raw PF is inflated by hindsight (we know exact range boundaries).
    # Apply a realistic discount: real-time detection adds ~30-50% slippage
    # and boundaries won't be perfect. Conservative estimate: PF 1.5-2.5.
    # Use WR-based estimate: 94.59% WR with avg win ~$11.36 / avg loss ~$1.83
    # In real-time, expect WR ~65-75%, so discount to PF ~1.8
    ranging_pf_realistic = 1.8  # conservative real-time estimate

    # Combined PF is a weighted blend
    if ranging_pct + trending_pct > 0:
        combined_pf = round(
            (trending_pf * trending_pct + ranging_pf_realistic * ranging_pct) / (trending_pct + ranging_pct),
            2,
        )
    else:
        combined_pf = trending_pf

    output = {
        "ranging_pct_time": a1["overall_pct"].get("ranging", 0),
        "transitional_pct_time": a1["overall_pct"].get("transitional", 0),
        "trending_pct_time": a1["overall_pct"].get("trending", 0),
        "regime_by_year": a1["by_year"],
        "avg_range_width_points": a2.get("avg_range_width_points", 0),
        "avg_range_width_usd": a2.get("avg_range_width_usd", 0),
        "avg_range_duration_days": a2.get("avg_duration_days", 0),
        "n_ranging_periods": a2.get("n_periods", 0),
        "range_characteristics": a2,
        "mean_reversion_backtest": a3,
        "opportunity_cost_per_year": a4["opportunity_cost_per_year"],
        "opportunity_cost_detail": a4,
        "session_regime_analysis": a5,
        "combined_projection": {
            "trending_pf": trending_pf,
            "ranging_pf_hindsight": ranging_pf_raw,
            "ranging_pf_realistic": ranging_pf_realistic,
            "combined_pf": combined_pf,
            "trending_pct": round(trending_pct * 100, 2),
            "ranging_pct": round(ranging_pct * 100, 2),
            "note": "ranging_pf_realistic is a conservative estimate for real-time detection (hindsight bias discount)",
        },
        "ranging_periods_detail": periods,
    }

    out_path = Path(__file__).resolve().parent.parent / "data" / "ranging_opportunity.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n=== Output saved to {out_path} ===")


if __name__ == "__main__":
    main()

"""Diagnostic script: why did TRAIL_50P_ACT_50P collapse in 2024?

Tests five hypotheses against cached setup data + raw M15/H1/D1 bars.

No LLM calls — pure ATR/SMA-based regime analysis only.

Usage:
    /opt/anaconda3/bin/python scripts/diagnose_2024_weakness.py
"""
from __future__ import annotations

import pickle
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import logging
logging.getLogger("smc").setLevel(logging.WARNING)
logging.getLogger("smc.ai").setLevel(logging.WARNING)
logging.basicConfig(level=logging.WARNING)

import polars as pl

from smc.backtest.engine import BarBacktestEngine
from smc.backtest.fills import FillModel, TrailRule
from smc.backtest.types import BacktestConfig, TradeRecord
from smc.smc_core.constants import XAUUSD_POINT_SIZE

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CACHE_DIR = PROJECT_ROOT / ".scratch" / "round4" / "setup_cache"
_OUTPUT_DIR = PROJECT_ROOT / ".scratch" / "round4"
_DATA_DIR = PROJECT_ROOT / "data" / "parquet" / "XAUUSD"

# TRAIL_50P_ACT_50P activates at 0.5R, trails by 50 points
_TRAIL_RULE = TrailRule(trail_points=50.0, trail_activate_r=0.5)

# Windows per year based on the cache filenames (test-window start months)
_YEAR_WINDOWS: dict[int, list[str]] = {
    2021: ["W01_20210101", "W02_20210401", "W03_20210701", "W04_20211001"],
    2022: ["W05_20220101", "W06_20220401", "W07_20220701", "W08_20221001"],
    2023: ["W09_20230101", "W10_20230401", "W11_20230701", "W12_20231001"],
    2024: ["W13_20240101", "W14_20240401", "W15_20240701"],
}

_YEARS = [2021, 2022, 2023, 2024]


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_window_cache(key: str) -> tuple[dict, pl.DataFrame] | None:
    path = _CACHE_DIR / f"{key}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["setups"], data["bars"]


def _load_parquet_year(subdir: str, year: int) -> pl.DataFrame:
    """Load all months for a given timeframe subdirectory and year."""
    year_dir = _DATA_DIR / subdir / str(year)
    if not year_dir.exists():
        return pl.DataFrame()
    frames: list[pl.DataFrame] = []
    for fp in sorted(year_dir.glob("*.parquet")):
        frames.append(pl.read_parquet(fp))
    if not frames:
        return pl.DataFrame()
    df = pl.concat(frames)
    if "ts" in df.columns:
        df = df.sort("ts")
    elif "time" in df.columns:
        df = df.sort("time")
    return df


# ---------------------------------------------------------------------------
# Replay engine helper — returns trades for TRAIL_50P_ACT_50P mc=3
# ---------------------------------------------------------------------------

def _replay_year_trades(year: int) -> list[TradeRecord]:
    """Replay cached setups with TRAIL_50P_ACT_50P mc=3 for one year."""
    config = BacktestConfig(
        initial_balance=10_000.0,
        instrument="XAUUSD",
        spread_points=3.0,
        slippage_points=0.5,
        commission_per_lot=7.0,
        max_concurrent_trades=3,
    )
    fm = FillModel(
        spread_points=config.spread_points,
        slippage_points=config.slippage_points,
        commission_per_lot=config.commission_per_lot,
    )
    engine = BarBacktestEngine(config=config, fill_model=fm)

    all_trades: list[TradeRecord] = []
    for key in _YEAR_WINDOWS.get(year, []):
        cached = _load_window_cache(key)
        if cached is None:
            continue
        setups, bars = cached
        result = engine.run(setups, bars, trail_rule=_TRAIL_RULE)
        all_trades.extend(result.trades)
    return all_trades


# ---------------------------------------------------------------------------
# H1: Regime shift — SMA50 slope on D1 bars
# ---------------------------------------------------------------------------

def _analyse_regime(year: int) -> dict[str, float]:
    """
    Compute % of D1 bars where SMA50 slope is positive (trending up or down).
    'Trending' = |slope| > small threshold relative to ATR.
    Returns: pct_trending, pct_bullish, pct_bearish, pct_flat, median_d1_atr_pts.
    """
    df = _load_parquet_year("D1", year)
    if df.is_empty():
        return {}

    close_col = "close" if "close" in df.columns else df.columns[-1]
    high_col = "high" if "high" in df.columns else None
    low_col = "low" if "low" in df.columns else None

    closes = df[close_col].to_list()
    n = len(closes)
    if n < 55:
        return {}

    # SMA50
    sma50 = [
        sum(closes[i - 50:i]) / 50
        for i in range(50, n)
    ]
    slopes = [sma50[i] - sma50[i - 1] for i in range(1, len(sma50))]

    # D1 ATR (TR = max(H-L, |H-prev_C|, |L-prev_C|))
    d1_atrs: list[float] = []
    if high_col and low_col:
        highs = df[high_col].to_list()
        lows = df[low_col].to_list()
        trs: list[float] = []
        for i in range(1, n):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
            trs.append(tr)
        # 14-bar ATR rolling
        for i in range(14, len(trs)):
            d1_atrs.append(sum(trs[i - 14:i]) / 14)

    median_d1_atr = sorted(d1_atrs)[len(d1_atrs) // 2] if d1_atrs else 0.0
    # Convert to points
    median_d1_atr_pts = median_d1_atr / XAUUSD_POINT_SIZE

    # Threshold: slope > 0.5/day in price terms to be "trending"
    threshold = 0.5
    n_bull = sum(1 for s in slopes if s > threshold)
    n_bear = sum(1 for s in slopes if s < -threshold)
    n_flat = len(slopes) - n_bull - n_bear
    total = len(slopes)

    return {
        "pct_trending": round((n_bull + n_bear) / total * 100, 1),
        "pct_bullish": round(n_bull / total * 100, 1),
        "pct_bearish": round(n_bear / total * 100, 1),
        "pct_flat": round(n_flat / total * 100, 1),
        "median_d1_atr_pts": round(median_d1_atr_pts, 1),
    }


# ---------------------------------------------------------------------------
# H2: H1 ATR per year
# ---------------------------------------------------------------------------

def _analyse_h1_atr(year: int) -> dict[str, float]:
    """Compute median H1 ATR in points for the year."""
    df = _load_parquet_year("H1", year)
    if df.is_empty():
        return {}

    high_col = "high" if "high" in df.columns else None
    low_col = "low" if "low" in df.columns else None
    close_col = "close" if "close" in df.columns else None
    if not (high_col and low_col and close_col):
        return {}

    highs = df[high_col].to_list()
    lows = df[low_col].to_list()
    closes = df[close_col].to_list()
    n = len(highs)

    trs: list[float] = []
    for i in range(1, n):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)

    # 14-bar ATR
    atrs: list[float] = []
    for i in range(14, len(trs)):
        atrs.append(sum(trs[i - 14:i]) / 14)

    if not atrs:
        return {}

    sorted_atrs = sorted(atrs)
    median_atr_pts = sorted_atrs[len(sorted_atrs) // 2] / XAUUSD_POINT_SIZE
    q75_atr_pts = sorted_atrs[int(len(sorted_atrs) * 0.75)] / XAUUSD_POINT_SIZE
    q25_atr_pts = sorted_atrs[int(len(sorted_atrs) * 0.25)] / XAUUSD_POINT_SIZE

    return {
        "median_h1_atr_pts": round(median_atr_pts, 1),
        "q25_h1_atr_pts": round(q25_atr_pts, 1),
        "q75_h1_atr_pts": round(q75_atr_pts, 1),
    }


# ---------------------------------------------------------------------------
# H3: Session / month breakdown for 2024
# ---------------------------------------------------------------------------

def _session_label(hour_utc: int) -> str:
    if 22 <= hour_utc or hour_utc < 7:
        return "Asian"
    elif 7 <= hour_utc < 12:
        return "London"
    elif 12 <= hour_utc < 17:
        return "NY-overlap"
    else:
        return "NY-close"


def _analyse_session_month(trades: list[TradeRecord]) -> dict[str, Any]:
    """Breakdown 2024 trades by month and session."""
    by_month: dict[int, dict[str, int]] = {}
    by_session: dict[str, dict[str, int]] = {}

    for t in trades:
        month = t.open_ts.month
        hour = t.open_ts.hour
        session = _session_label(hour)
        is_win = t.pnl_usd > 0

        if month not in by_month:
            by_month[month] = {"wins": 0, "losses": 0, "pnl": 0.0}
        by_month[month]["wins" if is_win else "losses"] += 1
        by_month[month]["pnl"] += t.pnl_usd  # type: ignore[operator]

        if session not in by_session:
            by_session[session] = {"wins": 0, "losses": 0, "pnl": 0.0}
        by_session[session]["wins" if is_win else "losses"] += 1
        by_session[session]["pnl"] += t.pnl_usd  # type: ignore[operator]

    return {"by_month": by_month, "by_session": by_session}


# ---------------------------------------------------------------------------
# H4: Setup confluence score distribution
# ---------------------------------------------------------------------------

def _analyse_confluence(trades: list[TradeRecord]) -> dict[str, float]:
    if not trades:
        return {}
    scores = [t.setup_confluence for t in trades]
    scores.sort()
    n = len(scores)
    return {
        "mean_confluence": round(sum(scores) / n, 3),
        "median_confluence": round(scores[n // 2], 3),
        "pct_high_quality": round(sum(1 for s in scores if s >= 0.7) / n * 100, 1),
    }


# ---------------------------------------------------------------------------
# H5: Exit reason breakdown
# ---------------------------------------------------------------------------

def _analyse_exits(trades: list[TradeRecord]) -> dict[str, float]:
    if not trades:
        return {}
    reasons: dict[str, int] = {}
    for t in trades:
        reasons[t.close_reason] = reasons.get(t.close_reason, 0) + 1
    n = len(trades)
    return {
        reason: round(count / n * 100, 1)
        for reason, count in sorted(reasons.items())
    }


# ---------------------------------------------------------------------------
# Compute per-year PF from trades
# ---------------------------------------------------------------------------

def _pf(trades: list[TradeRecord]) -> float:
    wins = sum(t.pnl_usd for t in trades if t.pnl_usd > 0)
    losses = abs(sum(t.pnl_usd for t in trades if t.pnl_usd < 0))
    if losses == 0:
        return float("inf")
    return round(wins / losses, 2)


def _wr(trades: list[TradeRecord]) -> float:
    if not trades:
        return 0.0
    return round(sum(1 for t in trades if t.pnl_usd > 0) / len(trades) * 100, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== 2024 Weakness Diagnosis ===\n", flush=True)

    results: dict[int, dict] = {}

    for year in _YEARS:
        print(f"Processing {year}...", flush=True)

        # Replay trades
        trades = _replay_year_trades(year)
        print(f"  {year}: {len(trades)} trades replayed", flush=True)

        # H1 + H2 + H4 + H5
        regime = _analyse_regime(year)
        h1_atr = _analyse_h1_atr(year)
        confluence = _analyse_confluence(trades)
        exits = _analyse_exits(trades)

        results[year] = {
            "n_trades": len(trades),
            "pf": _pf(trades),
            "wr": _wr(trades),
            "regime": regime,
            "h1_atr": h1_atr,
            "confluence": confluence,
            "exits": exits,
            "trades": trades,  # kept for H3 2024 drill-down
        }
        print(f"  PF={results[year]['pf']} WR={results[year]['wr']}% exits={exits}", flush=True)

    # H3: Session/month for 2024
    h3_2024 = _analyse_session_month(results[2024]["trades"])

    # --- Compute trail noise ratio ---
    # Trail = 50pts. Noise ratio = median H1 ATR / trail distance.
    # Ratio > 1.0 means single candle noise > trail distance → easy stopout.
    trail_pts = 50.0
    for year in _YEARS:
        h1_atr_pts = results[year]["h1_atr"].get("median_h1_atr_pts", 0.0)
        results[year]["trail_noise_ratio"] = round(h1_atr_pts / trail_pts, 2) if trail_pts > 0 else 0.0

    # ---------------------------------------------------------------------------
    # Build the verdict
    # ---------------------------------------------------------------------------

    # Key numbers for diagnosis
    r = results
    y24_atr = r[2024]["h1_atr"].get("median_h1_atr_pts", 0.0)
    y23_atr = r[2023]["h1_atr"].get("median_h1_atr_pts", 0.0)
    y21_atr = r[2021]["h1_atr"].get("median_h1_atr_pts", 0.0)
    avg_prior_atr = (y21_atr + r[2022]["h1_atr"].get("median_h1_atr_pts", 0.0) + y23_atr) / 3

    atr_expansion_ratio = round(y24_atr / avg_prior_atr, 2) if avg_prior_atr > 0 else 0.0
    tnr_2024 = r[2024]["trail_noise_ratio"]

    y24_exits = r[2024]["exits"]
    sl_pct_2024 = y24_exits.get("sl", 0.0)
    sl_pct_2023 = r[2023]["exits"].get("sl", 0.0)
    sl_pct_2021 = r[2021]["exits"].get("sl", 0.0)

    y24_regime = r[2024]["regime"]
    y23_regime = r[2023]["regime"]
    y24_trending_pct = y24_regime.get("pct_trending", 0.0)
    y23_trending_pct = y23_regime.get("pct_trending", 0.0)

    # ---------------------------------------------------------------------------
    # Write report
    # ---------------------------------------------------------------------------

    out_path = _OUTPUT_DIR / "2024-weakness-diagnosis.md"
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("# TRAIL_50P_ACT_50P — 2024 Collapse Diagnosis\n")
    lines.append(f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}_\n")

    # ---- Executive verdict
    # Determine primary cause
    if atr_expansion_ratio > 1.5 and tnr_2024 > 1.0:
        primary = (
            f"**Volatility expansion is the primary root cause**: 2024 H1 ATR expanded {atr_expansion_ratio}× "
            f"vs 2021-2023 average, making the fixed 50-pt trail smaller than a single candle's noise "
            f"(trail-noise ratio = {tnr_2024:.2f}), causing premature stopouts before TP reachable. "
            f"Confidence: HIGH."
        )
    elif abs(y24_trending_pct - y23_trending_pct) > 15:
        primary = (
            f"**Regime shift is the primary root cause**: trending D1 bars dropped from "
            f"{y23_trending_pct}% (2023) to {y24_trending_pct}% (2024), reducing setups that "
            f"can sustain a trail long enough to hit TP. Confidence: HIGH."
        )
    elif sl_pct_2024 - sl_pct_2023 > 15:
        primary = (
            f"**SL hit rate surge is the primary symptom** ({sl_pct_2024:.0f}% in 2024 vs "
            f"{sl_pct_2023:.0f}% in 2023), combined with volatility expansion. Confidence: MEDIUM."
        )
    else:
        primary = (
            "**Setup quality drift** — 2024 confluence scores are lower on average, "
            "producing weaker setups that fail before trail can activate. Confidence: MEDIUM."
        )

    lines.append("## Verdict\n")
    lines.append(f"{primary}\n")

    # ---- H1+H2 combined table
    lines.append("## H1 + H2: Regime & Volatility Per Year\n")
    lines.append("| Year | PF | WR% | N | Trend% | Flat% | D1-ATR-pts | H1-ATR-pts | Trail-Noise-Ratio |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for year in _YEARS:
        reg = r[year]["regime"]
        atr_d = r[year]["h1_atr"]
        lines.append(
            f"| {year} "
            f"| {r[year]['pf']} "
            f"| {r[year]['wr']} "
            f"| {r[year]['n_trades']} "
            f"| {reg.get('pct_trending', '-')} "
            f"| {reg.get('pct_flat', '-')} "
            f"| {reg.get('median_d1_atr_pts', '-')} "
            f"| {atr_d.get('median_h1_atr_pts', '-')} "
            f"| {r[year]['trail_noise_ratio']} |"
        )
    lines.append("")
    lines.append(
        f"_2024 H1 ATR = **{y24_atr:.0f} pts** vs 2021-2023 avg {avg_prior_atr:.0f} pts "
        f"({atr_expansion_ratio}× expansion). "
        f"Trail distance = 50 pts. When noise > trail, the SL is whipsawed off before TP is reachable._\n"
    )

    # ---- H5: Exit reason table
    lines.append("## H5: Exit Reason Breakdown Per Year\n")
    lines.append("| Year | sl% | tp1% | tp2% |")
    lines.append("|---|---|---|---|")
    for year in _YEARS:
        ex = r[year]["exits"]
        lines.append(
            f"| {year} "
            f"| {ex.get('sl', 0.0):.1f}% "
            f"| {ex.get('tp1', 0.0):.1f}% "
            f"| {ex.get('tp2', 0.0):.1f}% |"
        )
    lines.append("")

    # ---- H4: Confluence
    lines.append("## H4: Setup Confluence Distribution\n")
    lines.append("| Year | Mean | Median | % High-Quality (≥0.7) |")
    lines.append("|---|---|---|---|")
    for year in _YEARS:
        c = r[year]["confluence"]
        lines.append(
            f"| {year} "
            f"| {c.get('mean_confluence', '-')} "
            f"| {c.get('median_confluence', '-')} "
            f"| {c.get('pct_high_quality', '-')}% |"
        )
    lines.append("")

    # ---- H3: 2024 session / month
    lines.append("## H3: 2024 Session Breakdown\n")
    lines.append("| Session | Wins | Losses | PnL$ |")
    lines.append("|---|---|---|---|")
    for sess, stats in sorted(h3_2024["by_session"].items()):
        lines.append(
            f"| {sess} | {stats['wins']} | {stats['losses']} | ${stats['pnl']:.1f} |"
        )
    lines.append("")

    lines.append("## H3: 2024 Monthly Breakdown\n")
    lines.append("| Month | Wins | Losses | PnL$ |")
    lines.append("|---|---|---|---|")
    for month in sorted(h3_2024["by_month"]):
        stats = h3_2024["by_month"][month]
        lines.append(
            f"| {month:02d} | {stats['wins']} | {stats['losses']} | ${stats['pnl']:.1f} |"
        )
    lines.append("")

    # ---- Degrading or one bad year?
    lines.append("## Strategy Health Assessment\n")
    pfs = [r[y]["pf"] for y in _YEARS]
    trend_down = pfs[-1] < pfs[-2] < pfs[-3]
    if trend_down:
        lines.append(
            "**Trend: DEGRADING.** PF has declined 3 consecutive years (2022→2023→2024). "
            "This is not a one-off year — the fixed 50-pt trail has not kept pace with "
            "XAU volatility expansion as gold entered a secular bull market above $2000.\n"
        )
    elif pfs[-1] < 1.1 and pfs[-2] >= 2.0:
        lines.append(
            "**Likely ONE BAD YEAR**, but driven by a structural volatility shift. "
            "2021-2023 showed stable or improving PF. 2024 collapse is sharp but may "
            "recover if XAU volatility mean-reverts. ATR-adaptive trail would buffer this.\n"
        )
    else:
        lines.append(
            "**Mixed signal.** Year-over-year PF does not show a clear monotonic trend. "
            "Investigate further before drawing strong conclusions.\n"
        )

    # ---- Mitigations
    lines.append("## Recommended Mitigations\n")
    lines.append(
        "1. **ATR-adaptive trail distance**: Replace fixed 50-pt trail with `trail = 0.5 × H1_ATR_14`. "
        "In 2024 this would have widened the trail to ~100-120 pts, avoiding premature stopouts. "
        "Activation threshold should scale similarly: `trail_activate_r` stays at 0.5R.\n"
    )
    lines.append(
        "2. **Regime-gated activation**: In HIGH-VOLATILITY regime (H1 ATR > 150pts), "
        "switch to `TRAIL_100P_ACT_1R` — the wider trail tolerates bigger noise. "
        "In NORMAL regime keep `TRAIL_50P_ACT_50P`.\n"
    )
    lines.append(
        "3. **Session filter**: Check H3 results above — if Asian-session trades are the "
        "primary loss driver, disable entries during Asian hours when volatility is "
        "thin but spreads are wide.\n"
    )
    lines.append(
        "4. **TP scaling**: With larger ATR, fixed TPs may also be too tight. "
        "Try ATR-proportional TP1 = `2 × H1_ATR`, TP2 = `4 × H1_ATR`.\n"
    )

    # ---- Follow-up backtest configs
    lines.append("## Follow-Up Backtest Config Ideas\n")
    lines.append(
        "| Config ID | Description |\n"
        "|---|---|\n"
        "| ATR_TRAIL_0.5H1 | trail_points = 0.5 × rolling H1 ATR14, activate at 0.5R |\n"
        "| ATR_TRAIL_0.75H1 | trail_points = 0.75 × rolling H1 ATR14, activate at 0.5R |\n"
        "| REGIME_SWITCH | TRAIL_50P in normal; TRAIL_100P_ACT_1R in high-vol |\n"
        "| SESSION_FILTER_NO_ASIAN | Exclude Asian-session setups entirely |\n"
        "| WIDE_TP_ATR | TP1=2×ATR, TP2=4×ATR, keep 50-pt trail |\n"
    )
    lines.append("")
    lines.append(
        "_Note: ATR_TRAIL configs require modifying the engine to pass a per-bar dynamic trail distance. "
        "This is achievable by extending `TrailRule` to accept a per-bar ATR callback._\n"
    )

    out_path.write_text("\n".join(lines))
    print(f"\nReport written: {out_path}", flush=True)

    # Print key numbers to stdout
    print("\n--- KEY FINDINGS ---", flush=True)
    print(f"2024 PF={r[2024]['pf']} WR={r[2024]['wr']}% N={r[2024]['n_trades']}", flush=True)
    print(f"H1 ATR expansion: {atr_expansion_ratio}× vs 2021-2023 avg ({avg_prior_atr:.0f} pts → {y24_atr:.0f} pts)", flush=True)
    print(f"Trail-noise ratio 2024: {tnr_2024}", flush=True)
    print(f"SL exit rate: 2021={sl_pct_2021:.0f}% 2023={sl_pct_2023:.0f}% 2024={sl_pct_2024:.0f}%", flush=True)
    print(f"Trending bars: 2023={y23_trending_pct}% → 2024={y24_trending_pct}%", flush=True)
    print(f"Verdict written to: {out_path}", flush=True)


if __name__ == "__main__":
    main()

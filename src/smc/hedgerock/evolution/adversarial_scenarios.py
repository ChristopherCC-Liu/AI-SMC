"""Adversarial stress-test scenarios — historical replays + synthetic
shocks for the candidate stress tester.

This module is **pure data**: scenarios are hand-curated OHLC snippets
captured around well-known shocks (COVID 2020-03, Russia/Ukraine
2022-02, SVB 2023-03, …) plus a deterministic synthetic generator
keyed on (base_vol, shock_multiplier, duration, direction).

Isolation: no imports of ``rule_engine``; no I/O; no Tier-1 unsealed
prod modules. Hard-coded constants only — no external data files,
no network calls.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Iterable, Literal, Mapping


__all__ = [
    "BUILTIN_SCENARIOS",
    "OHLCBar",
    "SCENARIO_CATEGORIES",
    "SEVERITY_RANGE",
    "StressScenario",
    "generate_synthetic_stress",
    "iter_builtin_scenarios",
    "scenario_to_window_history",
]


SCENARIO_CATEGORIES = (
    "flash_crash",
    "geopolitical",
    "liquidity",
    "central_bank",
    "synthetic",
)
SEVERITY_RANGE = (1, 5)


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OHLCBar:
    """Single OHLC bar normalised to scenario-relative time.

    ``ts_offset_hours`` is hours since scenario start so scenarios
    are reusable across calendars without requiring a real timestamp.
    """

    ts_offset_hours: float
    open: float
    high: float
    low: float
    close: float


@dataclass(frozen=True)
class StressScenario:
    """One adversarial replay-able shock window.

    All fields are immutable + JSON-serialisable. Scenarios are
    intended to be passed by reference — never mutated.
    """

    scenario_id: str
    name: str
    description: str
    category: str
    severity: int
    ohlc_bars: tuple[OHLCBar, ...]
    duration_bars: int
    max_drawdown_pct: float
    source: Literal["historical", "synthetic"] = "historical"


# ---------------------------------------------------------------------------
# Helpers used by the builtin scenarios so the constants stay readable.
# ---------------------------------------------------------------------------


def _bars_from_closes(
    closes: Iterable[float],
    *,
    bar_minutes: int = 60,
    high_pct: float = 0.004,
    low_pct: float = 0.004,
) -> tuple[OHLCBar, ...]:
    """Synthesize OHLC bars from a sequence of closing prices.

    Each bar's open = previous close (or first close for bar 0); the
    high / low are placed within ``high_pct`` / ``low_pct`` of the
    bar's mid. Deterministic.
    """
    closes_list = [float(c) for c in closes]
    out: list[OHLCBar] = []
    prev_close: float | None = None
    for i, c in enumerate(closes_list):
        op = prev_close if prev_close is not None else c
        mid = (op + c) / 2.0
        spread_high = abs(c - op) / 2.0 + abs(mid) * high_pct
        spread_low = abs(c - op) / 2.0 + abs(mid) * low_pct
        out.append(OHLCBar(
            ts_offset_hours=i * (bar_minutes / 60.0),
            open=op,
            high=max(op, c) + spread_high,
            low=min(op, c) - spread_low,
            close=c,
        ))
        prev_close = c
    return tuple(out)


def _max_dd_pct_from_closes(closes: Iterable[float]) -> float:
    """Compute the historical peak-to-trough drawdown of a price series
    in **percent** (0 ≤ x ≤ 100)."""
    closes_list = [float(c) for c in closes]
    if not closes_list:
        return 0.0
    peak = closes_list[0]
    worst = 0.0
    for c in closes_list:
        peak = max(peak, c)
        if peak > 0:
            dd = (peak - c) / peak * 100.0
            worst = max(worst, dd)
    return round(worst, 4)


# ---------------------------------------------------------------------------
# Builtin scenarios. Each captures the rough peak-to-trough action of
# a real shock window in 24-48 hourly bars. Numbers are illustrative
# (rounded to whole / half dollars on XAUUSD; the tester only needs
# the SHAPE, not tick-perfect history).
# ---------------------------------------------------------------------------


def _covid_crash_2020_03() -> StressScenario:
    closes = (
        # 2020-03-09 → 2020-03-23 (first/last 24h compressed to 24 bars
        # capturing the down-then-up shape).
        1675, 1660, 1640, 1610, 1580, 1550, 1525, 1500,  # rapid sell-off
        1485, 1470, 1455, 1465, 1490, 1515, 1530, 1545,
        1560, 1575, 1590, 1605, 1620, 1635, 1650, 1665,
    )
    return StressScenario(
        scenario_id="covid_crash_2020_03",
        name="COVID crash — 2020-03",
        description=(
            "XAUUSD compressed to 24h: ~1700 → ~1450 then snap back to "
            "~1665 in 2 weeks. Liquidity crunch + USD scramble drove "
            "gold lower despite risk-off flight. Severity 5."
        ),
        category="liquidity",
        severity=5,
        ohlc_bars=_bars_from_closes(closes),
        duration_bars=len(closes),
        max_drawdown_pct=_max_dd_pct_from_closes(closes),
    )


def _russia_ukraine_2022_02() -> StressScenario:
    closes = (
        # 2022-02-24 → 2022-03-08 — invasion + commodity spike.
        1908, 1922, 1940, 1958, 1975, 1990, 2005, 2018,
        2030, 2045, 2058, 2065, 2070, 2055, 2045, 2025,
        2010, 2000, 2010, 2025, 2040, 2050, 2055, 2050,
    )
    return StressScenario(
        scenario_id="russia_ukraine_2022_02",
        name="Russia–Ukraine invasion — 2022-02",
        description=(
            "Risk-off flight + commodity squeeze drove XAUUSD from "
            "1900 to 2070 in 2 weeks. Severity 4."
        ),
        category="geopolitical",
        severity=4,
        ohlc_bars=_bars_from_closes(closes),
        duration_bars=len(closes),
        max_drawdown_pct=_max_dd_pct_from_closes(closes),
    )


def _svb_crisis_2023_03() -> StressScenario:
    closes = (
        # 2023-03-09 → 2023-03-20 — SVB collapse, banking-sector fear.
        1810, 1815, 1825, 1840, 1860, 1885, 1910, 1925,
        1935, 1945, 1960, 1972, 1985, 1995, 2005, 2010,
        2008, 2000, 1995, 2000, 2005, 2008, 2003, 1998,
    )
    return StressScenario(
        scenario_id="svb_crisis_2023_03",
        name="SVB / banking crisis — 2023-03",
        description=(
            "Bank-run cascade + Credit Suisse pulled XAUUSD from 1810 "
            "to ~2010 in 10 days. Severity 4."
        ),
        category="flash_crash",
        severity=4,
        ohlc_bars=_bars_from_closes(closes),
        duration_bars=len(closes),
        max_drawdown_pct=_max_dd_pct_from_closes(closes),
    )


def _mideast_2023_10() -> StressScenario:
    closes = (
        # 2023-10-07 → 2023-10-27 — Israel/Hamas escalation.
        1830, 1845, 1860, 1880, 1895, 1905, 1915, 1920,
        1925, 1930, 1940, 1955, 1970, 1985, 2000, 2005,
        2008, 2010, 2002, 1995, 1990, 1985, 1980, 1975,
    )
    return StressScenario(
        scenario_id="mideast_2023_10",
        name="Mid-east escalation — 2023-10",
        description=(
            "Geopolitical escalation drove XAUUSD from 1830 to 2010 "
            "over 3 weeks. Severity 3."
        ),
        category="geopolitical",
        severity=3,
        ohlc_bars=_bars_from_closes(closes),
        duration_bars=len(closes),
        max_drawdown_pct=_max_dd_pct_from_closes(closes),
    )


def _yen_intervention_2024_04() -> StressScenario:
    closes = (
        # 2024-04-29 → 2024-05-01 — JPY intervention rocked DXY, FX.
        2335, 2340, 2348, 2352, 2330, 2310, 2305, 2298,
        2295, 2290, 2295, 2300, 2305, 2310, 2308, 2312,
        2318, 2322, 2320, 2325, 2330, 2335, 2340, 2342,
    )
    return StressScenario(
        scenario_id="yen_intervention_2024_04",
        name="JPY intervention — 2024-04",
        description=(
            "BoJ-suspected USDJPY intervention bled into XAUUSD via "
            "DXY: ~$50 quick down + recovery in 24h. Severity 3."
        ),
        category="central_bank",
        severity=3,
        ohlc_bars=_bars_from_closes(closes),
        duration_bars=len(closes),
        max_drawdown_pct=_max_dd_pct_from_closes(closes),
    )


def _fed_pivot_2023_12() -> StressScenario:
    closes = (
        # 2023-12-13 → 2023-12-27 — dovish FOMC dot-plot.
        1995, 2002, 2010, 2020, 2032, 2040, 2050, 2058,
        2062, 2065, 2068, 2070, 2068, 2065, 2062, 2060,
        2055, 2052, 2055, 2060, 2065, 2068, 2070, 2068,
    )
    return StressScenario(
        scenario_id="fed_pivot_2023_12",
        name="Fed dovish pivot — 2023-12",
        description=(
            "Dovish Fed dot plot pulled XAUUSD from 1995 to 2070 in "
            "2 weeks. Severity 2."
        ),
        category="central_bank",
        severity=2,
        ohlc_bars=_bars_from_closes(closes),
        duration_bars=len(closes),
        max_drawdown_pct=_max_dd_pct_from_closes(closes),
    )


BUILTIN_SCENARIOS: Mapping[str, StressScenario] = MappingProxyType({
    s.scenario_id: s for s in (
        _covid_crash_2020_03(),
        _russia_ukraine_2022_02(),
        _svb_crisis_2023_03(),
        _mideast_2023_10(),
        _yen_intervention_2024_04(),
        _fed_pivot_2023_12(),
    )
})


def iter_builtin_scenarios() -> tuple[StressScenario, ...]:
    """Return the builtin scenarios as a tuple in insertion order."""
    return tuple(BUILTIN_SCENARIOS.values())


# ---------------------------------------------------------------------------
# Synthetic generator
# ---------------------------------------------------------------------------


def generate_synthetic_stress(
    *,
    scenario_id: str,
    base_vol: float,
    shock_multiplier: float,
    duration_bars: int,
    direction: Literal["up", "down"],
    base_price: float = 2000.0,
    severity: int = 3,
    category: str = "synthetic",
    description: str | None = None,
) -> StressScenario:
    """Generate a deterministic synthetic stress scenario.

    The first half of the run applies the shock at amplitude
    ``base_vol * shock_multiplier`` per bar in the chosen direction;
    the second half walks back toward (but not necessarily reaching)
    the start. Pure arithmetic — no randomness.
    """
    if duration_bars < 4:
        raise ValueError("duration_bars must be >= 4")
    if base_vol <= 0:
        raise ValueError("base_vol must be > 0")
    if shock_multiplier <= 0:
        raise ValueError("shock_multiplier must be > 0")
    if direction not in ("up", "down"):
        raise ValueError("direction must be 'up' or 'down'")
    if not (SEVERITY_RANGE[0] <= severity <= SEVERITY_RANGE[1]):
        raise ValueError(
            f"severity must be in {SEVERITY_RANGE}; got {severity}"
        )

    sign = 1.0 if direction == "up" else -1.0
    half = duration_bars // 2
    closes: list[float] = []
    price = float(base_price)
    # Shock leg.
    shock_step = sign * base_vol * shock_multiplier
    for _ in range(half):
        price = price * math.exp(shock_step)
        closes.append(price)
    # Recovery leg — half-amplitude pullback in the opposite direction.
    recovery_step = -sign * base_vol * shock_multiplier * 0.5
    for _ in range(duration_bars - half):
        price = price * math.exp(recovery_step)
        closes.append(price)

    return StressScenario(
        scenario_id=scenario_id,
        name=f"synthetic shock ({direction}, ×{shock_multiplier})",
        description=description or (
            f"Synthetic {direction} shock with base_vol={base_vol}, "
            f"shock_multiplier={shock_multiplier}, "
            f"duration_bars={duration_bars}."
        ),
        category=category,
        severity=severity,
        ohlc_bars=_bars_from_closes(closes),
        duration_bars=duration_bars,
        max_drawdown_pct=_max_dd_pct_from_closes(closes),
        source="synthetic",
    )


# ---------------------------------------------------------------------------
# Adapter: scenario → walk-forward backtest "history" (window dicts)
# ---------------------------------------------------------------------------


def scenario_to_window_history(
    scenario: StressScenario, *, window_bars: int = 4,
) -> list[dict]:
    """Slice the scenario into ``window_bars``-sized chunks suitable for
    :func:`phase_d_walk_forward.run_walk_forward_backtest`.

    Each window's ``pnl_pp`` is the cumulative log return; ``dd_pp`` is
    the in-window peak-to-trough drawdown in percent.
    """
    if window_bars < 1:
        raise ValueError("window_bars must be >= 1")
    bars = scenario.ohlc_bars
    out: list[dict] = []
    for start in range(0, len(bars), window_bars):
        chunk = bars[start: start + window_bars]
        if len(chunk) < 2:
            continue
        first_close = chunk[0].close
        last_close = chunk[-1].close
        if first_close <= 0:
            continue
        pnl_pp = math.log(last_close / first_close) * 100.0
        peak = chunk[0].high
        worst = 0.0
        for b in chunk:
            peak = max(peak, b.high)
            if peak > 0:
                worst = max(worst, (peak - b.low) / peak * 100.0)
        out.append({
            "window_id": f"{scenario.scenario_id}_w{start // window_bars}",
            "pnl_pp": round(pnl_pp, 6),
            "dd_pp": round(worst, 6),
            "n_signals": len(chunk),
            "regime_bucket": scenario.category,
        })
    return out

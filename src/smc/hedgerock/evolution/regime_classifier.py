"""Ticket 4 v2 Step 3 — XAUUSD-only sidecar regime classifier.

Pure function module: ingests a list of H1 bar dicts (``ts``, ``high``,
``low``, ``close``) and returns a tuple of regime bucket labels drawn
from :data:`window_coverage.REGIME_BUCKETS`.

**Sidecar layer.** No production import, no clock, no I/O, no mutation
of the input list.  The runner uses the output to compute the
``observed_buckets`` field consumed by
``window_coverage.check_window_coverage``.

Bucket detection rules (RFC v2 §3):

* ``range_low_vol``   — median bar range < 0.3% AND no strong trend
* ``range_high_vol``  — median bar range >= 0.3% AND no strong trend
* ``trend_up``        — net close change > +2% across the window
* ``trend_down``      — net close change < -2% across the window
* ``breakout``        — any single bar with intra-bar range > 2%
* ``weekend_gap``     — any consecutive ts gap >= 60 hours
* ``news_crisis``     — NOT detected here; requires external news feed
                        and is supplied by the runner if present.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


__all__ = [
    "classify_window",
    "compute_window_features",
    "VOL_THRESHOLD",
    "TREND_THRESHOLD",
    "BREAKOUT_THRESHOLD",
    "WEEKEND_GAP_HOURS",
]


# --- thresholds (RFC v2 §3) -------------------------------------------------

#: Bar-range volatility split (range_low_vol vs range_high_vol).
VOL_THRESHOLD: float = 0.003

#: Net close-change threshold for trend_up / trend_down.
TREND_THRESHOLD: float = 0.02

#: Single-bar intra-bar range threshold for breakout.
BREAKOUT_THRESHOLD: float = 0.02

#: Consecutive ts gap threshold (hours) for weekend_gap detection.
WEEKEND_GAP_HOURS: float = 60.0


# --- helpers ----------------------------------------------------------------


def _median(xs: Sequence[float]) -> float:
    """Pure median over a non-empty numeric sequence (sorted copy)."""
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return float(s[mid])
    return float((s[mid - 1] + s[mid]) / 2.0)


def _bar_range_pct(bar: Mapping[str, Any]) -> float:
    """Intra-bar range as a fraction of close. 0.0 if close <= 0."""
    close = float(bar["close"])
    if close <= 0:
        return 0.0
    return (float(bar["high"]) - float(bar["low"])) / close


def _gap_hours(prev_ts: Any, ts: Any) -> float:
    """Hours between two timestamps. 0.0 if either side is missing or
    the diff cannot be computed."""
    try:
        delta = ts - prev_ts
    except TypeError:
        return 0.0
    seconds = getattr(delta, "total_seconds", lambda: 0.0)()
    return float(seconds) / 3600.0


# --- features ---------------------------------------------------------------


def compute_window_features(
    bars: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Pure feature extraction for a window of H1 bar dicts.

    Each bar is a mapping with at least the keys ``ts``, ``high``,
    ``low``, ``close``. The function never mutates ``bars``.
    """
    if not bars:
        return {
            "n_bars": 0,
            "median_vol_rank": 0.0,
            "trend_bars_h4_proxy": 0.0,
            "weekend_gap_count": 0,
            "max_single_bar_move_pct": 0.0,
            "net_change_pct": 0.0,
        }

    n = len(bars)
    bar_ranges: list[float] = [_bar_range_pct(b) for b in bars]
    median_vol = _median(bar_ranges)
    max_single_bar = max(bar_ranges) if bar_ranges else 0.0

    # Weekend-gap counter: any consecutive ts delta >= WEEKEND_GAP_HOURS.
    weekend_gap_count = 0
    for i in range(1, n):
        if _gap_hours(bars[i - 1]["ts"], bars[i]["ts"]) >= WEEKEND_GAP_HOURS:
            weekend_gap_count += 1

    # Net close change as a fraction of starting close.
    first_close = float(bars[0]["close"])
    last_close = float(bars[-1]["close"])
    if first_close > 0:
        net_change_pct = (last_close - first_close) / first_close
    else:
        net_change_pct = 0.0

    # Lightweight trend-bias proxy: net signed step count over H4 (4-bar)
    # windows, expressed as a fraction of the number of H4 windows. A
    # crude stand-in for an H4 directional bias signal — sidecar-only.
    h4_windows = max(1, n // 4)
    h4_signed = 0
    for i in range(0, n - 4, 4):
        a = float(bars[i]["close"])
        b = float(bars[i + 4]["close"])
        if b > a:
            h4_signed += 1
        elif b < a:
            h4_signed -= 1
    trend_bars_h4_proxy = float(h4_signed) / float(h4_windows)

    return {
        "n_bars": n,
        "median_vol_rank": median_vol,
        "trend_bars_h4_proxy": trend_bars_h4_proxy,
        "weekend_gap_count": weekend_gap_count,
        "max_single_bar_move_pct": max_single_bar,
        "net_change_pct": net_change_pct,
    }


# --- classifier -------------------------------------------------------------


def classify_window(
    bars: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    """Map a window's H1 bars to a tuple of regime bucket labels.

    Empty input returns an empty tuple. Output ordering is stable and
    deterministic for a given input. ``bars`` is never mutated.
    """
    if not bars:
        return ()

    f = compute_window_features(bars)
    median_vol = float(f["median_vol_rank"])
    net_change = float(f["net_change_pct"])
    max_bar_move = float(f["max_single_bar_move_pct"])
    gap_count = int(f["weekend_gap_count"])

    buckets: list[str] = []

    # Trend detection (mutually exclusive direction).
    if net_change > TREND_THRESHOLD:
        buckets.append("trend_up")
    elif net_change < -TREND_THRESHOLD:
        buckets.append("trend_down")

    # Range classification only when the window is not strongly
    # directional. We always emit either low_vol or high_vol in this
    # case so the bucket coverage gate has something to count.
    if abs(net_change) <= TREND_THRESHOLD:
        if median_vol >= VOL_THRESHOLD:
            buckets.append("range_high_vol")
        else:
            buckets.append("range_low_vol")

    if max_bar_move > BREAKOUT_THRESHOLD:
        buckets.append("breakout")

    if gap_count > 0:
        buckets.append("weekend_gap")

    return tuple(buckets)

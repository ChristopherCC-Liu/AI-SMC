"""ATH synthetic zones — fallback supply/demand levels for new-high regimes.

Round 5 A-track Task #9 (follow-up to task #56 2024 setup drought diagnosis).

Context: XAUUSD W14 + W15 of 2024 (Apr–Sep, $2,300–$2,685 rally) generated
**zero** SMC setups — the detector found no unmitigated OB or unfilled FVG
zones because price broke to new highs with no historical supply/demand
reference.  This is a structural limitation of SMC pattern detection: it
needs *traded-through* levels to anchor entries.

Solution: when price is inside the ATH zone (> 95 % of 52-week range), fall
back to *synthetic* supply/demand levels derived from:

    - **VWAP bands** (M15 20-period VWAP ± 1 std dev) — volume-weighted
      mean reversion anchors
    - **Session highs/lows** (Asian/London/NY sessions of the current day) —
      institutional reference prices
    - **Round numbers** (e.g. $4,800 / $4,850 / $4,900 for XAU) — psychological
      support/resistance that traders actually watch
    - **Previous week H/L** (weekly pivots) — HTF structural anchors

Integration: `aggregator.generate_setups` augments historical zones with
synthetic when `len(historical_zones) < 2 AND price_52w_percentile > 0.95`.
It **augments**, does NOT replace — existing OBs/FVGs remain preferred.

Default OFF via `SMCConfig.synthetic_zones_enabled` (Round 5 kill-safe rollout).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

import polars as pl

from smc.data.schemas import Timeframe
from smc.strategy.types import TradeZone

__all__ = [
    "SyntheticZoneConfig",
    "build_synthetic_zones",
    "ATH_TRIGGER_PERCENTILE",
    "DEFAULT_ROUND_NUMBER_STEP",
    "DEFAULT_ROUND_NUMBER_WINDOW_PCT",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Fire synthetic zones when price sits in the top 5 % of the 52-week range.
ATH_TRIGGER_PERCENTILE: float = 0.95

# Round-number stride for XAUUSD — $50 steps cover the $1,500 / $2,000 /
# $4,500 / $4,800 psychological levels that traders actually watch.
DEFAULT_ROUND_NUMBER_STEP: float = 50.0

# Only include round numbers within ±1.5 % of current price.  Wider windows
# dilute confluence; tighter ones miss levels when price sits between two.
DEFAULT_ROUND_NUMBER_WINDOW_PCT: float = 1.5

# Zone width for a synthetic anchor — narrow because these are single-line
# references not range zones.  ±0.1 % of price keeps the box small enough
# for entries to register but wide enough to tolerate tick noise.
_ZONE_HALF_WIDTH_PCT: float = 0.1

# VWAP computation window.
_VWAP_PERIOD: int = 20

# Session UTC boundaries (approximate — XAU sessions inherit FX conventions).
_SESSION_BOUNDS_UTC: dict[str, tuple[int, int]] = {
    # session_name: (start_hour, end_hour)  — end exclusive
    "asian":  (0, 8),
    "london": (8, 13),
    "ny":     (13, 21),
}

# Max zones per source so a single source can't dominate the output.
_MAX_VWAP_ZONES: int = 2
_MAX_SESSION_ZONES: int = 3
_MAX_ROUND_NUMBER_ZONES: int = 3
_MAX_PREV_WEEK_ZONES: int = 2


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SyntheticZoneConfig:
    """Immutable tuning for the synthetic zone builder.

    All defaults match the module-level constants; callers can override
    selective knobs (e.g. round-number step for BTC would be much larger).
    """

    round_number_step: float = DEFAULT_ROUND_NUMBER_STEP
    round_number_window_pct: float = DEFAULT_ROUND_NUMBER_WINDOW_PCT
    zone_half_width_pct: float = _ZONE_HALF_WIDTH_PCT
    vwap_period: int = _VWAP_PERIOD
    ath_trigger_percentile: float = ATH_TRIGGER_PERCENTILE


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_zone(
    anchor: float,
    direction: Literal["long", "short"],
    *,
    timeframe: Timeframe = Timeframe.H1,
    confidence: float = 0.5,
    half_width_pct: float = _ZONE_HALF_WIDTH_PCT,
) -> TradeZone:
    """Build a frozen ``TradeZone`` of zone_type="synthetic" around ``anchor``.

    The half-width is a percentage of the anchor price so XAU (4000) and
    BTC (70k) both produce visually-reasonable zone boxes.
    """
    half = anchor * (half_width_pct / 100.0)
    return TradeZone(
        zone_high=round(anchor + half, 2),
        zone_low=round(anchor - half, 2),
        zone_type="synthetic",
        direction=direction,
        timeframe=timeframe,
        confidence=round(confidence, 3),
    )


def _direction_for_anchor(anchor: float, current_price: float) -> Literal["long", "short"]:
    """Anchors below current price act as demand (long); above as supply (short)."""
    return "long" if anchor < current_price else "short"


def _vwap_bands(
    m15_df: pl.DataFrame,
    current_price: float,
    cfg: SyntheticZoneConfig,
) -> list[TradeZone]:
    """Compute VWAP ± 1 std dev over last N M15 bars → 2 synthetic zones."""
    period = cfg.vwap_period
    if m15_df is None or len(m15_df) < period:
        return []

    recent = m15_df[-period:]
    # Use typical price (H+L+C)/3 as the VWAP input — industry standard.
    high = recent["high"].to_list()
    low = recent["low"].to_list()
    close = recent["close"].to_list()
    volume = recent["volume"].to_list() if "volume" in recent.columns else [1.0] * len(close)

    typical = [(high[i] + low[i] + close[i]) / 3.0 for i in range(len(close))]
    total_vol = sum(volume)
    if total_vol <= 0:
        return []

    vwap = sum(typical[i] * volume[i] for i in range(len(typical))) / total_vol

    # Volume-weighted variance around VWAP.
    variance = sum(volume[i] * (typical[i] - vwap) ** 2 for i in range(len(typical))) / total_vol
    std = max(variance, 0.0) ** 0.5

    upper = vwap + std
    lower = vwap - std
    zones: list[TradeZone] = []
    for anchor in (upper, lower):
        # Skip when the band coincides with current price (no edge).
        if abs(anchor - current_price) < 0.01 * current_price / 100.0:
            continue
        zones.append(
            _make_zone(
                anchor=anchor,
                direction=_direction_for_anchor(anchor, current_price),
                confidence=0.55,  # slightly above synthetic baseline
                half_width_pct=cfg.zone_half_width_pct,
            )
        )
    return zones[:_MAX_VWAP_ZONES]


def _session_highs_lows(
    m15_df: pl.DataFrame,
    current_price: float,
    cfg: SyntheticZoneConfig,
    *,
    now: datetime | None = None,
) -> list[TradeZone]:
    """Session H/L for each fully-closed session of the current UTC day."""
    if m15_df is None or len(m15_df) == 0 or "ts" not in m15_df.columns:
        return []

    reference = now if now is not None else datetime.now(timezone.utc)
    day_start = reference.replace(hour=0, minute=0, second=0, microsecond=0)

    zones: list[TradeZone] = []
    for session, (h_start, h_end) in _SESSION_BOUNDS_UTC.items():
        # Filter bars inside this UTC session of the current day.
        mask = (
            (m15_df["ts"] >= day_start.replace(hour=h_start))
            & (m15_df["ts"] < day_start.replace(hour=h_end))
        )
        session_bars = m15_df.filter(mask)
        if len(session_bars) == 0:
            continue

        sess_high = float(session_bars["high"].max())
        sess_low = float(session_bars["low"].min())

        for anchor in (sess_high, sess_low):
            if not (anchor > 0.0):
                continue
            zones.append(
                _make_zone(
                    anchor=anchor,
                    direction=_direction_for_anchor(anchor, current_price),
                    confidence=0.5,
                    half_width_pct=cfg.zone_half_width_pct,
                )
            )
    return zones[:_MAX_SESSION_ZONES]


def _round_numbers(
    current_price: float,
    cfg: SyntheticZoneConfig,
) -> list[TradeZone]:
    """Psychological round-number levels within ±window_pct of current price."""
    if current_price <= 0:
        return []

    step = cfg.round_number_step
    if step <= 0:
        return []

    window = current_price * (cfg.round_number_window_pct / 100.0)
    lower_bound = current_price - window
    upper_bound = current_price + window

    # Walk the integer grid across the window.
    start_idx = int(lower_bound // step) + 1
    end_idx = int(upper_bound // step) + 1

    zones: list[TradeZone] = []
    for k in range(start_idx, end_idx + 1):
        anchor = k * step
        if anchor <= 0 or anchor == current_price:
            continue
        if anchor < lower_bound or anchor > upper_bound:
            continue
        zones.append(
            _make_zone(
                anchor=anchor,
                direction=_direction_for_anchor(anchor, current_price),
                confidence=0.45,
                half_width_pct=cfg.zone_half_width_pct,
            )
        )
    return zones[:_MAX_ROUND_NUMBER_ZONES]


def _prev_week_high_low(
    h1_df: pl.DataFrame,
    current_price: float,
    cfg: SyntheticZoneConfig,
    *,
    now: datetime | None = None,
) -> list[TradeZone]:
    """Previous-week H/L as HTF synthetic pivots."""
    if h1_df is None or len(h1_df) == 0 or "ts" not in h1_df.columns:
        return []

    reference = now if now is not None else datetime.now(timezone.utc)
    # ISO week: Monday = weekday() 0, Sunday = 6. Go back to Monday of this
    # week, then subtract one more week for "last Monday" → "last Sunday".
    this_monday = reference - _timedelta_from_weekday(reference.weekday(), reference)
    last_monday = this_monday - _timedelta_days(7)
    last_sunday_end = this_monday - _timedelta_days(0)  # exclusive bound

    mask = (h1_df["ts"] >= last_monday) & (h1_df["ts"] < last_sunday_end)
    prev_week = h1_df.filter(mask)
    if len(prev_week) == 0:
        return []

    high = float(prev_week["high"].max())
    low = float(prev_week["low"].min())

    zones: list[TradeZone] = []
    for anchor in (high, low):
        if not (anchor > 0.0):
            continue
        zones.append(
            _make_zone(
                anchor=anchor,
                direction=_direction_for_anchor(anchor, current_price),
                confidence=0.55,
                half_width_pct=cfg.zone_half_width_pct,
            )
        )
    return zones[:_MAX_PREV_WEEK_ZONES]


def _timedelta_from_weekday(wday: int, reference: datetime) -> "object":
    """Return a ``datetime.timedelta`` aligning ``reference`` to Monday 00:00 UTC."""
    from datetime import timedelta
    return timedelta(
        days=wday,
        hours=reference.hour,
        minutes=reference.minute,
        seconds=reference.second,
        microseconds=reference.microsecond,
    )


def _timedelta_days(days: int) -> "object":
    from datetime import timedelta
    return timedelta(days=days)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_synthetic_zones(
    m15_df: pl.DataFrame | None,
    h1_df: pl.DataFrame | None,
    current_price: float,
    price_52w_high: float,
    price_52w_low: float,
    *,
    cfg: SyntheticZoneConfig | None = None,
    now: datetime | None = None,
) -> list[TradeZone]:
    """Return a list of synthetic ``TradeZone`` anchors for ATH regimes.

    Returns an empty list when:
        - ``current_price`` is below the ATH trigger threshold
          (``price_52w_percentile < cfg.ath_trigger_percentile``), or
        - all component functions return no anchors.

    Parameters
    ----------
    m15_df:
        M15 OHLCV DataFrame (must include ``ts`` + ``high`` + ``low`` +
        ``close``; ``volume`` is optional — defaults to equal weighting).
    h1_df:
        H1 OHLCV DataFrame (for previous-week H/L).
    current_price:
        Latest market price (used for direction assignment and round-number
        window).
    price_52w_high / price_52w_low:
        52-week range — used to decide whether the ATH trigger fires.
    cfg:
        ``SyntheticZoneConfig`` override. ``None`` uses module defaults.
    now:
        Injected "current" datetime for deterministic tests.  Defaults to
        ``datetime.now(timezone.utc)``.

    Returns
    -------
    list[TradeZone]
        Up to 10 synthetic zones (VWAP ≤ 2, session ≤ 3, round ≤ 3,
        prev-week ≤ 2), each with ``zone_type="synthetic"``.
    """
    cfg = cfg if cfg is not None else SyntheticZoneConfig()

    # Gate 1: only fire in ATH regime.
    if price_52w_high <= 0 or price_52w_high == price_52w_low:
        return []
    pct = (current_price - price_52w_low) / (price_52w_high - price_52w_low)
    if pct < cfg.ath_trigger_percentile:
        return []

    zones: list[TradeZone] = []
    zones.extend(_vwap_bands(m15_df, current_price, cfg) if m15_df is not None else [])
    zones.extend(_session_highs_lows(m15_df, current_price, cfg, now=now) if m15_df is not None else [])
    zones.extend(_round_numbers(current_price, cfg))
    zones.extend(_prev_week_high_low(h1_df, current_price, cfg, now=now) if h1_df is not None else [])

    # Deduplicate near-identical anchors (within 0.05 % of price) — keeps
    # tight VWAP + round-number overlap from producing 5 overlapping zones.
    zones = _dedupe_by_proximity(zones, current_price)
    return zones


def _dedupe_by_proximity(zones: list[TradeZone], current_price: float) -> list[TradeZone]:
    """Remove zones whose centres are within 0.05 % of another already kept."""
    tolerance = current_price * 0.0005  # 0.05 %
    kept: list[TradeZone] = []
    for z in zones:
        mid = (z.zone_high + z.zone_low) / 2.0
        conflict = any(
            abs(mid - (k.zone_high + k.zone_low) / 2.0) <= tolerance for k in kept
        )
        if not conflict:
            kept.append(z)
    return kept

"""Range detection and mean-reversion setup generation for Mode B (ranging).

Detects horizontal consolidation ranges on H1 using two methods:
  - Method A: OB boundaries (highest bearish OB.high + lowest bullish OB.low)
  - Method B: Swing extremes fallback (max swing high + min swing low, last 50 bars)

Generates support_bounce / resistance_rejection setups when price is near
a boundary, confirmed by M15 CHoCH in the corresponding zone.

SL is ATR-adaptive (reuses entry_trigger._compute_sl_buffer).
TP conservative = midpoint, TP aggressive = opposite boundary minus 10%.
"""

from __future__ import annotations

from datetime import datetime, timezone

import polars as pl

from smc.smc_core.constants import XAUUSD_POINT_SIZE
from smc.smc_core.types import OrderBlock, SMCSnapshot, SwingPoint
from smc.strategy.entry_trigger import _compute_sl_buffer, _find_choch_in_zone
from smc.strategy.range_types import RangeBounds, RangeSetup
from smc.strategy.types import TradeZone

__all__ = ["RangeTrader", "check_range_guards"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MIN_SWING_POINTS_FOR_RANGE = 2
_SWING_LOOKBACK_BARS = 50
_OB_BOUNDARY_CONFIDENCE = 0.8
_SWING_EXTREME_CONFIDENCE = 0.6
_DONCHIAN_CONFIDENCE = 0.5  # Round 4.5: lowest trust — pure statistical fallback
# Round 4.6-A: lookback 24 → 48 (2 trading days). Asian single-session window
# too narrow to form detectable channel on low-volatility days — widen to span
# Asian + prior London/NY session for more stable upper/lower bounds.
_DONCHIAN_LOOKBACK_BARS = 48

# Round 4.6-B: session-aware Guard 1 (width) & Guard 4 (duration) thresholds.
# Asian core market 低波动 — H1 48-bar channel 实测 400-700 pts 居多,
# 旧 800/12 threshold 把 Asian 全 reject. Downstream CircuitBreaker +
# RangeQuotaTracker 仍作 risk 兜底.
_ASIAN_SESSIONS: frozenset[str] = frozenset(
    {"ASIAN_CORE", "ASIAN_LONDON_TRANSITION"}
)
_GUARD_WIDTH_MIN_DEFAULT = 800.0
_GUARD_WIDTH_MIN_ASIAN = 400.0
_GUARD_DURATION_MIN_DEFAULT = 12
_GUARD_DURATION_MIN_ASIAN = 8
_SECONDS_PER_H1_BAR = 3600


def _h1_bars_between(earlier: datetime, later: datetime) -> int:
    """Round 4.5.1: approximate H1 bars elapsed between two timestamps.

    Used by Method A/B to populate RangeBounds.duration_bars so Guard 4 (>=12)
    can evaluate actual range age instead of seeing 0 (silent reject).
    """
    if earlier.tzinfo is None:
        earlier = earlier.replace(tzinfo=timezone.utc)
    if later.tzinfo is None:
        later = later.replace(tzinfo=timezone.utc)
    delta_seconds = (later - earlier).total_seconds()
    if delta_seconds <= 0:
        return 0
    return int(delta_seconds // _SECONDS_PER_H1_BAR)


# ---------------------------------------------------------------------------
# Range guard functions (module-level)
# ---------------------------------------------------------------------------


def check_range_guards(
    bounds: RangeBounds,
    setup: RangeSetup,
    session: str,
    h1_df: pl.DataFrame,
) -> bool:
    """Return True if all 5 guards pass. See v3.0 plan.

    Round 4.6-B: Guard 1 (width) and Guard 4 (duration) are session-aware.
    Asian sessions use relaxed thresholds (width 400/duration 8) because
    Asian low-volatility ranges rarely reach London/NY-calibrated 800/12.
    Guards 2 (RR>=1.2), 3 (touches>=2), 5 (lot) stay uniform to preserve
    quality floor.
    """
    is_asian = session in _ASIAN_SESSIONS
    min_width = _GUARD_WIDTH_MIN_ASIAN if is_asian else _GUARD_WIDTH_MIN_DEFAULT
    min_duration = (
        _GUARD_DURATION_MIN_ASIAN if is_asian else _GUARD_DURATION_MIN_DEFAULT
    )

    # Guard 1: width — session-aware
    if bounds.width_points < min_width:
        return False
    # Guard 2: RR >= 1.2
    if setup.rr_ratio < 1.2:
        return False
    # Guard 3: >= 2 boundary touches, tolerance = width * 5%
    touches = _count_boundary_touches(h1_df, bounds, tolerance_ratio=0.05)
    if touches < 2:
        return False
    # Guard 4: range_duration_h1_bars — session-aware
    if bounds.duration_bars < min_duration:
        return False
    # Guard 5: lot_multiplier is applied downstream — no direct check here
    return True


def _count_boundary_touches(
    h1_df: pl.DataFrame,
    bounds: RangeBounds,
    tolerance_ratio: float = 0.05,
) -> int:
    """Count bars that touched either upper or lower boundary within tolerance."""
    tolerance_pts = bounds.width_points * tolerance_ratio
    tol_price = tolerance_pts * XAUUSD_POINT_SIZE

    highs = h1_df["high"].to_list()
    lows = h1_df["low"].to_list()
    touches = 0
    for h, l in zip(highs, lows):
        if abs(h - bounds.upper) <= tol_price or abs(l - bounds.lower) <= tol_price:
            touches += 1
    return touches


# ---------------------------------------------------------------------------
# RangeTrader class
# ---------------------------------------------------------------------------


class RangeTrader:
    """Detects horizontal ranges and generates mean-reversion setups.

    Parameters
    ----------
    min_range_width:
        Minimum range width in points to qualify as a valid range.
    max_range_width:
        Maximum range width in points (reject overly wide ranges).
    boundary_pct:
        Fraction of range width that defines the "boundary zone" at each edge.
    """

    def __init__(
        self,
        min_range_width: float = 200.0,  # Round 4.6-A: 300 → 200 (Asian 低波动接受更窄 range)
        max_range_width: float = 3000.0,
        boundary_pct: float = 0.15,
    ) -> None:
        self._min_range_width = min_range_width
        self._max_range_width = max_range_width
        self._boundary_pct = boundary_pct

    # ------------------------------------------------------------------
    # Range detection
    # ------------------------------------------------------------------

    def detect_range(
        self,
        h1_df: pl.DataFrame,
        h1_snapshot: SMCSnapshot,
    ) -> RangeBounds | None:
        """Detect a horizontal range from H1 data.

        Tries Method A (OB boundaries), then Method B (swing extremes),
        then Method D (Donchian channel — Round 4.5 hotfix for Asian core
        low-volatility fallback). Returns None if no valid range is found.
        """
        now = datetime.now(tz=timezone.utc)

        # Method A: OB boundaries
        bounds = self._detect_from_ob_boundaries(h1_snapshot, now)
        if bounds is not None:
            return bounds

        # Method B: Swing extreme fallback
        bounds = self._detect_from_swing_extremes(h1_snapshot, now)
        if bounds is not None:
            return bounds

        # Method D: Donchian channel fallback (Round 4.5)
        # Asian core 低波动场景下 OB/swing 稀疏 → 用 N-bar high/low channel
        # Guard 1 (width >= 800) 自然过滤过窄 channel
        return self._detect_from_donchian_channel(h1_df, now)

    # ------------------------------------------------------------------
    # Setup generation
    # ------------------------------------------------------------------

    def generate_range_setups(
        self,
        h1_snapshot: SMCSnapshot,
        m15_snapshot: SMCSnapshot,
        current_price: float,
        bounds: RangeBounds,
        h1_atr: float = 0.0,
    ) -> tuple[RangeSetup, ...]:
        """Generate mean-reversion setups at range boundaries.

        Returns at most 2 setups (one long at lower boundary, one short at
        upper boundary).  Each requires M15 CHoCH confirmation.
        """
        setups: list[RangeSetup] = []
        boundary_width = bounds.width_points * XAUUSD_POINT_SIZE * self._boundary_pct

        # --- Lower boundary: support bounce (long) ---
        if current_price <= bounds.lower + boundary_width:
            setup = self._build_setup(
                direction="long",
                trigger="support_bounce",
                current_price=current_price,
                bounds=bounds,
                m15_snapshot=m15_snapshot,
                h1_atr=h1_atr,
            )
            if setup is not None:
                setups.append(setup)

        # --- Upper boundary: resistance rejection (short) ---
        if current_price >= bounds.upper - boundary_width:
            setup = self._build_setup(
                direction="short",
                trigger="resistance_rejection",
                current_price=current_price,
                bounds=bounds,
                m15_snapshot=m15_snapshot,
                h1_atr=h1_atr,
            )
            if setup is not None:
                setups.append(setup)

        return tuple(setups[:2])

    # ------------------------------------------------------------------
    # Private: range detection methods
    # ------------------------------------------------------------------

    def _detect_from_ob_boundaries(
        self,
        h1_snapshot: SMCSnapshot,
        now: datetime,
    ) -> RangeBounds | None:
        """Method A: Use highest bearish OB.high and lowest bullish OB.low."""
        bearish_obs = tuple(
            ob for ob in h1_snapshot.order_blocks
            if ob.ob_type == "bearish" and not ob.mitigated
        )
        bullish_obs = tuple(
            ob for ob in h1_snapshot.order_blocks
            if ob.ob_type == "bullish" and not ob.mitigated
        )

        if not bearish_obs or not bullish_obs:
            return None

        upper = max(ob.high for ob in bearish_obs)
        lower = min(ob.low for ob in bullish_obs)

        if upper <= lower:
            return None

        # Round 4.5.1 fix: compute duration from earliest boundary-defining OB.
        # Previously defaulted to 0 → Guard 4 (>=12) silently rejected Method A.
        earliest_ts = min(ob.ts_start for ob in (*bearish_obs, *bullish_obs))
        duration_bars = _h1_bars_between(earliest_ts, now)

        return self._validate_bounds(
            upper=upper,
            lower=lower,
            source="ob_boundaries",
            confidence=_OB_BOUNDARY_CONFIDENCE,
            now=now,
            duration_bars=duration_bars,
        )

    def _detect_from_swing_extremes(
        self,
        h1_snapshot: SMCSnapshot,
        now: datetime,
    ) -> RangeBounds | None:
        """Method B: Use max swing high + min swing low from recent swings."""
        swings = h1_snapshot.swing_points
        if len(swings) < _MIN_SWING_POINTS_FOR_RANGE:
            return None

        # Take last N swing points (proxy for lookback window)
        recent = swings[-_SWING_LOOKBACK_BARS:]

        highs = tuple(s for s in recent if s.swing_type == "high")
        lows = tuple(s for s in recent if s.swing_type == "low")

        if not highs or not lows:
            return None

        upper = max(s.price for s in highs)
        lower = min(s.price for s in lows)

        if upper <= lower:
            return None

        # Round 4.5.1 fix: compute duration from earliest swing defining bounds.
        # Previously defaulted to 0 → Guard 4 (>=12) silently rejected Method B.
        earliest_ts = min(s.ts for s in (*highs, *lows))
        duration_bars = _h1_bars_between(earliest_ts, now)

        return self._validate_bounds(
            upper=upper,
            lower=lower,
            source="swing_extremes",
            confidence=_SWING_EXTREME_CONFIDENCE,
            now=now,
            duration_bars=duration_bars,
        )

    def _detect_from_donchian_channel(
        self,
        h1_df: pl.DataFrame,
        now: datetime,
    ) -> RangeBounds | None:
        """Method D (Round 4.5): N-bar high/low channel (Donchian).

        Pure statistical fallback when Method A/B fail (Asian core low
        volatility → OB/swing sparse). Uses last _DONCHIAN_LOOKBACK_BARS
        (24 H1 bars = 1 trading day) to define upper/lower channel.

        Width must exceed _min_range_width (inline equivalent of Guard 1).
        Downstream 5 guards (width/RR/touches/duration/lot) still enforce
        full quality gate — this only opens candidate detection.
        """
        if h1_df.is_empty() or h1_df.height < _DONCHIAN_LOOKBACK_BARS:
            return None

        recent = h1_df.tail(_DONCHIAN_LOOKBACK_BARS)
        upper = float(recent["high"].max())
        lower = float(recent["low"].min())

        if upper <= lower:
            return None

        return self._validate_bounds(
            upper=upper,
            lower=lower,
            source="donchian_channel",
            confidence=_DONCHIAN_CONFIDENCE,
            now=now,
            duration_bars=_DONCHIAN_LOOKBACK_BARS,
        )

    def _validate_bounds(
        self,
        *,
        upper: float,
        lower: float,
        source: str,
        confidence: float,
        now: datetime,
        duration_bars: int = 0,
    ) -> RangeBounds | None:
        """Shared width validation for both detection methods."""
        width_points = (upper - lower) / XAUUSD_POINT_SIZE

        if width_points < self._min_range_width:
            return None
        if width_points > self._max_range_width:
            return None

        return RangeBounds(
            upper=round(upper, 2),
            lower=round(lower, 2),
            width_points=round(width_points, 1),
            midpoint=round((upper + lower) / 2.0, 2),
            detected_at=now,
            source=source,  # type: ignore[arg-type]
            confidence=confidence,
            duration_bars=duration_bars,
        )

    # ------------------------------------------------------------------
    # Private: setup building
    # ------------------------------------------------------------------

    def _build_setup(
        self,
        *,
        direction: str,
        trigger: str,
        current_price: float,
        bounds: RangeBounds,
        m15_snapshot: SMCSnapshot,
        h1_atr: float,
    ) -> RangeSetup | None:
        """Build a single mean-reversion setup with M15 CHoCH confirmation."""
        # Create a synthetic TradeZone at the boundary for CHoCH check
        if direction == "long":
            boundary_width = bounds.width_points * XAUUSD_POINT_SIZE * self._boundary_pct
            zone = TradeZone(
                zone_high=round(bounds.lower + boundary_width, 2),
                zone_low=round(bounds.lower, 2),
                zone_type="ob",
                direction="long",
                timeframe=m15_snapshot.timeframe,
                confidence=bounds.confidence,
            )
        else:
            boundary_width = bounds.width_points * XAUUSD_POINT_SIZE * self._boundary_pct
            zone = TradeZone(
                zone_high=round(bounds.upper, 2),
                zone_low=round(bounds.upper - boundary_width, 2),
                zone_type="ob",
                direction="short",
                timeframe=m15_snapshot.timeframe,
                confidence=bounds.confidence,
            )

        # Require M15 CHoCH confirmation in the synthetic zone
        if not _find_choch_in_zone(m15_snapshot, zone):
            return None

        # SL: boundary +/- ATR-adaptive buffer
        sl_buffer = _compute_sl_buffer(h1_atr) * XAUUSD_POINT_SIZE
        if direction == "long":
            stop_loss = bounds.lower - sl_buffer
        else:
            stop_loss = bounds.upper + sl_buffer

        entry_price = current_price
        risk_points = abs(entry_price - stop_loss) / XAUUSD_POINT_SIZE
        if risk_points == 0:
            return None

        # TP conservative: midpoint
        take_profit = bounds.midpoint

        # TP aggressive: opposite boundary minus 10% inset
        range_price = bounds.width_points * XAUUSD_POINT_SIZE
        inset = range_price * 0.10
        if direction == "long":
            take_profit_ext = bounds.upper - inset
        else:
            take_profit_ext = bounds.lower + inset

        reward_points = abs(take_profit - entry_price) / XAUUSD_POINT_SIZE
        rr_ratio = reward_points / risk_points if risk_points > 0 else 0.0

        grade = self._grade_setup(bounds, rr_ratio)

        return RangeSetup(
            direction=direction,  # type: ignore[arg-type]
            entry_price=round(entry_price, 2),
            stop_loss=round(stop_loss, 2),
            take_profit=round(take_profit, 2),
            take_profit_ext=round(take_profit_ext, 2),
            risk_points=round(risk_points, 1),
            reward_points=round(reward_points, 1),
            rr_ratio=round(rr_ratio, 2),
            range_bounds=bounds,
            confidence=bounds.confidence,
            trigger=trigger,  # type: ignore[arg-type]
            grade=grade,  # type: ignore[arg-type]
        )

    @staticmethod
    def _grade_setup(bounds: RangeBounds, rr_ratio: float) -> str:
        """Assign A/B/C grade based on detection confidence and RR."""
        score = 0.0

        # Source confidence
        if bounds.source == "ob_boundaries":
            score += 0.5
        else:
            score += 0.3

        # RR scoring
        if rr_ratio >= 2.0:
            score += 0.4
        elif rr_ratio >= 1.5:
            score += 0.3
        else:
            score += 0.1

        if score >= 0.7:
            return "A"
        if score >= 0.5:
            return "B"
        return "C"

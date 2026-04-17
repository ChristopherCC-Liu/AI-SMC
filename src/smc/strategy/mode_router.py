"""Pure-function mode router: decides trending vs ranging vs v1-passthrough.

Priority logic:
  1. AI bullish/bearish + confidence >= 0.5 + session != ASIAN_CORE → trending (v1 5-gate)
  2. range_bounds exists + 5 guards pass + session in active sessions → ranging
  3. Fallback → v1_passthrough (allowed in ALL sessions including ASIAN_CORE)
"""

from __future__ import annotations

from smc.strategy.range_types import RangeBounds, TradingMode

__all__ = ["route_trading_mode"]

_RANGING_SESSIONS: frozenset[str] = frozenset({
    "LONDON",
    "LONDON/NY OVERLAP",
    "NEW YORK",
    "LATE NY",
    "ASIAN_LONDON_TRANSITION",
})


def route_trading_mode(
    ai_direction: str,
    ai_confidence: float,
    regime: str,
    session: str,
    range_bounds: RangeBounds | None,
    guards_passed: bool = False,
) -> TradingMode:
    """Decide between trending, ranging, and v1-passthrough trading mode.

    Parameters
    ----------
    ai_direction:
        "bullish", "bearish", or "neutral" from the AI direction engine.
    ai_confidence:
        Confidence score 0.0–1.0 from the AI direction engine.
    regime:
        Market regime from ``classify_regime()``: "trending", "transitional",
        or "ranging".
    session:
        Trading session from ``get_session_info()``.
    range_bounds:
        Detected range boundaries from ``RangeTrader.detect_range()``, or
        None if no range was detected.
    guards_passed:
        True if all 5 range guards pass (computed by caller via
        ``check_range_guards()``).

    Returns
    -------
    TradingMode
        Frozen model with mode, reason, and context fields.
    """
    # Priority 1: AI strong directional + NOT ASIAN_CORE → trending (v1 5-gate)
    if (
        ai_direction in ("bullish", "bearish")
        and ai_confidence >= 0.5
        and session != "ASIAN_CORE"
    ):
        return TradingMode(
            mode="trending",
            reason=f"AI {ai_direction} (conf={ai_confidence:.2f}) — trend-following mode",
            ai_direction=ai_direction,
            ai_confidence=ai_confidence,
            regime=regime,
            range_bounds=range_bounds,
        )

    # Priority 2: range detected + 5 guards pass + active session → ranging
    if range_bounds is not None and guards_passed and session in _RANGING_SESSIONS:
        return TradingMode(
            mode="ranging",
            reason=(
                f"Range detected (guards passed), session={session} — mean-reversion mode"
            ),
            ai_direction=ai_direction,
            ai_confidence=ai_confidence,
            regime=regime,
            range_bounds=range_bounds,
        )

    # Priority 3: fallback — v1 pipeline runs in all sessions including ASIAN_CORE
    return TradingMode(
        mode="v1_passthrough",
        reason=(
            f"AI unsure (dir={ai_direction}, conf={ai_confidence:.2f}) "
            f"— v1 pipeline runs with HTF bias only"
        ),
        ai_direction=ai_direction,
        ai_confidence=ai_confidence,
        regime=regime,
        range_bounds=range_bounds,
    )

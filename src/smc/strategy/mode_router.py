"""Pure-function mode router: decides trending vs ranging trading mode.

Priority logic:
  1. Asian session → HOLD (mode="trending", reason explains session block)
  2. AI bullish/bearish + confidence >= 0.5 → trending
  3. AI neutral OR confidence < 0.5:
     a. range_bounds exists AND regime != "trending" → ranging
     b. else → trending (hold — no setups expected)
"""

from __future__ import annotations

from smc.strategy.range_types import RangeBounds, TradingMode

__all__ = ["route_trading_mode"]


def route_trading_mode(
    ai_direction: str,
    ai_confidence: float,
    regime: str,
    session: str,
    range_bounds: RangeBounds | None,
) -> TradingMode:
    """Decide between trending and ranging trading mode.

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
        Trading session from ``get_session_info()``: "ASIAN", "LONDON",
        "NEW_YORK", "LONDON_CLOSE", etc.
    range_bounds:
        Detected range boundaries from ``RangeTrader.detect_range()``, or
        None if no range was detected.

    Returns
    -------
    TradingMode
        Frozen model with mode, reason, and context fields.
    """
    # Priority 1: Asian session → hold (no trading)
    if session == "ASIAN":
        return TradingMode(
            mode="trending",
            reason="Asian session — no setups generated",
            ai_direction=ai_direction,
            ai_confidence=ai_confidence,
            regime=regime,
            range_bounds=range_bounds,
        )

    # Priority 2: AI has directional conviction
    if ai_direction in ("bullish", "bearish") and ai_confidence >= 0.5:
        return TradingMode(
            mode="trending",
            reason=f"AI {ai_direction} (conf={ai_confidence:.2f}) — trend-following mode",
            ai_direction=ai_direction,
            ai_confidence=ai_confidence,
            regime=regime,
            range_bounds=range_bounds,
        )

    # Priority 3: Low conviction — check for ranging opportunity
    if range_bounds is not None and regime != "trending":
        return TradingMode(
            mode="ranging",
            reason=f"Low AI conviction (dir={ai_direction}, conf={ai_confidence:.2f}), "
            f"range detected, regime={regime} — mean-reversion mode",
            ai_direction=ai_direction,
            ai_confidence=ai_confidence,
            regime=regime,
            range_bounds=range_bounds,
        )

    # Fallback: trending hold — no actionable setups expected
    return TradingMode(
        mode="trending",
        reason=f"Low AI conviction (dir={ai_direction}, conf={ai_confidence:.2f}), "
        f"no range detected — holding, no setups expected",
        ai_direction=ai_direction,
        ai_confidence=ai_confidence,
        regime=regime,
        range_bounds=range_bounds,
    )

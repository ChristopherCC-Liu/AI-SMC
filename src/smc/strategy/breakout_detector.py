"""ATR-buffered breakout detection for range-mode exits.

A breakout is confirmed when price exceeds the range boundary by an
ATR-based buffer.  This prevents false breakouts caused by normal
volatility within the range.

Buffer formula:  ``h1_atr_points * XAUUSD_POINT_SIZE * multiplier``
Default multiplier: 0.25 (25% of H1 ATR converted to price units).
"""

from __future__ import annotations

from smc.smc_core.constants import XAUUSD_POINT_SIZE
from smc.strategy.range_types import RangeBounds

__all__ = ["BreakoutDetector"]


class BreakoutDetector:
    """Detect breakouts from a horizontal range using ATR-based buffering.

    Parameters
    ----------
    atr_buffer_mult:
        Multiplier applied to ``h1_atr_points * XAUUSD_POINT_SIZE`` to
        compute the buffer distance.  Default 0.25 means price must exceed
        the boundary by 25 % of H1 ATR (in price terms) to qualify as a
        breakout.
    """

    def __init__(self, atr_buffer_mult: float = 0.25) -> None:
        self._mult = atr_buffer_mult

    def check_breakout(
        self,
        current_price: float,
        range_bounds: RangeBounds,
        h1_atr_points: float,
    ) -> str:
        """Check whether *current_price* has broken out of *range_bounds*.

        Parameters
        ----------
        current_price:
            The latest price to evaluate.
        range_bounds:
            The detected range (upper / lower boundaries).
        h1_atr_points:
            H1 ATR expressed in **points** (e.g. 500 means $5.00 for
            XAUUSD).

        Returns
        -------
        str
            ``"bullish_breakout"`` if price is above upper + buffer,
            ``"bearish_breakout"`` if price is below lower - buffer,
            ``"none"`` otherwise.
        """
        buffer = h1_atr_points * XAUUSD_POINT_SIZE * self._mult

        if current_price > range_bounds.upper + buffer:
            return "bullish_breakout"

        if current_price < range_bounds.lower - buffer:
            return "bearish_breakout"

        return "none"

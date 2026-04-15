"""Fill model for simulating order execution in backtests.

Simulates realistic fill prices accounting for spread and slippage.
Exit checks use PESSIMISTIC logic: when both SL and TP could trigger on
the same bar, the stop loss always triggers first.

All distances are in POINTS (1 point = $0.01 for XAUUSD), not pips.
"""

from __future__ import annotations

from typing import Literal, NamedTuple

from smc.smc_core.constants import XAUUSD_POINT_SIZE


class BarOHLC(NamedTuple):
    """Minimal bar representation for fill simulation."""

    open: float
    high: float
    low: float
    close: float


class ExitResult(NamedTuple):
    """Result of an exit check: price and the reason it triggered."""

    exit_price: float
    reason: Literal["tp1", "tp2", "sl"]


class FillModel:
    """Simulates order fills with spread, slippage, and commission.

    Args:
        spread_points: Bid-ask spread in points (e.g. 3.0 = $0.03).
        slippage_points: Maximum slippage in points (e.g. 0.5 = $0.005).
        commission_per_lot: Commission charged per standard lot round-trip.
    """

    __slots__ = ("_spread_points", "_slippage_points", "_commission_per_lot")

    def __init__(
        self,
        spread_points: float,
        slippage_points: float,
        commission_per_lot: float,
    ) -> None:
        self._spread_points = spread_points
        self._slippage_points = slippage_points
        self._commission_per_lot = commission_per_lot

    @property
    def spread_points(self) -> float:
        return self._spread_points

    @property
    def slippage_points(self) -> float:
        return self._slippage_points

    @property
    def commission_per_lot(self) -> float:
        return self._commission_per_lot

    def simulate_fill(
        self,
        direction: Literal["long", "short"],
        entry_price: float,
        bar: BarOHLC,
    ) -> float | None:
        """Simulate entry fill for the given bar.

        Long entries fill at bar.open + spread + slippage (worst-case ask).
        Short entries fill at bar.open - slippage (worst-case bid slip).

        Returns the fill price, or None if the bar's range cannot
        accommodate the entry (price outside high/low).

        Args:
            direction: Trade direction.
            entry_price: Desired entry price from the signal.
            bar: OHLC bar on which the entry is attempted.

        Returns:
            Filled price or None if fill is impossible on this bar.
        """
        spread_adj = self._spread_points * XAUUSD_POINT_SIZE
        slip_adj = self._slippage_points * XAUUSD_POINT_SIZE

        if direction == "long":
            fill = bar.open + spread_adj + slip_adj
            # Fill must be achievable within the bar's range
            if fill > bar.high:
                return None
            return fill
        else:
            fill = bar.open - slip_adj
            # Fill must be achievable within the bar's range
            if fill < bar.low:
                return None
            return fill

    def check_exit(
        self,
        direction: Literal["long", "short"],
        entry_price: float,
        sl: float,
        tp1: float,
        tp2: float | None,
        bar: BarOHLC,
    ) -> ExitResult | None:
        """Check whether SL or TP is hit on the given bar.

        PESSIMISTIC rule: if both SL and any TP could trigger on the same
        bar, the stop loss is assumed to trigger first.

        For long trades:
            - SL triggers if bar.low <= sl
            - TP1 triggers if bar.high >= tp1
            - TP2 triggers if bar.high >= tp2 (when provided)

        For short trades:
            - SL triggers if bar.high >= sl
            - TP1 triggers if bar.low <= tp1
            - TP2 triggers if bar.low <= tp2 (when provided)

        Args:
            direction: Trade direction.
            entry_price: The fill price of the open trade.
            sl: Stop loss price level.
            tp1: First take profit price level.
            tp2: Second take profit price level (optional).
            bar: OHLC bar to check.

        Returns:
            ExitResult with the exit price and reason, or None if neither
            SL nor TP triggered.
        """
        sl_hit = False
        tp1_hit = False
        tp2_hit = False

        if direction == "long":
            sl_hit = bar.low <= sl
            tp1_hit = bar.high >= tp1
            tp2_hit = tp2 is not None and bar.high >= tp2
        else:
            sl_hit = bar.high >= sl
            tp1_hit = bar.low <= tp1
            tp2_hit = tp2 is not None and bar.low <= tp2

        # PESSIMISTIC: SL always wins on ambiguous bars
        if sl_hit:
            return ExitResult(exit_price=sl, reason="sl")

        # TP2 before TP1 (further target hit implies TP1 also hit)
        if tp2_hit:
            return ExitResult(exit_price=tp2, reason="tp2")  # type: ignore[arg-type]

        if tp1_hit:
            return ExitResult(exit_price=tp1, reason="tp1")

        return None


__all__ = ["FillModel", "BarOHLC", "ExitResult"]

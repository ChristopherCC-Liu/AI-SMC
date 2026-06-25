"""Round-trip cost model for the scalping / bounded-basket backtester.

WHY A NEW COST MODEL
--------------------
The legacy ``fills.py`` ``FillModel`` charges the full spread on the *entry*
fill only and applies no modelled cost on exit.  That is acceptable for a
single-shot M15 strategy holding for hundreds of points, but it is fatal for a
high-frequency scalper whose target is 5–15 points: at that scale the
bid-ask spread (3–5 pt ≈ $0.30–0.50 on XAUUSD) plus commission can eat
40–70 % of gross profit.  Cost is THE deciding variable, so it must be a
first-class, audited line item — not an entry-only approximation.

This model is symmetric and explicit:

  * every market order (entry OR exit) crosses HALF the spread and suffers
    ``slippage`` points,  →  a full round trip pays the FULL spread + 2×slip,
  * commission is charged per lot per SIDE (entry and exit each).

A "do nothing but open and immediately close at the same mid" trade therefore
loses exactly ``round_trip_cost_usd(lots)`` — this is the auditable Gate-1
property (``tests/.../test_tick_fills.py``).

All distances are POINTS (1 point = $0.01 for XAUUSD).  ``point_value_per_lot``
is dollars of P&L per 1 point of price move per 1 lot (XAUUSD = $1.00, i.e.
``pip_value_per_lot / 10``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Direction = Literal["long", "short"]


@dataclass(frozen=True, slots=True)
class CostBreakdown:
    """Decomposed USD cost of a fill or a round trip — for telemetry & gates."""

    spread_usd: float
    slippage_usd: float
    commission_usd: float

    @property
    def total_usd(self) -> float:
        return self.spread_usd + self.slippage_usd + self.commission_usd


@dataclass(frozen=True, slots=True)
class CostModel:
    """Symmetric round-trip execution-cost model.

    Args:
        spread_points: Full bid-ask spread in points (crossed half per side).
        slippage_points: Adverse slippage per market order, in points.
        commission_per_lot_per_side: Commission per lot charged on each of
            entry and exit (so a round trip pays 2×).
        point_size: Price value of one point (XAUUSD = 0.01).
        point_value_per_lot: USD P&L per 1 point per 1 lot (XAUUSD = 1.0).
    """

    spread_points: float
    slippage_points: float
    commission_per_lot_per_side: float
    point_size: float
    point_value_per_lot: float

    # -- price adjustments ---------------------------------------------------

    def entry_fill_price(self, direction: Direction, ref_price: float) -> float:
        """Worst-case entry fill: pay half-spread + slippage in the adverse
        direction (buy higher, sell lower)."""
        adj = (self.spread_points / 2.0 + self.slippage_points) * self.point_size
        return ref_price + adj if direction == "long" else ref_price - adj

    def exit_fill_price(self, direction: Direction, ref_price: float) -> float:
        """Worst-case exit fill: closing a long SELLS at bid (lower), closing a
        short BUYS at ask (higher) — half-spread + slippage either way."""
        adj = (self.spread_points / 2.0 + self.slippage_points) * self.point_size
        return ref_price - adj if direction == "long" else ref_price + adj

    # -- dollar costs --------------------------------------------------------

    def commission_usd(self, lots: float) -> float:
        """Commission for a single side."""
        return self.commission_per_lot_per_side * lots

    def one_way_cost(self, lots: float) -> CostBreakdown:
        """Cost of a single market order (entry or exit) in USD."""
        spread_usd = (self.spread_points / 2.0) * self.point_value_per_lot * lots
        slippage_usd = self.slippage_points * self.point_value_per_lot * lots
        return CostBreakdown(
            spread_usd=spread_usd,
            slippage_usd=slippage_usd,
            commission_usd=self.commission_usd(lots),
        )

    def round_trip_cost(self, lots: float) -> CostBreakdown:
        """Total USD cost of opening and closing ``lots`` (entry + exit)."""
        one = self.one_way_cost(lots)
        return CostBreakdown(
            spread_usd=one.spread_usd * 2.0,
            slippage_usd=one.slippage_usd * 2.0,
            commission_usd=one.commission_usd * 2.0,
        )

    def round_trip_cost_usd(self, lots: float) -> float:
        """Convenience: total round-trip cost in USD."""
        return self.round_trip_cost(lots).total_usd


__all__ = ["CostModel", "CostBreakdown", "Direction"]

"""Bounded, regime-gated averaging policy — the single source of truth for
basket (averaging) decisions, shared by the Python backtester and the MQL5 EA.

WHY THIS MODULE EXISTS
----------------------
The uploaded HedgeRock EA recovers losses with an *unbounded* martingale
(``GearRH`` ladder, ``CloseOnAverageLoss = 1e10`` ≈ no hard stop).  That is
exactly why grid/martingale EAs show a smooth equity curve and then blow up:
in a trend the basket averages down forever until margin is gone.

This module keeps the *useful* half of that idea — averaging into a position
to improve the basket's weighted-average entry — while making it **bounded**
and **regime-gated**:

  * averaging is only allowed when the slow brain says ``allow_averaging``
    (true ONLY in CONFIRMED_RANGE / PULLBACK regimes; false in any trend or
    high-volatility state),
  * the basket has at most ``max_layers`` layers,
  * lot growth per layer is hard-capped (NOT a martingale doubling),
  * a hard ``basket_stop_usd`` floor closes the whole basket, and
  * the invariant ``worst_case_basket_loss ≤ K × typical_win`` is enforceable
    (see :func:`validate_tail_bound`).

These are *pure functions over plain frozen dataclasses* so the exact same
decision logic can be (a) imported by ``backtest/tick_engine.py`` and (b)
ported line-for-line into the EA's ``BasketManager``.  A parity test feeds the
same fixtures to both and asserts identical decisions, guaranteeing
live == backtest.

All distances are in POINTS (1 point = $0.01 for XAUUSD).  Dollar P&L is
computed from ``point_value_per_lot`` = dollars of P&L per 1 point of price
move per 1 lot (for XAUUSD this is ``pip_value_per_lot / 10`` = $1.00).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from datetime import datetime

Direction = Literal["long", "short"]


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BasketLayer:
    """One filled entry inside a basket."""

    entry_price: float
    lots: float
    open_ts: datetime


@dataclass(frozen=True, slots=True)
class BasketState:
    """Immutable snapshot of an open basket (all layers share one direction)."""

    direction: Direction
    layers: tuple[BasketLayer, ...]

    @property
    def layer_count(self) -> int:
        return len(self.layers)

    @property
    def total_lots(self) -> float:
        return sum(layer.lots for layer in self.layers)

    @property
    def is_open(self) -> bool:
        return len(self.layers) > 0


@dataclass(frozen=True, slots=True)
class BasketParams:
    """Bounded-averaging parameters supplied by the slow brain's
    ``MarketAssessment`` and mirrored verbatim in the EA.

    Attributes
    ----------
    allow_averaging:
        HARD gate.  When False the basket may open a single layer but will
        never add a second — averaging is forbidden (trend / high-vol).
    max_layers:
        Maximum number of layers in the basket (>= 1).
    layer_spacing_points:
        Minimum adverse excursion (in points) past the *last* layer before a
        new layer may be added.  Prevents stacking layers on top of each other.
    basket_stop_usd:
        Hard floor on basket floating P&L.  When floating P&L <= -this value
        the whole basket is force-closed.  This is the tail-risk bound.
    basket_tp_usd:
        Basket take-profit on floating P&L.  When floating P&L >= this value
        the whole basket is closed in profit (HedgeRock-style close-on-average,
        but bounded and explicit).
    base_lot:
        Lot size of the first layer.
    layer_lot_mult:
        Multiplier applied to the *previous* layer's lot for the next layer.
        Default 1.0 = equal lots.  Hard-capped by ``max_layer_lot_mult`` so it
        can never become a runaway martingale.
    max_layer_lot_mult:
        Absolute cap on ``layer_lot_mult`` (defence in depth against config
        error — a value > this is clamped down, never up).
    """

    allow_averaging: bool
    max_layers: int
    layer_spacing_points: float
    basket_stop_usd: float
    basket_tp_usd: float
    base_lot: float
    layer_lot_mult: float = 1.0
    max_layer_lot_mult: float = 1.5


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def weighted_avg_entry(state: BasketState) -> float:
    """Lot-weighted average entry price of the basket (0.0 if empty)."""
    total = state.total_lots
    if total <= 0.0:
        return 0.0
    return sum(layer.entry_price * layer.lots for layer in state.layers) / total


def basket_floating_pnl_usd(
    state: BasketState,
    current_price: float,
    point_size: float,
    point_value_per_lot: float,
) -> float:
    """Floating (unrealised) P&L of the whole basket in USD.

    Long P&L grows as price rises above the weighted-average entry; short P&L
    grows as price falls below it.  Computed on the aggregate position so it
    matches how the EA evaluates the basket each tick.
    """
    if not state.is_open or point_size <= 0.0:
        return 0.0
    avg = weighted_avg_entry(state)
    move_points = (current_price - avg) / point_size
    if state.direction == "short":
        move_points = -move_points
    return move_points * point_value_per_lot * state.total_lots


def adverse_excursion_points(
    state: BasketState,
    current_price: float,
    point_size: float,
) -> float:
    """How far (in points) price has moved *against* the last layer.

    Positive = adverse (losing).  Used to enforce ``layer_spacing_points``.
    """
    if not state.is_open or point_size <= 0.0:
        return 0.0
    last = state.layers[-1].entry_price
    move_points = (current_price - last) / point_size
    # Adverse for a long means price went DOWN; for a short, price went UP.
    return -move_points if state.direction == "long" else move_points


def next_layer_lot(state: BasketState, params: BasketParams) -> float:
    """Lot size for the next layer, with a HARD cap (never a martingale).

    The multiplier is clamped to ``[0, max_layer_lot_mult]`` defensively, so a
    misconfigured ``layer_lot_mult`` can only ever shrink growth, never explode.
    """
    mult = max(0.0, min(params.layer_lot_mult, params.max_layer_lot_mult))
    if not state.is_open:
        return params.base_lot
    return state.layers[-1].lots * mult


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------


def should_add_layer(
    state: BasketState,
    current_price: float,
    params: BasketParams,
    *,
    point_size: float,
    point_value_per_lot: float,
) -> bool:
    """Decide whether to add a new averaging layer to an open basket.

    ALL of the following must hold (any one False ⇒ no add):
      1. ``params.allow_averaging`` (slow-brain hard gate — range/pullback only)
      2. basket is open and below ``max_layers``
      3. price has moved adversely at least ``layer_spacing_points`` past the
         last layer (we only average DOWN, never up into profit)
      4. the basket is not already at/through its hard stop (never throw good
         money after bad — if we're at the floor we close, not add)

    The caller is responsible for HTF-direction alignment (the EA only ever
    opens baskets in the assessment's permitted direction).
    """
    if not params.allow_averaging:
        return False
    if not state.is_open or state.layer_count >= params.max_layers:
        return False
    if adverse_excursion_points(state, current_price, point_size) < params.layer_spacing_points:
        return False
    floating = basket_floating_pnl_usd(state, current_price, point_size, point_value_per_lot)
    # Never average into a basket already at/through its hard stop.
    return floating > -params.basket_stop_usd


def should_close_basket(
    state: BasketState,
    current_price: float,
    params: BasketParams,
    *,
    point_size: float,
    point_value_per_lot: float,
) -> str | None:
    """Return a close reason, or None to hold the basket.

    Priority (pessimistic — risk first):
      1. ``"basket_stop"`` — floating P&L <= -basket_stop_usd  (HARD tail bound)
      2. ``"basket_tp"``   — floating P&L >=  basket_tp_usd    (take profit)
    """
    if not state.is_open:
        return None
    floating = basket_floating_pnl_usd(state, current_price, point_size, point_value_per_lot)
    if floating <= -params.basket_stop_usd:
        return "basket_stop"
    if params.basket_tp_usd > 0.0 and floating >= params.basket_tp_usd:
        return "basket_tp"
    return None


# ---------------------------------------------------------------------------
# Tail-risk invariant
# ---------------------------------------------------------------------------


def worst_case_basket_loss_usd(
    params: BasketParams,
    *,
    point_size: float,
    point_value_per_lot: float,
) -> float:
    """Theoretical worst-case basket loss in USD under this policy.

    Because every layer is added only on adverse excursion and the whole
    basket is force-closed the instant floating P&L breaches
    ``-basket_stop_usd``, the realised worst case is bounded by
    ``basket_stop_usd`` plus at most one ``layer_spacing`` of overshoot on the
    final fully-loaded basket (the gap between two checks before the stop
    fires).  We return that conservative upper bound so callers/tests can
    assert it against the win distribution.
    """
    # Fully-loaded basket lots (geometric series, hard-capped multiplier).
    mult = max(0.0, min(params.layer_lot_mult, params.max_layer_lot_mult))
    lots = params.base_lot
    total_lots = 0.0
    for _ in range(max(1, params.max_layers)):
        total_lots += lots
        lots *= mult if mult > 0.0 else 1.0
    overshoot_usd = params.layer_spacing_points * point_value_per_lot * total_lots
    return params.basket_stop_usd + overshoot_usd


def validate_tail_bound(
    params: BasketParams,
    *,
    typical_win_usd: float,
    k: float,
    point_size: float,
    point_value_per_lot: float,
) -> bool:
    """The core invariant: worst-case basket loss must be recoverable by at
    most ``k`` typical wins.

    ``worst_case_basket_loss ≤ k × typical_win``.  Phase 5 calibrates
    ``basket_stop_usd`` from the measured win distribution so this holds in
    every walk-forward fold; this function makes it a checkable assertion.
    """
    if typical_win_usd <= 0.0 or k <= 0.0:
        return False
    worst = worst_case_basket_loss_usd(
        params, point_size=point_size, point_value_per_lot=point_value_per_lot
    )
    return worst <= k * typical_win_usd


__all__ = [
    "Direction",
    "BasketLayer",
    "BasketState",
    "BasketParams",
    "weighted_avg_entry",
    "basket_floating_pnl_usd",
    "adverse_excursion_points",
    "next_layer_lot",
    "should_add_layer",
    "should_close_basket",
    "worst_case_basket_loss_usd",
    "validate_tail_bound",
]

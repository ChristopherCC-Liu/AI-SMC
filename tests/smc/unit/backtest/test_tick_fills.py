"""Tests for the round-trip cost model (Gate-1 cost audit).

The load-bearing property: a trade that opens and immediately closes at the
same mid price loses EXACTLY the modelled round-trip cost — nothing is hidden.
"""

from __future__ import annotations

import pytest

from smc.backtest.tick_fills import CostModel


# XAUUSD-like cost model: 3pt spread, 0.5pt slippage, $3.50/lot/side commission
# (=$7 round trip), 1 point = $0.01, $1.00 P&L per point per lot.
def _xau_model() -> CostModel:
    return CostModel(
        spread_points=3.0,
        slippage_points=0.5,
        commission_per_lot_per_side=3.5,
        point_size=0.01,
        point_value_per_lot=1.0,
    )


def test_round_trip_cost_decomposition() -> None:
    m = _xau_model()
    cost = m.round_trip_cost(lots=1.0)
    # full spread crossed once across the round trip: 3.0 pt * $1 * 1 lot
    assert cost.spread_usd == pytest.approx(3.0)
    # slippage on both sides: 2 * 0.5 pt * $1 * 1 lot
    assert cost.slippage_usd == pytest.approx(1.0)
    # commission both sides: 2 * $3.50
    assert cost.commission_usd == pytest.approx(7.0)
    assert cost.total_usd == pytest.approx(11.0)
    assert m.round_trip_cost_usd(1.0) == pytest.approx(11.0)


def test_open_close_at_same_mid_loses_exactly_round_trip_cost() -> None:
    """THE audit: flat round-trip at one price == -round_trip_cost."""
    m = _xau_model()
    mid = 2000.0
    lots = 1.0

    entry = m.entry_fill_price("long", mid)
    exit_ = m.exit_fill_price("long", mid)

    # P&L from the bid-ask + slippage portion (in USD)
    move_points = (exit_ - entry) / m.point_size
    price_pnl_usd = move_points * m.point_value_per_lot * lots
    commission_usd = 2.0 * m.commission_usd(lots)
    net = price_pnl_usd - commission_usd

    assert net == pytest.approx(-m.round_trip_cost_usd(lots))
    assert net == pytest.approx(-11.0)


def test_cost_scales_linearly_with_lots() -> None:
    m = _xau_model()
    assert m.round_trip_cost_usd(2.0) == pytest.approx(2.0 * m.round_trip_cost_usd(1.0))


def test_fill_prices_are_adverse() -> None:
    m = _xau_model()
    mid = 2000.0
    # Long pays up on entry, sells down on exit.
    assert m.entry_fill_price("long", mid) > mid
    assert m.exit_fill_price("long", mid) < mid
    # Short sells down on entry, buys up on exit.
    assert m.entry_fill_price("short", mid) < mid
    assert m.exit_fill_price("short", mid) > mid


def test_cost_drag_is_severe_for_a_small_scalp() -> None:
    """A 5pt scalp on 1 lot grosses $5 but the round trip costs $11 → the
    strategy is net-negative on cost alone.  This is the whole reason cost is a
    first-class gate."""
    m = _xau_model()
    gross_target_usd = 5.0 * m.point_value_per_lot * 1.0  # 5 points * $1 * 1 lot
    cost = m.round_trip_cost_usd(1.0)
    assert cost > gross_target_usd  # $11 > $5
    cost_drag_pct = cost / gross_target_usd
    assert cost_drag_pct > 1.0

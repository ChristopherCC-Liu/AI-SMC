"""Tests for the bounded, regime-gated averaging policy.

Covers the decision gates and — most importantly — the tail-risk invariant
``worst_case_basket_loss <= K * typical_win`` that keeps this from becoming a
HedgeRock-style unbounded martingale.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from smc.execution.basket_policy import (
    BasketLayer,
    BasketParams,
    BasketState,
    adverse_excursion_points,
    basket_floating_pnl_usd,
    next_layer_lot,
    should_add_layer,
    should_close_basket,
    validate_tail_bound,
    weighted_avg_entry,
    worst_case_basket_loss_usd,
)

_TS = datetime(2024, 1, 1, tzinfo=UTC)
_POINT = 0.01
_PV = 1.0  # $ per point per lot (XAUUSD)


def _long(*prices_lots: tuple[float, float]) -> BasketState:
    layers = tuple(BasketLayer(entry_price=p, lots=lot, open_ts=_TS) for p, lot in prices_lots)
    return BasketState(direction="long", layers=layers)


def _close(state: BasketState, price: float, p: BasketParams) -> str | None:
    return should_close_basket(state, price, p, point_size=_POINT, point_value_per_lot=_PV)


def _params(**overrides: object) -> BasketParams:
    base = dict(
        allow_averaging=True,
        max_layers=3,
        layer_spacing_points=500.0,
        basket_stop_usd=50.0,
        basket_tp_usd=20.0,
        base_lot=0.1,
        layer_lot_mult=1.0,
        max_layer_lot_mult=1.5,
    )
    base.update(overrides)
    return BasketParams(**base)  # type: ignore[arg-type]


# --- helpers ----------------------------------------------------------------


def test_weighted_avg_entry() -> None:
    state = _long((2000.0, 1.0), (1990.0, 1.0))
    assert weighted_avg_entry(state) == pytest.approx(1995.0)


def test_weighted_avg_entry_empty() -> None:
    assert weighted_avg_entry(BasketState("long", ())) == 0.0


def test_floating_pnl_long() -> None:
    state = _long((2000.0, 1.0), (1990.0, 1.0))  # avg 1995, total 2 lots
    # price up to 2000 → +500 points * $1 * 2 lots = +$1000
    assert basket_floating_pnl_usd(state, 2000.0, _POINT, _PV) == pytest.approx(1000.0)
    # price down to 1990 → -500 points * $1 * 2 = -$1000
    assert basket_floating_pnl_usd(state, 1990.0, _POINT, _PV) == pytest.approx(-1000.0)


def test_adverse_excursion_long() -> None:
    state = _long((2000.0, 1.0))
    # price 5.00 below last layer = 500 points adverse for a long
    assert adverse_excursion_points(state, 1995.0, _POINT) == pytest.approx(500.0)
    # price above = negative adverse (favourable)
    assert adverse_excursion_points(state, 2005.0, _POINT) == pytest.approx(-500.0)


def test_next_layer_lot_equal_by_default() -> None:
    state = _long((2000.0, 0.1))
    assert next_layer_lot(state, _params()) == pytest.approx(0.1)


def test_next_layer_lot_multiplier_is_hard_capped() -> None:
    state = _long((2000.0, 0.1))
    # request 5x growth but cap is 1.5x → 0.15, never 0.5
    p = _params(layer_lot_mult=5.0, max_layer_lot_mult=1.5)
    assert next_layer_lot(state, p) == pytest.approx(0.15)


# --- should_add_layer gates -------------------------------------------------


def test_no_add_when_averaging_disabled() -> None:
    state = _long((2000.0, 0.1))
    p = _params(allow_averaging=False)
    assert should_add_layer(state, 1990.0, p, point_size=_POINT, point_value_per_lot=_PV) is False


def test_no_add_at_max_layers() -> None:
    state = _long((2000.0, 0.1), (1995.0, 0.1), (1990.0, 0.1))
    p = _params(max_layers=3)
    assert should_add_layer(state, 1980.0, p, point_size=_POINT, point_value_per_lot=_PV) is False


def test_no_add_before_spacing_met() -> None:
    state = _long((2000.0, 0.1))
    p = _params(layer_spacing_points=500.0)
    # only 100 points adverse, need 500
    assert should_add_layer(state, 1999.0, p, point_size=_POINT, point_value_per_lot=_PV) is False


def test_add_when_spacing_met_and_aligned() -> None:
    state = _long((2000.0, 0.1))
    p = _params(layer_spacing_points=500.0, basket_stop_usd=1e9)
    # 500 points adverse, averaging allowed, below max layers, not at stop
    assert should_add_layer(state, 1995.0, p, point_size=_POINT, point_value_per_lot=_PV) is True


def test_no_add_when_already_at_hard_stop() -> None:
    # Big lots so a modest adverse move blows the stop.
    state = _long((2000.0, 1.0))
    p = _params(layer_spacing_points=100.0, basket_stop_usd=50.0)
    # 100 points adverse * $1 * 1 lot = -$100 floating <= -$50 stop → must NOT add
    assert should_add_layer(state, 1999.0, p, point_size=_POINT, point_value_per_lot=_PV) is False


# --- should_close_basket ----------------------------------------------------


def test_close_on_hard_stop() -> None:
    state = _long((2000.0, 1.0))
    p = _params(basket_stop_usd=50.0, basket_tp_usd=20.0)
    # -100 points * $1 * 1 lot = -$100 <= -$50
    assert _close(state, 1999.0, p) == "basket_stop"


def test_close_on_take_profit() -> None:
    state = _long((2000.0, 1.0))
    p = _params(basket_stop_usd=50.0, basket_tp_usd=20.0)
    # +30 points * $1 = +$30 >= $20
    assert _close(state, 2000.3, p) == "basket_tp"


def test_hold_inside_band() -> None:
    state = _long((2000.0, 1.0))
    p = _params(basket_stop_usd=50.0, basket_tp_usd=20.0)
    # +5 points = +$5, between -$50 and +$20 → hold
    assert _close(state, 2000.05, p) is None


def test_stop_takes_priority_over_tp() -> None:
    # Degenerate config where both could fire — risk first.
    state = _long((2000.0, 1.0))
    p = _params(basket_stop_usd=5.0, basket_tp_usd=5.0)
    assert _close(state, 1999.9, p) == "basket_stop"


# --- tail-risk invariant ----------------------------------------------------


def test_worst_case_loss_bounded_by_stop_plus_one_spacing() -> None:
    p = _params(max_layers=3, layer_spacing_points=100.0, basket_stop_usd=50.0,
                base_lot=0.1, layer_lot_mult=1.0)
    worst = worst_case_basket_loss_usd(p, point_size=_POINT, point_value_per_lot=_PV)
    # stop ($50) + one spacing overshoot on full basket (0.3 lots): 100 * $1 * 0.3 = $30
    assert worst == pytest.approx(50.0 + 30.0)


def test_tail_bound_holds_for_sane_config() -> None:
    # basket_stop $50, typical win $30, allow up to K=3 wins to recover.
    p = _params(max_layers=2, layer_spacing_points=50.0, basket_stop_usd=50.0,
                base_lot=0.1, layer_lot_mult=1.0)
    assert validate_tail_bound(
        p, typical_win_usd=30.0, k=3.0, point_size=_POINT, point_value_per_lot=_PV
    ) is True


def test_tail_bound_fails_for_reckless_stop() -> None:
    # A $500 basket stop cannot be recovered by 3 x $30 wins.
    p = _params(max_layers=3, layer_spacing_points=100.0, basket_stop_usd=500.0)
    assert validate_tail_bound(
        p, typical_win_usd=30.0, k=3.0, point_size=_POINT, point_value_per_lot=_PV
    ) is False

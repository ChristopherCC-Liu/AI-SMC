"""Unit tests for mt5_positions_adapter — Round 5 T1 F2."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from smc.execution.mt5_positions_adapter import (
    XAUUSD_MAGIC,
    fetch_broker_positions,
    fetch_closed_pnl_since,
)
from smc.execution.types import PositionState


class FakeMT5:
    """Minimal stand-in for the MetaTrader5 module."""

    POSITION_TYPE_BUY = 0
    POSITION_TYPE_SELL = 1
    DEAL_ENTRY_IN = 0
    DEAL_ENTRY_OUT = 1
    DEAL_ENTRY_INOUT = 2

    def __init__(self, positions=None, deals=None, tick=None, raise_on_positions=False):
        self._positions = positions or []
        self._deals = deals or []
        self._tick = tick
        self._raise_on_positions = raise_on_positions

    def positions_get(self, *, symbol):  # noqa: ARG002 — test stub
        if self._raise_on_positions:
            raise RuntimeError("mt5 unreachable")
        return tuple(self._positions)

    def history_deals_get(self, from_time, to_time):  # noqa: ARG002
        return tuple(self._deals)

    def symbol_info_tick(self, symbol):  # noqa: ARG002
        return self._tick


def _mk_position(ticket=1, magic=XAUUSD_MAGIC, direction="long", profit=12.5):
    return SimpleNamespace(
        ticket=ticket,
        magic=magic,
        symbol="XAUUSD",
        type=FakeMT5.POSITION_TYPE_BUY if direction == "long" else FakeMT5.POSITION_TYPE_SELL,
        volume=0.01,
        price_open=2350.0,
        price_current=2362.5,
        sl=2340.0,
        tp=2370.0,
        profit=profit,
        time=1_700_000_000,
    )


def _mk_deal(ticket=99, magic=XAUUSD_MAGIC, entry=FakeMT5.DEAL_ENTRY_OUT,
             profit=5.0, commission=-0.25, swap=-0.05, ts=1_700_001_000):
    return SimpleNamespace(
        position_id=ticket,
        ticket=ticket,
        magic=magic,
        entry=entry,
        profit=profit,
        commission=commission,
        swap=swap,
        time=ts,
    )


class TestFetchBrokerPositions:
    def test_empty_when_no_positions(self):
        mt5 = FakeMT5(positions=[])
        assert fetch_broker_positions(mt5) == ()

    def test_projects_single_position(self):
        mt5 = FakeMT5(
            positions=[_mk_position(ticket=42)],
            tick=SimpleNamespace(bid=2363.1, ask=2363.3),
        )
        out = fetch_broker_positions(mt5)
        assert len(out) == 1
        p = out[0]
        assert isinstance(p, PositionState)
        assert p.ticket == 42
        assert p.instrument == "XAUUSD"
        assert p.direction == "long"
        assert p.lots == 0.01
        assert p.open_price == 2350.0
        # long → use bid
        assert p.current_price == 2363.1
        assert p.sl == 2340.0
        assert p.tp == 2370.0
        assert p.pnl_usd == 12.5
        assert p.open_time.tzinfo is timezone.utc

    def test_filters_other_magic(self):
        mt5 = FakeMT5(positions=[
            _mk_position(ticket=1, magic=XAUUSD_MAGIC),
            _mk_position(ticket=2, magic=9999999),  # some other EA
        ])
        out = fetch_broker_positions(mt5)
        tickets = [p.ticket for p in out]
        assert tickets == [1]

    def test_short_uses_ask_for_current_price(self):
        mt5 = FakeMT5(
            positions=[_mk_position(ticket=7, direction="short")],
            tick=SimpleNamespace(bid=2363.1, ask=2363.3),
        )
        out = fetch_broker_positions(mt5)
        assert out[0].direction == "short"
        assert out[0].current_price == 2363.3

    def test_empty_on_exception(self):
        mt5 = FakeMT5(raise_on_positions=True)
        assert fetch_broker_positions(mt5) == ()

    def test_falls_back_to_price_current_when_tick_missing(self):
        mt5 = FakeMT5(positions=[_mk_position(ticket=3)], tick=None)
        out = fetch_broker_positions(mt5)
        assert out[0].current_price == 2362.5


class TestFetchClosedPnlSince:
    def test_empty_when_no_deals(self):
        mt5 = FakeMT5(deals=[])
        from_t = datetime(2026, 4, 18, tzinfo=timezone.utc)
        assert fetch_closed_pnl_since(mt5, from_t) == []

    def test_sums_profit_commission_swap(self):
        mt5 = FakeMT5(deals=[_mk_deal(profit=10.0, commission=-0.3, swap=-0.1)])
        from_t = datetime(2026, 4, 18, tzinfo=timezone.utc)
        out = fetch_closed_pnl_since(mt5, from_t)
        assert len(out) == 1
        assert out[0]["ticket"] == 99
        assert out[0]["pnl_usd"] == round(10.0 - 0.3 - 0.1, 2)

    def test_skips_entry_in_deals(self):
        """A DEAL_ENTRY_IN (open) should not count as a close."""
        mt5 = FakeMT5(deals=[
            _mk_deal(ticket=1, entry=FakeMT5.DEAL_ENTRY_IN, profit=0.0),
            _mk_deal(ticket=2, entry=FakeMT5.DEAL_ENTRY_OUT, profit=5.0),
        ])
        from_t = datetime(2026, 4, 18, tzinfo=timezone.utc)
        out = fetch_closed_pnl_since(mt5, from_t)
        tickets = [d["ticket"] for d in out]
        assert tickets == [2]

    def test_filters_other_magic(self):
        mt5 = FakeMT5(deals=[
            _mk_deal(ticket=1, magic=XAUUSD_MAGIC),
            _mk_deal(ticket=2, magic=777),
        ])
        from_t = datetime(2026, 4, 18, tzinfo=timezone.utc)
        out = fetch_closed_pnl_since(mt5, from_t)
        tickets = [d["ticket"] for d in out]
        assert tickets == [1]

    def test_empty_on_exception(self):
        class Boom:
            def history_deals_get(self, *a, **k):
                raise RuntimeError("boom")
        from_t = datetime(2026, 4, 18, tzinfo=timezone.utc)
        assert fetch_closed_pnl_since(Boom(), from_t) == []

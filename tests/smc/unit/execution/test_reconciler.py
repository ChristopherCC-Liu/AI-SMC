"""Tests for smc.execution.reconciler — position reconciliation."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from smc.execution.reconciler import reconcile
from smc.execution.types import PositionState


def _make_position(
    ticket: int,
    pnl: float = 0.0,
    direction: str = "long",
) -> PositionState:
    """Helper to create a PositionState with minimal boilerplate."""
    return PositionState(
        ticket=ticket,
        instrument="XAUUSD",
        direction=direction,
        lots=0.01,
        open_price=2350.0,
        current_price=2355.0,
        sl=2340.0,
        tp=2370.0,
        pnl_usd=pnl,
        open_time=datetime.now(tz=timezone.utc),
    )


@pytest.mark.unit
class TestReconcileInSync:
    """When local and broker agree, result is in_sync."""

    def test_both_empty(self) -> None:
        result = reconcile({}, ())
        assert result.in_sync is True

    def test_matching_positions(self) -> None:
        p1 = _make_position(1001, pnl=5.0)
        p2 = _make_position(1002, pnl=-3.0)
        result = reconcile(
            {1001: p1, 1002: p2},
            (p1, p2),
        )
        assert result.in_sync is True
        assert result.summary == "IN_SYNC"


@pytest.mark.unit
class TestReconcileClosedByBroker:
    """Detect positions we think are open but broker has closed."""

    def test_single_closed(self) -> None:
        local = {1001: _make_position(1001)}
        result = reconcile(local, ())
        assert not result.in_sync
        assert 1001 in result.closed_by_broker

    def test_mixed_closed_and_open(self) -> None:
        p1 = _make_position(1001)
        p2 = _make_position(1002)
        local = {1001: p1, 1002: p2}
        # Only p2 still at broker
        result = reconcile(local, (p2,))
        assert 1001 in result.closed_by_broker
        assert 1002 not in result.closed_by_broker


@pytest.mark.unit
class TestReconcileUntracked:
    """Detect positions at broker that we don't track locally."""

    def test_orphaned_position(self) -> None:
        broker_pos = _make_position(9999)
        result = reconcile({}, (broker_pos,))
        assert not result.in_sync
        assert 9999 in result.untracked

    def test_multiple_orphans(self) -> None:
        p1 = _make_position(8001)
        p2 = _make_position(8002)
        result = reconcile({}, (p1, p2))
        assert 8001 in result.untracked
        assert 8002 in result.untracked


@pytest.mark.unit
class TestReconcilePnlMismatch:
    """Detect PnL divergence beyond tolerance."""

    def test_within_tolerance(self) -> None:
        local = _make_position(1001, pnl=5.0)
        broker = _make_position(1001, pnl=5.30)
        result = reconcile({1001: local}, (broker,))
        assert result.in_sync is True

    def test_beyond_tolerance(self) -> None:
        local = _make_position(1001, pnl=5.0)
        broker = _make_position(1001, pnl=10.0)
        result = reconcile({1001: local}, (broker,))
        assert not result.in_sync
        assert 1001 in result.pnl_mismatches

    def test_custom_tolerance(self) -> None:
        local = _make_position(1001, pnl=5.0)
        broker = _make_position(1001, pnl=5.80)
        # Default tolerance = 0.50, should flag
        result = reconcile({1001: local}, (broker,))
        assert 1001 in result.pnl_mismatches

        # With higher tolerance, should pass
        result2 = reconcile({1001: local}, (broker,), pnl_tolerance=1.0)
        assert result2.in_sync is True

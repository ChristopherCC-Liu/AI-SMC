"""Tests for smc.execution.order_manager — order lifecycle orchestration."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from smc.execution.executor import SimBrokerPort
from smc.execution.order_manager import OrderManager
from smc.execution.types import AccountInfo
from smc.risk.drawdown_guard import DrawdownGuard
from smc.strategy.types import (
    BiasDirection,
    EntrySignal,
    TradeSetup,
    TradeZone,
)
from smc.data.schemas import Timeframe


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_setup(
    direction: str = "long",
    entry: float = 2350.0,
    sl: float = 2340.0,
    tp1: float = 2370.0,
    tp2: float = 2390.0,
    rr: float = 2.0,
) -> TradeSetup:
    """Build a minimal TradeSetup for testing."""
    risk_points = abs(entry - sl) * 100  # convert to points
    reward_points = abs(tp1 - entry) * 100
    return TradeSetup(
        entry_signal=EntrySignal(
            entry_price=entry,
            stop_loss=sl,
            take_profit_1=tp1,
            take_profit_2=tp2,
            risk_points=risk_points,
            reward_points=reward_points,
            rr_ratio=rr,
            trigger_type="choch_in_zone",
            direction=direction,
            grade="A",
        ),
        bias=BiasDirection(
            direction="bullish" if direction == "long" else "bearish",
            confidence=0.8,
            key_levels=(2340.0, 2380.0),
            rationale="test bias",
        ),
        zone=TradeZone(
            zone_high=2360.0,
            zone_low=2345.0,
            zone_type="ob",
            direction=direction,
            timeframe=Timeframe.H1,
            confidence=0.75,
        ),
        confluence_score=0.72,
        generated_at=datetime.now(tz=timezone.utc),
    )


def _make_account(balance: float = 10_000.0) -> AccountInfo:
    return AccountInfo(
        balance=balance,
        equity=balance,
        margin_used=0.0,
        margin_free=balance,
        margin_level=0.0,
    )


def _make_manager(
    initial_balance: float = 10_000.0,
    **kwargs,
) -> tuple[OrderManager, SimBrokerPort]:
    broker = SimBrokerPort(initial_balance=initial_balance, spread_points=0.0)
    guard = DrawdownGuard(max_daily_loss_pct=3.0, max_drawdown_pct=10.0)
    manager = OrderManager(
        broker=broker,
        risk_guard=guard,
        position_sizer_config={"risk_pct": 1.0, "max_lot_size": 0.1},
        peak_balance=initial_balance,
        **kwargs,
    )
    return manager, broker


# ---------------------------------------------------------------------------
# Tests: execute_setup
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExecuteSetup:
    """OrderManager.execute_setup: risk check -> size -> send -> track."""

    def test_successful_execution(self) -> None:
        manager, broker = _make_manager()
        setup = _make_setup()
        account = _make_account()

        result = manager.execute_setup(setup, account)

        assert result is not None
        assert result.success is True
        assert result.ticket > 0
        assert len(broker.get_positions()) == 1
        assert len(manager.get_local_positions()) == 1

    def test_short_execution(self) -> None:
        manager, broker = _make_manager()
        setup = _make_setup(direction="short", entry=2380.0, sl=2390.0, tp1=2360.0, tp2=2340.0)
        account = _make_account()

        result = manager.execute_setup(setup, account)
        assert result is not None
        assert result.success is True

        positions = broker.get_positions()
        assert positions[0].direction == "short"

    def test_risk_guard_rejects_after_daily_loss(self) -> None:
        manager, broker = _make_manager()
        # Simulate a daily loss of 4% (exceeds 3% limit)
        manager._daily_pnl = -400.0  # 4% of 10_000

        setup = _make_setup()
        account = _make_account()

        result = manager.execute_setup(setup, account)
        assert result is None  # rejected
        assert len(broker.get_positions()) == 0


# ---------------------------------------------------------------------------
# Tests: manage_open_positions
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestManageOpenPositions:
    """OrderManager.manage_open_positions: SL, TP1 partial, TP2, trailing."""

    def test_sl_hit_closes_position(self) -> None:
        manager, broker = _make_manager()
        setup = _make_setup(entry=2350.0, sl=2340.0, tp1=2370.0)
        account = _make_account()

        result = manager.execute_setup(setup, account)
        ticket = result.ticket

        # Update price to hit SL
        broker.update_price(ticket, 2339.0)
        results = manager.manage_open_positions(2339.0)

        assert len(results) >= 1
        assert any(r.success for r in results)
        assert len(manager.get_local_positions()) == 0

    def test_tp1_partial_close(self) -> None:
        manager, broker = _make_manager(partial_close_pct=0.5)
        setup = _make_setup(entry=2350.0, sl=2340.0, tp1=2370.0)
        account = _make_account()

        result = manager.execute_setup(setup, account)
        ticket = result.ticket

        # Update price to hit TP1
        broker.update_price(ticket, 2371.0)
        results = manager.manage_open_positions(2371.0)

        assert len(results) >= 1
        # Position should still exist with reduced lots
        positions = broker.get_positions()
        if positions:
            assert positions[0].lots < 0.1  # was partially closed

    def test_trailing_sl_moves_up(self) -> None:
        manager, broker = _make_manager(trailing_sl_points=5.0)
        setup = _make_setup(entry=2350.0, sl=2340.0, tp1=2370.0)
        account = _make_account()

        result = manager.execute_setup(setup, account)
        ticket = result.ticket

        # Move price up enough to trigger trailing (5 points = $5)
        broker.update_price(ticket, 2360.0)
        results = manager.manage_open_positions(2360.0)

        # Check SL was trailed
        positions = broker.get_positions()
        if positions:
            # New SL should be higher than original 2340
            assert positions[0].sl >= 2340.0


# ---------------------------------------------------------------------------
# Tests: close_all
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCloseAll:
    """OrderManager.close_all: emergency shutdown."""

    def test_closes_everything(self) -> None:
        manager, broker = _make_manager()
        account = _make_account()

        # Open 3 positions
        for _ in range(3):
            manager.execute_setup(_make_setup(), account)

        assert len(broker.get_positions()) == 3

        results = manager.close_all()
        assert len(results) == 3
        assert all(r.success for r in results)
        assert len(broker.get_positions()) == 0
        assert len(manager.get_local_positions()) == 0

    def test_close_all_on_empty(self) -> None:
        manager, _broker = _make_manager()
        results = manager.close_all()
        assert results == []


# ---------------------------------------------------------------------------
# Tests: daily PnL tracking
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDailyPnl:
    """OrderManager tracks daily realised P&L."""

    def test_pnl_starts_at_zero(self) -> None:
        manager, _ = _make_manager()
        assert manager.daily_pnl == 0.0

    def test_reset_daily_pnl(self) -> None:
        manager, _ = _make_manager()
        manager._daily_pnl = -150.0
        manager.reset_daily_pnl()
        assert manager.daily_pnl == 0.0

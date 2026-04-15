"""Tests for smc.execution.types — frozen Pydantic models."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from smc.execution.types import (
    AccountInfo,
    OrderRequest,
    OrderResult,
    PositionState,
    ReconciliationResult,
)


@pytest.mark.unit
class TestOrderRequest:
    """OrderRequest is frozen and has sensible defaults."""

    def test_default_instrument(self) -> None:
        req = OrderRequest(direction="long", lots=0.01, stop_loss=2340.0, take_profit_1=2370.0)
        assert req.instrument == "XAUUSD"

    def test_default_order_type_is_market(self) -> None:
        req = OrderRequest(direction="short", lots=0.05, stop_loss=2380.0, take_profit_1=2350.0)
        assert req.order_type == "market"
        assert req.entry_price == 0.0

    def test_frozen(self) -> None:
        req = OrderRequest(direction="long", lots=0.01, stop_loss=2340.0, take_profit_1=2370.0)
        with pytest.raises(Exception):
            req.lots = 0.02  # type: ignore[misc]

    def test_optional_tp2(self) -> None:
        req = OrderRequest(direction="long", lots=0.01, stop_loss=2340.0, take_profit_1=2370.0)
        assert req.take_profit_2 is None

        req2 = OrderRequest(
            direction="long", lots=0.01, stop_loss=2340.0,
            take_profit_1=2370.0, take_profit_2=2390.0,
        )
        assert req2.take_profit_2 == 2390.0


@pytest.mark.unit
class TestOrderResult:
    """OrderResult defaults to failure state."""

    def test_failure_defaults(self) -> None:
        result = OrderResult(success=False)
        assert result.ticket == 0
        assert result.fill_price == 0.0
        assert result.error_message == ""

    def test_success_with_fields(self) -> None:
        result = OrderResult(success=True, ticket=12345, fill_price=2355.15)
        assert result.success is True
        assert result.ticket == 12345

    def test_frozen(self) -> None:
        result = OrderResult(success=True, ticket=1)
        with pytest.raises(Exception):
            result.success = False  # type: ignore[misc]


@pytest.mark.unit
class TestPositionState:
    """PositionState is a frozen snapshot of an open position."""

    def test_roundtrip(self) -> None:
        now = datetime.now(tz=timezone.utc)
        pos = PositionState(
            ticket=1001,
            instrument="XAUUSD",
            direction="long",
            lots=0.05,
            open_price=2350.0,
            current_price=2355.0,
            sl=2340.0,
            tp=2370.0,
            pnl_usd=25.0,
            open_time=now,
        )
        assert pos.ticket == 1001
        assert pos.direction == "long"
        assert pos.pnl_usd == 25.0

    def test_frozen(self) -> None:
        now = datetime.now(tz=timezone.utc)
        pos = PositionState(
            ticket=1, instrument="XAUUSD", direction="short",
            lots=0.01, open_price=2380.0, current_price=2375.0,
            sl=2390.0, tp=2360.0, pnl_usd=5.0, open_time=now,
        )
        with pytest.raises(Exception):
            pos.pnl_usd = 10.0  # type: ignore[misc]


@pytest.mark.unit
class TestAccountInfo:
    """AccountInfo captures a broker account snapshot."""

    def test_fields(self) -> None:
        info = AccountInfo(
            balance=10_000.0, equity=10_250.0,
            margin_used=400.0, margin_free=9_850.0,
            margin_level=2562.5,
        )
        assert info.balance == 10_000.0
        assert info.margin_level == 2562.5


@pytest.mark.unit
class TestReconciliationResult:
    """ReconciliationResult tracks divergences."""

    def test_in_sync(self) -> None:
        result = ReconciliationResult(in_sync=True, summary="IN_SYNC")
        assert result.in_sync is True
        assert result.closed_by_broker == ()
        assert result.untracked == ()

    def test_with_divergences(self) -> None:
        result = ReconciliationResult(
            in_sync=False,
            closed_by_broker=(1001,),
            untracked=(2001,),
            pnl_mismatches=(3001,),
            summary="DIVERGED",
        )
        assert not result.in_sync
        assert 1001 in result.closed_by_broker

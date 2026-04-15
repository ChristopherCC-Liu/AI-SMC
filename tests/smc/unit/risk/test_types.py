"""Unit tests for risk management data types."""

from __future__ import annotations

import pytest

from smc.risk.types import PositionSize, RiskBudget


# ---------------------------------------------------------------------------
# PositionSize
# ---------------------------------------------------------------------------


class TestPositionSize:
    def test_creation(self) -> None:
        ps = PositionSize(
            lots=0.33,
            risk_usd=100.0,
            risk_points=300.0,
            margin_required_usd=660.0,
        )
        assert ps.lots == 0.33
        assert ps.risk_usd == 100.0
        assert ps.risk_points == 300.0
        assert ps.margin_required_usd == 660.0

    def test_frozen(self) -> None:
        ps = PositionSize(lots=0.1, risk_usd=50.0, risk_points=200.0, margin_required_usd=200.0)
        with pytest.raises(Exception):
            ps.lots = 0.5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# RiskBudget
# ---------------------------------------------------------------------------


class TestRiskBudget:
    def test_can_trade(self) -> None:
        budget = RiskBudget(
            can_trade=True,
            available_risk_pct=3.0,
            used_risk_pct=0.0,
            daily_loss_pct=0.0,
            total_drawdown_pct=0.0,
        )
        assert budget.can_trade is True
        assert budget.rejection_reason is None

    def test_rejected(self) -> None:
        budget = RiskBudget(
            can_trade=False,
            available_risk_pct=0.0,
            used_risk_pct=3.0,
            daily_loss_pct=3.5,
            total_drawdown_pct=2.0,
            rejection_reason="Daily loss limit breached",
        )
        assert budget.can_trade is False
        assert budget.rejection_reason is not None

    def test_frozen(self) -> None:
        budget = RiskBudget(
            can_trade=True,
            available_risk_pct=3.0,
            used_risk_pct=0.0,
            daily_loss_pct=0.0,
            total_drawdown_pct=0.0,
        )
        with pytest.raises(Exception):
            budget.can_trade = False  # type: ignore[misc]

    def test_default_rejection_reason_is_none(self) -> None:
        budget = RiskBudget(
            can_trade=True,
            available_risk_pct=3.0,
            used_risk_pct=0.0,
            daily_loss_pct=0.0,
            total_drawdown_pct=0.0,
        )
        assert budget.rejection_reason is None

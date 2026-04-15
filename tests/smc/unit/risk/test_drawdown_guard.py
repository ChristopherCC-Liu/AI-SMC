"""Unit tests for the DrawdownGuard circuit breaker."""

from __future__ import annotations

import pytest

from smc.risk.drawdown_guard import DrawdownGuard


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestDrawdownGuardInit:
    def test_default_thresholds(self) -> None:
        guard = DrawdownGuard()
        # Verify it constructs without error (thresholds are private)
        budget = guard.check_budget(balance=10_000.0, peak_balance=10_000.0, daily_pnl=0.0)
        assert budget.can_trade is True

    def test_custom_thresholds(self) -> None:
        guard = DrawdownGuard(max_daily_loss_pct=5.0, max_drawdown_pct=15.0)
        budget = guard.check_budget(balance=10_000.0, peak_balance=10_000.0, daily_pnl=0.0)
        assert budget.can_trade is True

    def test_zero_daily_loss_pct_raises(self) -> None:
        with pytest.raises(ValueError, match="max_daily_loss_pct"):
            DrawdownGuard(max_daily_loss_pct=0.0)

    def test_negative_drawdown_pct_raises(self) -> None:
        with pytest.raises(ValueError, match="max_drawdown_pct"):
            DrawdownGuard(max_drawdown_pct=-1.0)


# ---------------------------------------------------------------------------
# Daily loss circuit breaker
# ---------------------------------------------------------------------------


class TestDailyLoss:
    def test_no_loss_can_trade(self) -> None:
        guard = DrawdownGuard(max_daily_loss_pct=3.0)
        budget = guard.check_budget(balance=10_000.0, peak_balance=10_000.0, daily_pnl=0.0)
        assert budget.can_trade is True
        assert budget.daily_loss_pct == 0.0

    def test_small_loss_can_trade(self) -> None:
        guard = DrawdownGuard(max_daily_loss_pct=3.0)
        budget = guard.check_budget(balance=10_000.0, peak_balance=10_000.0, daily_pnl=-200.0)
        assert budget.can_trade is True
        assert budget.daily_loss_pct == pytest.approx(2.0)

    def test_exactly_at_limit_halts(self) -> None:
        """Daily loss exactly at 3% should halt trading (>= check)."""
        guard = DrawdownGuard(max_daily_loss_pct=3.0)
        budget = guard.check_budget(balance=10_000.0, peak_balance=10_000.0, daily_pnl=-300.0)
        assert budget.can_trade is False
        assert "Daily loss limit" in (budget.rejection_reason or "")

    def test_exceeds_limit_halts(self) -> None:
        guard = DrawdownGuard(max_daily_loss_pct=3.0)
        budget = guard.check_budget(balance=10_000.0, peak_balance=10_000.0, daily_pnl=-500.0)
        assert budget.can_trade is False

    def test_positive_pnl_is_fine(self) -> None:
        """A profitable day should not trigger the daily loss guard."""
        guard = DrawdownGuard(max_daily_loss_pct=3.0)
        budget = guard.check_budget(balance=10_000.0, peak_balance=10_000.0, daily_pnl=500.0)
        assert budget.can_trade is True
        assert budget.daily_loss_pct == 0.0

    def test_available_risk_decreases_with_loss(self) -> None:
        guard = DrawdownGuard(max_daily_loss_pct=3.0)
        budget = guard.check_budget(balance=10_000.0, peak_balance=10_000.0, daily_pnl=-100.0)
        assert budget.available_risk_pct == pytest.approx(2.0)
        assert budget.used_risk_pct == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Total drawdown circuit breaker
# ---------------------------------------------------------------------------


class TestTotalDrawdown:
    def test_no_drawdown_can_trade(self) -> None:
        guard = DrawdownGuard(max_drawdown_pct=10.0)
        budget = guard.check_budget(balance=10_000.0, peak_balance=10_000.0, daily_pnl=0.0)
        assert budget.can_trade is True
        assert budget.total_drawdown_pct == 0.0

    def test_5pct_drawdown_can_trade(self) -> None:
        guard = DrawdownGuard(max_drawdown_pct=10.0)
        budget = guard.check_budget(balance=9_500.0, peak_balance=10_000.0, daily_pnl=0.0)
        assert budget.can_trade is True
        assert budget.total_drawdown_pct == pytest.approx(5.0)

    def test_exactly_10pct_drawdown_halts(self) -> None:
        """10% drawdown exactly at limit should halt trading (>= check)."""
        guard = DrawdownGuard(max_drawdown_pct=10.0)
        budget = guard.check_budget(balance=9_000.0, peak_balance=10_000.0, daily_pnl=0.0)
        assert budget.can_trade is False
        assert "Max drawdown" in (budget.rejection_reason or "")

    def test_exceeds_10pct_drawdown_halts(self) -> None:
        guard = DrawdownGuard(max_drawdown_pct=10.0)
        budget = guard.check_budget(balance=8_500.0, peak_balance=10_000.0, daily_pnl=0.0)
        assert budget.can_trade is False

    def test_drawdown_takes_priority_over_daily(self) -> None:
        """When both limits breached, drawdown rejection is reported."""
        guard = DrawdownGuard(max_daily_loss_pct=3.0, max_drawdown_pct=10.0)
        budget = guard.check_budget(balance=8_000.0, peak_balance=10_000.0, daily_pnl=-500.0)
        assert budget.can_trade is False
        assert "Max drawdown" in (budget.rejection_reason or "")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestDrawdownGuardEdgeCases:
    def test_zero_balance(self) -> None:
        guard = DrawdownGuard()
        budget = guard.check_budget(balance=0.0, peak_balance=10_000.0, daily_pnl=0.0)
        assert budget.can_trade is False
        assert "zero or negative" in (budget.rejection_reason or "").lower()

    def test_negative_balance(self) -> None:
        guard = DrawdownGuard()
        budget = guard.check_budget(balance=-100.0, peak_balance=10_000.0, daily_pnl=0.0)
        assert budget.can_trade is False

    def test_peak_below_balance_uses_balance_as_peak(self) -> None:
        """If peak_balance < balance (data error), use balance as effective peak."""
        guard = DrawdownGuard(max_drawdown_pct=10.0)
        budget = guard.check_budget(balance=10_000.0, peak_balance=9_000.0, daily_pnl=0.0)
        # effective_peak = max(9000, 10000) = 10000; drawdown = 0%
        assert budget.can_trade is True
        assert budget.total_drawdown_pct == 0.0

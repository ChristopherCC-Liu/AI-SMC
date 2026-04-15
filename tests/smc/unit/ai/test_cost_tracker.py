"""Unit tests for smc.ai.cost_tracker — daily LLM spend tracking."""

from __future__ import annotations

import pytest

from smc.ai.cost_tracker import CostTracker


class TestCostTracker:
    def test_initial_state(self) -> None:
        tracker = CostTracker(daily_budget_usd=7.0, burst_budget_usd=2.0)
        assert tracker.daily_spend == 0.0
        assert tracker.classification_count == 0
        assert tracker.remaining_budget == 7.0

    def test_can_classify_under_budget(self) -> None:
        tracker = CostTracker(daily_budget_usd=7.0)
        assert tracker.can_classify() is True

    def test_record_spend(self) -> None:
        tracker = CostTracker(daily_budget_usd=7.0)
        tracker.record_spend(0.78)
        assert tracker.daily_spend == pytest.approx(0.78)
        assert tracker.classification_count == 1
        assert tracker.remaining_budget == pytest.approx(6.22)

    def test_cumulative_spend(self) -> None:
        tracker = CostTracker(daily_budget_usd=7.0)
        tracker.record_spend(0.78)
        tracker.record_spend(0.78)
        tracker.record_spend(0.78)
        assert tracker.daily_spend == pytest.approx(2.34)
        assert tracker.classification_count == 3

    def test_budget_exhausted(self) -> None:
        tracker = CostTracker(daily_budget_usd=1.0, burst_budget_usd=0.5)
        tracker.record_spend(1.01)
        assert tracker.can_classify() is False

    def test_burst_budget(self) -> None:
        tracker = CostTracker(daily_budget_usd=1.0, burst_budget_usd=0.5)
        tracker.record_spend(1.01)
        assert tracker.can_classify() is False
        assert tracker.can_burst_classify() is True

    def test_burst_budget_also_exhausted(self) -> None:
        tracker = CostTracker(daily_budget_usd=1.0, burst_budget_usd=0.5)
        tracker.record_spend(1.6)
        assert tracker.can_classify() is False
        assert tracker.can_burst_classify() is False

    def test_negative_spend_raises(self) -> None:
        tracker = CostTracker()
        with pytest.raises(ValueError, match="cost_usd must be >= 0"):
            tracker.record_spend(-1.0)

    def test_zero_spend_allowed(self) -> None:
        tracker = CostTracker()
        tracker.record_spend(0.0)
        assert tracker.daily_spend == 0.0
        assert tracker.classification_count == 1

    def test_reset(self) -> None:
        tracker = CostTracker(daily_budget_usd=7.0)
        tracker.record_spend(5.0)
        tracker.reset()
        assert tracker.daily_spend == 0.0
        assert tracker.classification_count == 0
        assert tracker.can_classify() is True

    def test_default_budgets(self) -> None:
        tracker = CostTracker()
        assert tracker.daily_budget == 7.0
        assert tracker.burst_budget == 2.0

    def test_custom_budgets(self) -> None:
        tracker = CostTracker(daily_budget_usd=10.0, burst_budget_usd=3.0)
        assert tracker.daily_budget == 10.0
        assert tracker.burst_budget == 3.0

    def test_remaining_budget_never_negative(self) -> None:
        tracker = CostTracker(daily_budget_usd=1.0)
        tracker.record_spend(5.0)
        assert tracker.remaining_budget == 0.0

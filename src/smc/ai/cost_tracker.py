"""Daily LLM spend tracking for AI regime classification.

Tracks cumulative spend per calendar day (UTC) against a configurable
budget.  When the budget is exhausted, the regime classifier silently
falls back to the ATR-based classifier.

Thread-safe via a simple lock — the tracker is called from the
aggregator's ``generate_setups()`` which runs on a single thread, but
the lock protects against accidental concurrent use.

Usage::

    tracker = CostTracker(daily_budget_usd=7.0, burst_budget_usd=2.0)
    if tracker.can_classify():
        # run LLM classification ...
        tracker.record_spend(0.78)
    else:
        # use ATR fallback
        ...
"""

from __future__ import annotations

import threading
from datetime import date, timezone
from datetime import datetime as dt

__all__ = ["CostTracker"]


class CostTracker:
    """Tracks daily LLM spend against a configurable budget.

    Parameters
    ----------
    daily_budget_usd:
        Maximum normal daily spend in USD.  Default $7.00.
    burst_budget_usd:
        Extra budget for event-driven (NFP, FOMC) classifications
        beyond the normal H4 schedule.  Default $2.00.
    """

    def __init__(
        self,
        daily_budget_usd: float = 7.0,
        burst_budget_usd: float = 2.0,
    ) -> None:
        self._daily_budget = daily_budget_usd
        self._burst_budget = burst_budget_usd
        self._lock = threading.Lock()
        self._current_date: date = dt.now(tz=timezone.utc).date()
        self._daily_spend: float = 0.0
        self._classification_count: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def daily_spend(self) -> float:
        """Total LLM spend today in USD."""
        with self._lock:
            self._maybe_reset_day()
            return self._daily_spend

    @property
    def daily_budget(self) -> float:
        """Configured daily budget in USD."""
        return self._daily_budget

    @property
    def burst_budget(self) -> float:
        """Configured burst budget in USD."""
        return self._burst_budget

    @property
    def classification_count(self) -> int:
        """Number of classifications performed today."""
        with self._lock:
            self._maybe_reset_day()
            return self._classification_count

    @property
    def remaining_budget(self) -> float:
        """Remaining normal budget for today in USD."""
        with self._lock:
            self._maybe_reset_day()
            return max(0.0, self._daily_budget - self._daily_spend)

    # ------------------------------------------------------------------
    # Budget checks
    # ------------------------------------------------------------------

    def can_classify(self) -> bool:
        """Check if the normal daily budget allows another classification."""
        with self._lock:
            self._maybe_reset_day()
            return self._daily_spend < self._daily_budget

    def can_burst_classify(self) -> bool:
        """Check if the burst budget allows an extra classification.

        Burst budget is additive on top of the daily budget — used for
        event-driven classifications (NFP, FOMC).
        """
        with self._lock:
            self._maybe_reset_day()
            return self._daily_spend < (self._daily_budget + self._burst_budget)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_spend(self, cost_usd: float) -> None:
        """Record an LLM classification cost.

        Parameters
        ----------
        cost_usd:
            The cost of the classification in USD.  Must be >= 0.
        """
        if cost_usd < 0:
            raise ValueError(f"cost_usd must be >= 0, got {cost_usd}")
        with self._lock:
            self._maybe_reset_day()
            self._daily_spend += cost_usd
            self._classification_count += 1

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Force-reset the daily counters.  Used in testing."""
        with self._lock:
            self._current_date = dt.now(tz=timezone.utc).date()
            self._daily_spend = 0.0
            self._classification_count = 0

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _maybe_reset_day(self) -> None:
        """Reset counters if the UTC date has changed.  Caller holds lock."""
        today = dt.now(tz=timezone.utc).date()
        if today != self._current_date:
            self._current_date = today
            self._daily_spend = 0.0
            self._classification_count = 0

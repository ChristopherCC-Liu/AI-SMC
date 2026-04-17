"""Asian ranging one-per-day cooldown tracker.

Rationale: Asian session (incl. ASIAN_LONDON_TRANSITION) ranging setups are
rare and high-risk. Cap at 1 open per UTC day to prevent regime-flip cascades.

LONDON/NY sessions are NOT subject to this quota — high liquidity allows
multiple setups per day.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone

__all__ = ["AsianRangeQuota"]


@dataclass(frozen=True)
class AsianRangeQuota:
    """Immutable tracker of last Asian ranging open date.

    Usage:
        quota = AsianRangeQuota()
        if quota.is_exhausted_today(datetime.now(tz=timezone.utc)):
            return None  # skip setup
        # ... open trade ...
        quota = quota.record_open(datetime.now(tz=timezone.utc))
    """
    last_open_date: date | None = None

    def is_exhausted_today(self, now_utc: datetime) -> bool:
        """True iff an Asian range was already opened on now_utc's UTC date."""
        if self.last_open_date is None:
            return False
        return self.last_open_date == now_utc.astimezone(timezone.utc).date()

    def record_open(self, now_utc: datetime) -> "AsianRangeQuota":
        """Return new quota with today's UTC date recorded."""
        today = now_utc.astimezone(timezone.utc).date()
        return AsianRangeQuota(last_open_date=today)

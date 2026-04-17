"""Asian ranging one-per-day cooldown tracker.

Rationale: Asian session (incl. ASIAN_LONDON_TRANSITION) ranging setups are
rare and high-risk. Cap at 1 open per UTC day to prevent regime-flip cascades.

LONDON/NY sessions are NOT subject to this quota — high liquidity allows
multiple setups per day.

Round 4.6-H2: adds JSON persistence so process restarts do not reset the
daily cap and allow a second open on the same UTC day.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

__all__ = ["AsianRangeQuota", "DEFAULT_QUOTA_STATE_PATH"]

DEFAULT_QUOTA_STATE_PATH = Path("data/asian_range_quota_state.json")


@dataclass(frozen=True)
class AsianRangeQuota:
    """Immutable tracker of last Asian ranging open date.

    Usage:
        quota = AsianRangeQuota.load()  # Round 4.6-H2: restore across restarts
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

    def record_open(
        self,
        now_utc: datetime,
        state_path: Path | None = None,
    ) -> "AsianRangeQuota":
        """Return new quota with today's UTC date recorded.

        Round 4.6-H2: also persists the new state to JSON so process
        restarts don't silently reset the cap. Callers that want to opt
        out of persistence can pass state_path explicitly via load(None).
        """
        today = now_utc.astimezone(timezone.utc).date()
        new_quota = AsianRangeQuota(last_open_date=today)
        if state_path is None:
            state_path = DEFAULT_QUOTA_STATE_PATH
        try:
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text(
                json.dumps({"last_open_date": today.isoformat()})
            )
        except Exception:
            # persistence is best-effort; in-memory state still correct
            pass
        return new_quota

    @classmethod
    def load(cls, state_path: Path | None = None) -> "AsianRangeQuota":
        """Round 4.6-H2: restore quota from JSON, or fresh if no file.

        Corrupt/missing files yield a fresh quota (fail-open). This keeps
        paper trading tolerant of dev-env noise while giving live runs a
        durable daily cap.
        """
        if state_path is None:
            state_path = DEFAULT_QUOTA_STATE_PATH
        if not state_path.exists():
            return cls()
        try:
            raw = json.loads(state_path.read_text())
            iso = raw.get("last_open_date")
            if not iso:
                return cls()
            return cls(last_open_date=date.fromisoformat(iso))
        except Exception:
            return cls()

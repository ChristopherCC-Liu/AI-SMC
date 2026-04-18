"""Consecutive-loss daily halt — Round 5 T1 F3.

User directive (2026-04-18): "亏 3 单自动停" — if 3 losing trades in a row,
halt trading for the rest of the UTC day. Any WIN between resets the counter.

Pattern matches :class:`Phase1aCircuitBreaker` (daily UTC 00:00 auto-reset,
JSON-persisted state) so operators only have one mental model to learn. The
key difference is *consecutive* semantics — this counter cares about streaks,
not cumulative PnL.

State is persisted to ``data/consec_loss_state.json`` so the halt survives
process restarts.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

__all__ = ["ConsecLossHalt", "ConsecLossState"]

DEFAULT_STATE_PATH = Path("data/consec_loss_state.json")
# audit-r3 R4: default limit preserved for backward-compat; real value now
# comes from cfg.consec_loss_limit per-instrument (XAU=3, BTC=3 today).
_DEFAULT_CONSEC_LIMIT = 3


@dataclass(frozen=True)
class ConsecLossState:
    """Immutable snapshot of the halt state."""
    consec_losses: int = 0
    last_pnl_usd: float = 0.0
    tripped: bool = False
    tripped_at: str | None = None
    last_updated: str | None = None
    last_reset_date: str | None = None


class ConsecLossHalt:
    """Halts trading after N consecutive losing trades, auto-resets daily.

    Call :meth:`record` after every trade closes with the realised PnL in USD.
    Check :meth:`is_tripped` before opening a new trade.

    Daily UTC 00:00 rollover clears both the streak and the trip flag, so the
    halt is "today only" rather than permanent.
    """

    def __init__(
        self,
        state_path: Path | str = DEFAULT_STATE_PATH,
        *,
        consec_limit: int = _DEFAULT_CONSEC_LIMIT,
    ) -> None:
        if consec_limit < 1:
            raise ValueError(f"consec_limit must be >= 1, got {consec_limit}")
        self._state_path = Path(state_path)
        self._consec_limit = int(consec_limit)
        self._state = self._load_state()

    def _load_state(self) -> ConsecLossState:
        if not self._state_path.exists():
            return ConsecLossState(last_reset_date=_today_utc_iso())
        try:
            raw = json.loads(self._state_path.read_text())
            return ConsecLossState(**raw)
        except Exception:
            return ConsecLossState(last_reset_date=_today_utc_iso())

    def _save_state(self) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._state_path.write_text(json.dumps(asdict(self._state), indent=2))

    def _apply_daily_reset_if_needed(self) -> None:
        today_iso = _today_utc_iso()
        if self._state.last_reset_date != today_iso:
            self._state = ConsecLossState(last_reset_date=today_iso)
            self._save_state()

    def is_tripped(self) -> bool:
        self._apply_daily_reset_if_needed()
        return self._state.tripped

    def snapshot(self) -> ConsecLossState:
        self._apply_daily_reset_if_needed()
        return self._state

    def record(self, pnl_usd: float) -> ConsecLossState:
        """Record a trade close. A LOSS (pnl<0) increments the streak,
        a WIN or break-even (pnl>=0) resets it to 0.

        Trips when the streak reaches _CONSEC_LIMIT. Once tripped, subsequent
        records are no-ops until the next daily reset.
        """
        self._apply_daily_reset_if_needed()

        if self._state.tripped:
            return self._state

        if pnl_usd < 0:
            new_streak = self._state.consec_losses + 1
        else:
            new_streak = 0

        now_iso = datetime.now(tz=timezone.utc).isoformat()
        trip = new_streak >= self._consec_limit

        self._state = ConsecLossState(
            consec_losses=new_streak,
            last_pnl_usd=round(pnl_usd, 2),
            tripped=trip,
            tripped_at=now_iso if trip else None,
            last_updated=now_iso,
            last_reset_date=self._state.last_reset_date,
        )
        self._save_state()
        return self._state

    def reset(self) -> ConsecLossState:
        """Manual reset — operator override (daily reset handles normal case)."""
        self._state = ConsecLossState(last_reset_date=_today_utc_iso())
        self._save_state()
        return self._state


def _today_utc_iso() -> str:
    return datetime.now(tz=timezone.utc).date().isoformat()

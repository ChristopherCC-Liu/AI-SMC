"""Rolling-window loss halt — Round 10 P3.2 hygiene.

Pre-R10 design used a daily UTC reset: 3 consecutive losses halt trading
until the next UTC midnight. The reset window proved too generous —
operators saw the bot resume after midnight even when the underlying
market regime hadn't changed, creating a "fresh streak after midnight"
illusion.

R10 P3.2 replaces the daily reset with a rolling-window streak: any
``consec_limit`` losses (default 3) inside the most recent
``window_size`` trades (default 6) trips the halt. Wins inside the
window decay the loss count gradually as they push old losses out;
operator must :meth:`reset` manually OR accumulate enough wins to evict
all losses from the window before trading resumes.

State persistence pattern matches the pre-R10 implementation
(``@dataclass(frozen=True)`` + JSON checkpoint) so operators see
familiar telemetry. Legacy state files (with ``last_reset_date`` /
``consec_losses`` int field) are migrated on load with a WARN log.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

__all__ = ["ConsecLossHalt", "ConsecLossState"]

DEFAULT_STATE_PATH = Path("data/consec_loss_state.json")
_DEFAULT_CONSEC_LIMIT = 3
_DEFAULT_WINDOW_SIZE = 6

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConsecLossState:
    """Immutable snapshot of the halt state.

    ``recent_outcomes`` is a tuple of booleans (True = loss, False =
    non-loss) bounded to the most recent ``window_size`` trades.
    ``consec_losses`` is exposed as a read-only @property over the
    window for backward compatibility with pre-R10 callers / tests
    that read ``snapshot().consec_losses``.
    """

    recent_outcomes: Tuple[bool, ...] = field(default_factory=tuple)
    last_pnl_usd: float = 0.0
    tripped: bool = False
    tripped_at: str | None = None
    last_updated: str | None = None

    @property
    def consec_losses(self) -> int:
        """Trailing consecutive losses from the most recent trade backward.

        Backward-compat alias for the pre-R10 ``consec_losses`` int field.
        Pre-R10 callers (e.g. ``live_demo.py:1041`` halt-reason string
        template, snapshots in monitor/guards_snapshot.py) read this as
        "how many losses in a row right now?", so we count from the tail
        of the window backward until we hit a non-loss. This differs from
        "total losses inside the window" (which is what trips the halt);
        use ``recent_outcomes`` directly for window-aware semantics.
        """
        count = 0
        for outcome in reversed(self.recent_outcomes):
            if outcome:
                count += 1
            else:
                break
        return count

    @property
    def loss_count_in_window(self) -> int:
        """Total losses inside the rolling window (drives the trip flag)."""
        return sum(1 for outcome in self.recent_outcomes if outcome)


class ConsecLossHalt:
    """Halt trading after ``consec_limit`` losses inside ``window_size`` trades.

    Call :meth:`record` after every trade closes with the realised PnL in
    USD. Check :meth:`is_tripped` before opening a new trade.

    Once tripped, the halt stays active until either:
    - :meth:`reset` is called explicitly (operator override), OR
    - Subsequent wins push the rolling window's loss count below
      ``consec_limit``.

    NOTE — semantic change vs pre-R10: there is NO automatic UTC-midnight
    reset. Operators must take action (or wait for organic recovery via
    wins) for the halt to clear. This is intentional: a fresh midnight
    should not paper over a losing regime.
    """

    def __init__(
        self,
        state_path: Path | str = DEFAULT_STATE_PATH,
        *,
        consec_limit: int = _DEFAULT_CONSEC_LIMIT,
        window_size: int = _DEFAULT_WINDOW_SIZE,
    ) -> None:
        if consec_limit < 1:
            raise ValueError(f"consec_limit must be >= 1, got {consec_limit}")
        if window_size < consec_limit:
            raise ValueError(
                f"window_size ({window_size}) must be >= consec_limit "
                f"({consec_limit}); a halt that can never be tripped "
                "would be a silent dead-letter."
            )
        self._state_path = Path(state_path)
        self._consec_limit = int(consec_limit)
        self._window_size = int(window_size)
        self._state = self._load_state()

    # ------------------------------------------------------------------
    # State persistence (with legacy migration)
    # ------------------------------------------------------------------

    def _load_state(self) -> ConsecLossState:
        if not self._state_path.exists():
            return ConsecLossState()
        try:
            raw = json.loads(self._state_path.read_text())
        except Exception:
            return ConsecLossState()

        # Legacy R5 schema (pre-R10): {consec_losses: int, last_reset_date: ...}.
        # Migrate by synthesising recent_outcomes as `[True] * consec_losses`.
        # This is a deliberately approximate rebuild — we lost the original
        # ordering but preserve the tripped state, which is what matters
        # for the next is_tripped() call.
        if "recent_outcomes" not in raw:
            legacy_losses = int(raw.get("consec_losses", 0))
            synth = (True,) * min(legacy_losses, self._window_size)
            tripped = bool(raw.get("tripped", False))
            _logger.warning(
                "consec_loss_state_legacy_migrated path=%s legacy_losses=%d "
                "synth_recent_outcomes_len=%d tripped=%s",
                self._state_path,
                legacy_losses,
                len(synth),
                tripped,
            )
            return ConsecLossState(
                recent_outcomes=synth,
                last_pnl_usd=float(raw.get("last_pnl_usd", 0.0)),
                tripped=tripped,
                tripped_at=raw.get("tripped_at"),
                last_updated=raw.get("last_updated"),
            )

        # R10 schema.
        outcomes_raw = raw.get("recent_outcomes") or []
        outcomes = tuple(bool(x) for x in outcomes_raw)
        return ConsecLossState(
            recent_outcomes=outcomes,
            last_pnl_usd=float(raw.get("last_pnl_usd", 0.0)),
            tripped=bool(raw.get("tripped", False)),
            tripped_at=raw.get("tripped_at"),
            last_updated=raw.get("last_updated"),
        )

    def _save_state(self) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        # asdict on a frozen dataclass emits the field tuple as a list
        # which is the right JSON shape; @property consec_losses is NOT
        # an asdict field, so it correctly stays out of the serialised
        # payload (it's purely a derived view).
        payload = {
            "recent_outcomes": list(self._state.recent_outcomes),
            "last_pnl_usd": self._state.last_pnl_usd,
            "tripped": self._state.tripped,
            "tripped_at": self._state.tripped_at,
            "last_updated": self._state.last_updated,
        }
        self._state_path.write_text(json.dumps(payload, indent=2))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_tripped(self) -> bool:
        return self._state.tripped

    def snapshot(self) -> ConsecLossState:
        return self._state

    def record(self, pnl_usd: float) -> ConsecLossState:
        """Record a trade close. Append outcome to the rolling window
        (truncated to ``window_size``) and re-evaluate the trip flag.

        A LOSS (``pnl<0``) is appended as ``True``; a WIN or break-even
        (``pnl>=0``) is appended as ``False``. Once the halt is tripped,
        subsequent records are no-ops — operator reset / wins evicting
        losses from the window are the only paths back to is_tripped()=False.
        """
        if self._state.tripped:
            return self._state

        is_loss = pnl_usd < 0
        new_window = self._state.recent_outcomes + (is_loss,)
        if len(new_window) > self._window_size:
            new_window = new_window[-self._window_size:]

        loss_count = sum(1 for x in new_window if x)
        now_iso = datetime.now(tz=timezone.utc).isoformat()
        trip = loss_count >= self._consec_limit

        # Use replace-equivalent (frozen dataclass) to construct the new state.
        self._state = ConsecLossState(
            recent_outcomes=new_window,
            last_pnl_usd=round(pnl_usd, 2),
            tripped=trip,
            tripped_at=now_iso if trip else None,
            last_updated=now_iso,
        )
        self._save_state()
        return self._state

    def reset(self) -> ConsecLossState:
        """Operator override — clear the rolling window and trip flag."""
        self._state = ConsecLossState()
        self._save_state()
        return self._state

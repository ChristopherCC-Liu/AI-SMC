"""Persistent multi-day drawdown circuit breaker — Round 10 P1.1.

Wraps the stateless :class:`smc.risk.drawdown_guard.DrawdownGuard` with a
persisted equity peak so the guard reasons about *all-time* (and rolling
weekly) drawdown rather than a 24h-rolling cap.

Why this exists
---------------
Pre-R10, ``scripts/live_demo.py`` reset ``peak_balance`` to the current
balance on every UTC 00:00 boundary. With the underlying 10% threshold
that turned the guard into a daily-rolling cap: on 2026-04-25 the demo
account sat at ``-12.76%`` over 5 days but no halt fired because the peak
ratcheted down with the bleed each midnight.

This module restores monotonic peak semantics. Once we set an all-time
high, it never decreases. A parallel rolling 5-day rail catches faster
multi-session bleeds without a single bad week permanently freezing the
account.

Halt rules (applied on top of the wrapped DrawdownGuard's own daily-loss
trip-wire)::

    equity / persisted_peak <= 0.90  ->  halt new opens for 24h
    equity / persisted_peak <= 0.85  ->  halt new opens until manual reset

Both halts only **block_opens**; existing positions continue to be managed
by their own SL/TP/trail. A future ratio<=0.80 emergency tier will be the
only path that sets ``force_close=True``.

The 0.85 cliff is *sticky*: it requires the operator to drop a sentinel
file (``data/dd_manual_reset{suffix}.flag``) for the guard to clear. The
sentinel is consumed atomically (unlinked once observed) so the same flag
cannot accidentally clear two consecutive halts. Any text content in the
sentinel is captured as the operator note and persisted into the audit
history.

Audit trail
-----------
Every alarm and cliff trip appends a :class:`HaltEvent` to
``manual_halt_history``. The event records *when* the halt opened, *what*
the equity / peak / ratio looked like, *which rail* fired, and (after
clearing) *when* and *with what note* the halt cleared. The list is
append-only — the JSON file is the source of truth for post-mortem
forensics.

State persistence mirrors :mod:`smc.risk.consec_loss_halt` —
``@dataclass(frozen=True)`` snapshot serialised as JSON, loaded best-effort
(corrupt files reset to a fresh state rather than crashing the trading
loop).
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field, replace
from datetime import date as _date
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from smc.risk.drawdown_guard import DrawdownGuard
from smc.risk.types import RiskBudget

_logger = logging.getLogger(__name__)

# V3-B (R10 Adopt #1): mirror foundation's IO failure counter pattern from
# `smc.monitor.gate_diagnostic_journal._failure_counter`. The original
# ``_save_state`` was silent-fail on disk-full / readonly filesystem; this
# layer escalates persistent IO trouble from WARNING (transient) to ERROR
# (operator-actionable) at the third failure on the same UTC day. Same shape
# as the foundation pattern so a future reader recognises both.
#
# No threading.Lock() — PersistentDrawdownGuard runs single-threaded per leg
# (one process per arm). Foundation's gate_diagnostic_journal pattern uses
# a Lock because its write path could race in a multi-thread context; here
# that constraint doesn't apply. Mirror in shape, simplify by scope.
_SAVE_FAIL_ESCALATE_AFTER = 3
_save_failure_counter: dict[_date, int] = {}


def reset_save_failure_counter() -> None:
    """Clear the per-UTC-day save failure counter — exposed for tests."""
    _save_failure_counter.clear()


def _record_save_failure(day: _date, exc: BaseException, **fields: Any) -> None:
    """Increment failure count for ``day`` and log warn / error accordingly."""
    _save_failure_counter[day] = _save_failure_counter.get(day, 0) + 1
    count = _save_failure_counter[day]
    log_fn = _logger.error if count >= _SAVE_FAIL_ESCALATE_AFTER else _logger.warning
    log_fn(
        "persistent_dd_save_failed (%d/%d) — disk full or readonly?",
        count,
        _SAVE_FAIL_ESCALATE_AFTER,
        extra={
            "event": "persistent_dd_save_failed",
            "failure_count": count,
            "utc_date": str(day),
            **fields,
        },
        exc_info=exc,
    )


def _record_save_success(day: _date) -> None:
    """Reset the failure counter for ``day`` after a successful write."""
    _save_failure_counter.pop(day, None)


__all__ = [
    "PersistentDrawdownGuard",
    "PersistentPeakState",
    "HaltEvent",
    "DEFAULT_WEEKLY_WINDOW_DAYS",
    "ALARM_RATIO",
    "CLIFF_RATIO",
    "AUTO_RESUME_HOURS",
    "reset_save_failure_counter",
]


DEFAULT_WEEKLY_WINDOW_DAYS = 5
ALARM_RATIO = 0.90
CLIFF_RATIO = 0.85
AUTO_RESUME_HOURS = 24


@dataclass(frozen=True)
class HaltEvent:
    """Immutable audit record for one halt cycle (open + close).

    Attributes
    ----------
    tripped_at:
        UTC ISO-8601 timestamp when the halt opened.
    tier:
        ``"alarm"`` (0.90 24h auto-resume) or ``"cliff"`` (0.85 manual).
    rail:
        ``"all-time"`` or ``"weekly"`` — which peak rail fired.
    equity_at_trip:
        Account balance at the moment the halt opened.
    peak_at_trip:
        Persisted peak (of the firing rail) at the moment the halt opened.
    ratio_at_trip:
        ``equity_at_trip / peak_at_trip``, rounded to 4 dp.
    cleared_at:
        UTC ISO-8601 when the halt cleared (None while still active).
        Auto-filled for alarm trips when the 24h window passes; written
        when the operator drops a sentinel file for cliff trips.
    operator_note:
        Free-text content read from the sentinel file when cleared
        (None for alarm trips, which auto-resume without operator action).
    """

    tripped_at: str
    tier: str
    rail: str
    equity_at_trip: float
    peak_at_trip: float
    ratio_at_trip: float
    cleared_at: Optional[str] = None
    operator_note: Optional[str] = None


@dataclass(frozen=True)
class PersistentPeakState:
    """Immutable snapshot of the persisted peak rails."""

    all_time_peak: float = 0.0
    # Rolling 5-day weekly rail: list of (utc_date_iso, max_balance_that_day).
    # update() shifts entries older than the window out and refreshes today.
    weekly_balances: list = field(default_factory=list)
    manual_halt_active: bool = False
    last_halt_at: Optional[str] = None
    last_updated: Optional[str] = None
    # Append-only audit trail of all halt cycles.
    manual_halt_history: list = field(default_factory=list)

    @property
    def weekly_peak(self) -> float:
        """Max balance across the rolling weekly window (0.0 if empty)."""
        if not self.weekly_balances:
            return 0.0
        return max(float(entry[1]) for entry in self.weekly_balances)


class PersistentDrawdownGuard:
    """All-time + rolling-week drawdown circuit breaker with persistence.

    The guard owns the equity peak so the caller no longer has to. It
    composes the existing stateless :class:`DrawdownGuard` for the daily
    loss trip-wire (which is independent of multi-day drawdown).

    Parameters
    ----------
    state_path:
        JSON file backing :class:`PersistentPeakState`.
    manual_reset_sentinel_path:
        Operator-controlled file that clears the sticky 0.85 halt. Any
        text content is captured into the audit log as ``operator_note``.
    inner_guard:
        Optional pre-built daily-loss guard. Defaults to the production
        ``DrawdownGuard(max_daily_loss_pct=3.0, max_drawdown_pct=10.0)``;
        we still use its daily-loss check, but its drawdown_pct field is
        ignored because we run our own dual-rail logic.
    weekly_window_days:
        Width of the rolling weekly peak rail (default 5 trading days).
    alarm_ratio:
        ``equity/peak`` threshold for the 24h auto-resume halt.
    cliff_ratio:
        ``equity/peak`` threshold for the manual-reset-required halt.
    auto_resume_hours:
        Halt duration for the 0.90 alarm before auto-resume.
    clock:
        Test seam — callable returning ``datetime`` aware in UTC.
    """

    def __init__(
        self,
        state_path: Path | str,
        manual_reset_sentinel_path: Path | str,
        *,
        inner_guard: DrawdownGuard | None = None,
        weekly_window_days: int = DEFAULT_WEEKLY_WINDOW_DAYS,
        alarm_ratio: float = ALARM_RATIO,
        cliff_ratio: float = CLIFF_RATIO,
        auto_resume_hours: int = AUTO_RESUME_HOURS,
        clock=None,
    ) -> None:
        if weekly_window_days < 1:
            raise ValueError(
                f"weekly_window_days must be >= 1, got {weekly_window_days}"
            )
        if not 0.0 < cliff_ratio < alarm_ratio < 1.0:
            raise ValueError(
                "ratios must satisfy 0 < cliff_ratio < alarm_ratio < 1, "
                f"got cliff={cliff_ratio} alarm={alarm_ratio}"
            )
        if auto_resume_hours < 1:
            raise ValueError(
                f"auto_resume_hours must be >= 1, got {auto_resume_hours}"
            )

        self._state_path = Path(state_path)
        self._sentinel_path = Path(manual_reset_sentinel_path)
        self._inner = inner_guard or DrawdownGuard(
            max_daily_loss_pct=3.0,
            max_drawdown_pct=10.0,
        )
        self._weekly_window_days = weekly_window_days
        self._alarm_ratio = alarm_ratio
        self._cliff_ratio = cliff_ratio
        self._auto_resume = timedelta(hours=auto_resume_hours)
        self._clock = clock or _utc_now
        self._state = self._load_state()

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _load_state(self) -> PersistentPeakState:
        if not self._state_path.exists():
            return PersistentPeakState()
        try:
            raw = json.loads(self._state_path.read_text())
        except Exception:
            # Corrupt file — fail safe to a fresh state rather than crash.
            return PersistentPeakState()
        history_raw = raw.get("manual_halt_history") or []
        history = [HaltEvent(**ev) for ev in history_raw]
        weekly = [tuple(entry) for entry in (raw.get("weekly_balances") or [])]
        return PersistentPeakState(
            all_time_peak=float(raw.get("all_time_peak", 0.0)),
            weekly_balances=weekly,
            manual_halt_active=bool(raw.get("manual_halt_active", False)),
            last_halt_at=raw.get("last_halt_at"),
            last_updated=raw.get("last_updated"),
            manual_halt_history=history,
        )

    def _save_state(self) -> None:
        # V3-B: wrap the IO write in a fail-closed escalation counter
        # (mirrors smc.monitor.gate_diagnostic_journal pattern). The
        # trading loop never crashes on disk-full / readonly fs; the
        # third consecutive failure on a UTC day promotes to ERROR.
        day = self._clock().date()
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "all_time_peak": self._state.all_time_peak,
                "weekly_balances": [
                    list(entry) for entry in self._state.weekly_balances
                ],
                "manual_halt_active": self._state.manual_halt_active,
                "last_halt_at": self._state.last_halt_at,
                "last_updated": self._state.last_updated,
                "manual_halt_history": [
                    asdict(ev) for ev in self._state.manual_halt_history
                ],
            }
            self._state_path.write_text(json.dumps(payload, indent=2))
        except Exception as exc:
            _record_save_failure(day, exc, path=str(self._state_path))
            return
        _record_save_success(day)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def snapshot(self) -> PersistentPeakState:
        """Return the current persisted state (read-only)."""
        return self._state

    def update(self, balance: float) -> PersistentPeakState:
        """Ratchet peaks against ``balance`` and persist the result.

        - ``all_time_peak`` is monotonic — only ever increases.
        - ``weekly_balances`` records today's max and drops entries older
          than ``weekly_window_days`` calendar UTC days.

        Returns the updated immutable snapshot.
        """
        if balance <= 0:
            return self._state

        now = self._clock()
        new_all_time = max(self._state.all_time_peak, balance)
        new_weekly = self._refresh_weekly(balance, now)

        self._state = replace(
            self._state,
            all_time_peak=new_all_time,
            weekly_balances=new_weekly,
            last_updated=now.isoformat(),
        )
        self._save_state()
        return self._state

    def check_budget(self, balance: float, daily_pnl: float) -> RiskBudget:
        """Combined drawdown + daily-loss check.

        The returned :class:`RiskBudget` is the inner guard's, possibly
        replaced if our dual-rail multi-day check halts more aggressively.
        """
        # R10 P1.1 V3 (foundation cross-review C1): sweep stale sentinel.
        # If the sentinel file exists while no halt is active, an operator
        # has either pre-armed it by mistake or left it from a prior
        # programmatic reset. Leaving it in place is dangerous: the next
        # cliff trip would be silently cleared on the cycle right after
        # the trip, and ops would never see the halt fire. We unlink the
        # stale sentinel proactively and emit a WARNING so misconfig is
        # visible. The unlink itself is best-effort — if we can't delete
        # the file we still log; the guard rejects subsequent halt-trip
        # cycles correctly because the sentinel guard is gated on the
        # halt-active flag (we rely on the operator to investigate after
        # seeing the warning).
        if (
            not self._state.manual_halt_active
            and self._state.last_halt_at is None
            and self._sentinel_path.exists()
        ):
            try:
                self._sentinel_path.unlink()
                _logger.warning(
                    "stale_sentinel swept (no active halt) at %s",
                    self._sentinel_path,
                )
            except FileNotFoundError:
                # Race: another process already removed it.  Nothing to do.
                pass
            except Exception:
                # Filesystem error — log loud so ops investigates.
                _logger.warning(
                    "stale_sentinel unlink failed at %s",
                    self._sentinel_path,
                    exc_info=True,
                )

        # 1. Consume sentinel file if operator dropped one (clears manual halt).
        # Operator intent is "resume trading now" — clear both the sticky
        # cliff flag AND the 24h alarm timer in the same atomic step.
        if self._state.manual_halt_active and self._sentinel_path.exists():
            note = _read_sentinel_note(self._sentinel_path)
            try:
                self._sentinel_path.unlink()
            except FileNotFoundError:
                pass
            now = self._clock()
            history = _close_open_event(
                history=self._state.manual_halt_history,
                cleared_at=now.isoformat(),
                operator_note=note,
            )
            self._state = replace(
                self._state,
                manual_halt_active=False,
                last_halt_at=None,
                manual_halt_history=history,
            )
            self._save_state()

        now = self._clock()
        peak = self._state.all_time_peak
        # 2. Manual halt is sticky — only sentinel above clears it.
        if self._state.manual_halt_active:
            return _halted(
                balance=balance,
                peak=peak,
                reason=(
                    "Manual reset required: drawdown breached cliff ratio "
                    f"{self._cliff_ratio:.2f}; drop sentinel "
                    f"{self._sentinel_path} to resume"
                ),
            )

        # 3. 24h auto-resume halt still active?
        if self._state.last_halt_at is not None:
            try:
                halted_at = datetime.fromisoformat(self._state.last_halt_at)
            except ValueError:
                halted_at = None
            if halted_at is not None and now - halted_at < self._auto_resume:
                resume_at = halted_at + self._auto_resume
                return _halted(
                    balance=balance,
                    peak=peak,
                    reason=(
                        f"Multi-day drawdown halt active until "
                        f"{resume_at.isoformat()} (alarm ratio "
                        f"{self._alarm_ratio:.2f})"
                    ),
                )
            # Auto-resume window passed — clear the timer and stamp audit.
            history = _close_open_event(
                history=self._state.manual_halt_history,
                cleared_at=now.isoformat(),
                operator_note=None,
            )
            self._state = replace(
                self._state,
                last_halt_at=None,
                manual_halt_history=history,
            )
            self._save_state()

        # 4. Run the dual-rail ratio check.
        all_time_ratio = _ratio(balance, self._state.all_time_peak)
        weekly_ratio = _ratio(balance, self._state.weekly_peak)
        if all_time_ratio <= weekly_ratio:
            worst_ratio, worst_rail, worst_peak = (
                all_time_ratio,
                "all-time",
                self._state.all_time_peak,
            )
        else:
            worst_ratio, worst_rail, worst_peak = (
                weekly_ratio,
                "weekly",
                self._state.weekly_peak,
            )

        if worst_ratio <= self._cliff_ratio:
            event = HaltEvent(
                tripped_at=now.isoformat(),
                tier="cliff",
                rail=worst_rail,
                equity_at_trip=round(balance, 2),
                peak_at_trip=round(worst_peak, 2),
                ratio_at_trip=round(worst_ratio, 4),
            )
            self._state = replace(
                self._state,
                manual_halt_active=True,
                last_halt_at=now.isoformat(),
                manual_halt_history=list(self._state.manual_halt_history) + [event],
            )
            self._save_state()
            return _halted(
                balance=balance,
                peak=peak,
                reason=(
                    f"Multi-day drawdown CLIFF: {worst_rail} ratio "
                    f"{worst_ratio:.3f} <= {self._cliff_ratio:.2f} "
                    "— manual reset required"
                ),
            )

        if worst_ratio <= self._alarm_ratio:
            event = HaltEvent(
                tripped_at=now.isoformat(),
                tier="alarm",
                rail=worst_rail,
                equity_at_trip=round(balance, 2),
                peak_at_trip=round(worst_peak, 2),
                ratio_at_trip=round(worst_ratio, 4),
            )
            self._state = replace(
                self._state,
                last_halt_at=now.isoformat(),
                manual_halt_history=list(self._state.manual_halt_history) + [event],
            )
            self._save_state()
            return _halted(
                balance=balance,
                peak=peak,
                reason=(
                    f"Multi-day drawdown ALARM: {worst_rail} ratio "
                    f"{worst_ratio:.3f} <= {self._alarm_ratio:.2f} "
                    f"— halted for {self._auto_resume.total_seconds() / 3600:.0f}h"
                ),
            )

        # 5. Multi-day clear — fall through to inner daily-loss guard.
        return self._inner.check_budget(
            balance=balance,
            peak_balance=self._state.all_time_peak,
            daily_pnl=daily_pnl,
        )

    def reset(self) -> PersistentPeakState:
        """Operator override — clears halt timers but preserves peaks."""
        now = self._clock()
        history = _close_open_event(
            history=self._state.manual_halt_history,
            cleared_at=now.isoformat(),
            operator_note="programmatic reset()",
        )
        self._state = replace(
            self._state,
            manual_halt_active=False,
            last_halt_at=None,
            manual_halt_history=history,
        )
        self._save_state()
        return self._state

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _refresh_weekly(self, balance: float, now: datetime) -> list:
        """Return a fresh weekly_balances list with today's max merged in.

        Drops entries older than ``(today - weekly_window_days + 1)``.
        With ``weekly_window_days=5`` the window includes today plus the
        prior 4 UTC dates — five distinct calendar days total.
        """
        today = now.date()
        today_iso = today.isoformat()
        cutoff = today - timedelta(days=self._weekly_window_days - 1)
        merged = []
        seen_today = False
        for date_iso, max_bal in self._state.weekly_balances:
            try:
                entry_date = datetime.fromisoformat(date_iso).date()
            except ValueError:
                continue
            if entry_date < cutoff:
                continue
            if date_iso == today_iso:
                merged.append((today_iso, max(float(max_bal), balance)))
                seen_today = True
            else:
                merged.append((date_iso, float(max_bal)))
        if not seen_today:
            merged.append((today_iso, balance))
        merged.sort(key=lambda e: e[0])
        return merged


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _ratio(balance: float, peak: float) -> float:
    if peak <= 0:
        return 1.0
    return balance / peak


def _read_sentinel_note(path: Path) -> Optional[str]:
    """Read the operator's note from the sentinel file (best-effort)."""
    try:
        text = path.read_text().strip()
    except Exception:
        return None
    return text or None


def _close_open_event(
    *,
    history: list,
    cleared_at: str,
    operator_note: Optional[str],
) -> list:
    """Stamp ``cleared_at`` and ``operator_note`` on the most recent open event.

    Returns a new list — :class:`HaltEvent` is frozen, so updates are made
    via :func:`dataclasses.replace`. If the latest event is already closed
    (or the history is empty), returns the input unchanged.
    """
    if not history:
        return history
    last = history[-1]
    if last.cleared_at is not None:
        return history
    closed = replace(
        last,
        cleared_at=cleared_at,
        operator_note=operator_note,
    )
    return list(history[:-1]) + [closed]


def _halted(*, balance: float, peak: float, reason: str) -> RiskBudget:
    effective_peak = max(peak, balance)
    drawdown_pct = (
        0.0
        if effective_peak <= 0
        else (effective_peak - balance) / effective_peak * 100.0
    )
    # Round 10 P1.1: every multi-day halt only blocks new opens. Existing
    # positions stay broker-side and exit via SL/TP/trail. force_close is
    # reserved for a future ratio<=0.80 emergency tier.
    return RiskBudget(
        can_trade=False,
        available_risk_pct=0.0,
        used_risk_pct=100.0,
        daily_loss_pct=0.0,
        total_drawdown_pct=round(drawdown_pct, 4),
        rejection_reason=reason,
        block_opens=True,
        force_close=False,
    )

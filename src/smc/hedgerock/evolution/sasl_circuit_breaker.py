"""Stage 6 / r10-phase1 — SASL frequency-based circuit breaker.

Freezes the SASL auto-adjustment loop when too many adjustments have
fired inside a rolling time window (default: 5+ in 7 days).  Manual
operator reset (``human_reset``) is required to unfreeze.

Design notes
------------

* In-memory state is the source of truth at runtime; ``state_path``
  (optional) persists every event as append-only JSONL so the breaker
  survives process restarts.
* JSONL is *append-only*: the file is never truncated or rewritten.
  ``human_reset`` writes a sentinel line with ``"kind": "reset"`` so
  reload reconstructs the post-reset state via a "_reset_at" cursor.
* No live trading runtime imports.  Stdlib only.

Public API: :class:`AdjustmentRecord`, :class:`FreezeReport`,
:class:`SASLCircuitBreaker`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


__all__ = [
    "AdjustmentRecord",
    "FreezeReport",
    "SASLCircuitBreaker",
]


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AdjustmentRecord:
    """One auto-adjustment event."""

    timestamp: str  # ISO-8601 UTC
    adjustment_id: str


@dataclass(frozen=True)
class FreezeReport:
    """Snapshot of the breaker's freeze state."""

    is_frozen: bool
    n_adjustments_in_window: int
    max_adjustments: int
    window_days: int
    earliest_recorded_at: str | None
    latest_recorded_at: str | None
    freeze_reason: str | None
    requires_human_review: bool
    generated_at: str


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _coerce_to_utc(ts: datetime | str) -> datetime:
    """Coerce a datetime or ISO-8601 string to a tz-aware UTC datetime."""
    if isinstance(ts, str):
        # ``datetime.fromisoformat`` handles "+00:00" and naive forms.
        # Trailing "Z" is not parsed by ``fromisoformat`` until 3.11.
        clean = ts.rstrip()
        if clean.endswith("Z"):
            clean = clean[:-1] + "+00:00"
        parsed = datetime.fromisoformat(clean)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _to_iso(dt: datetime) -> str:
    return _coerce_to_utc(dt).isoformat()


# --------------------------------------------------------------------------- #
# Circuit breaker
# --------------------------------------------------------------------------- #


class SASLCircuitBreaker:
    """Rolling-window adjustment counter with manual-reset semantics."""

    def __init__(
        self,
        *,
        max_adjustments: int = 4,
        window_days: int = 7,
        state_path: Path | None = None,
        now: datetime | None = None,
    ) -> None:
        if max_adjustments < 0:
            raise ValueError("max_adjustments must be non-negative")
        if window_days <= 0:
            raise ValueError("window_days must be positive")

        self._max_adjustments = int(max_adjustments)
        self._window_days = int(window_days)
        self._state_path: Path | None = (
            Path(state_path) if state_path is not None else None
        )
        self._now_override: datetime | None = (
            _coerce_to_utc(now) if now is not None else None
        )

        self._records: list[AdjustmentRecord] = []
        self._reset_history: list[dict[str, Any]] = []
        # Filter cursor — adjustments at or before this time are ignored
        # by ``adjustments_in_window`` even though they remain on disk.
        self._reset_cursor: datetime | None = None

        if self._state_path is not None:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            if self._state_path.exists():
                self._load_from_disk()

    # ------------------------------------------------------------------ #
    # Clock
    # ------------------------------------------------------------------ #

    def _now(self) -> datetime:
        if self._now_override is not None:
            return self._now_override
        return datetime.now(timezone.utc)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def _load_from_disk(self) -> None:
        assert self._state_path is not None
        with self._state_path.open("r", encoding="utf-8") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    # Corrupt line — skip rather than crash, but do not
                    # rewrite the file.
                    continue
                kind = payload.get("kind")
                if kind == "adjustment":
                    ts = payload.get("timestamp")
                    aid = payload.get("adjustment_id")
                    if isinstance(ts, str) and isinstance(aid, str):
                        self._records.append(
                            AdjustmentRecord(timestamp=ts, adjustment_id=aid)
                        )
                elif kind == "reset":
                    ts = payload.get("timestamp")
                    operator_id = payload.get("operator_id", "")
                    n_cleared = int(payload.get("n_records_cleared", 0))
                    if isinstance(ts, str):
                        self._reset_cursor = _coerce_to_utc(ts)
                        self._reset_history.append(
                            {
                                "timestamp": ts,
                                "operator_id": operator_id,
                                "n_records_cleared": n_cleared,
                            }
                        )

    def _append_jsonl(self, payload: dict[str, Any]) -> None:
        if self._state_path is None:
            return
        line = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        with self._state_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    # ------------------------------------------------------------------ #
    # Recording
    # ------------------------------------------------------------------ #

    def record_adjustment(
        self,
        timestamp: datetime | str,
        adjustment_id: str,
    ) -> None:
        if not adjustment_id:
            raise ValueError("adjustment_id must be a non-empty string")
        ts_iso = _to_iso(_coerce_to_utc(timestamp))
        record = AdjustmentRecord(timestamp=ts_iso, adjustment_id=adjustment_id)
        self._records = [*self._records, record]
        self._append_jsonl(
            {
                "kind": "adjustment",
                "timestamp": ts_iso,
                "adjustment_id": adjustment_id,
            }
        )

    # ------------------------------------------------------------------ #
    # Window queries
    # ------------------------------------------------------------------ #

    def _resolve_now(self, now: datetime | None) -> datetime:
        if now is not None:
            return _coerce_to_utc(now)
        return self._now()

    def adjustments_in_window(
        self, *, now: datetime | None = None,
    ) -> tuple[AdjustmentRecord, ...]:
        end = self._resolve_now(now)
        start = end - timedelta(days=self._window_days)
        in_window: list[AdjustmentRecord] = []
        for record in self._records:
            ts = _coerce_to_utc(record.timestamp)
            if self._reset_cursor is not None and ts <= self._reset_cursor:
                continue
            if start <= ts <= end:
                in_window.append(record)
        return tuple(in_window)

    def is_frozen(self, *, now: datetime | None = None) -> bool:
        return (
            len(self.adjustments_in_window(now=now))
            > self._max_adjustments
        )

    def freeze_reason(self, *, now: datetime | None = None) -> str | None:
        in_window = self.adjustments_in_window(now=now)
        n = len(in_window)
        if n <= self._max_adjustments:
            return None
        return (
            f"circuit_breaker_frozen: {n} adjustments in last "
            f"{self._window_days} days (max={self._max_adjustments})"
        )

    # ------------------------------------------------------------------ #
    # Reporting + reset
    # ------------------------------------------------------------------ #

    def request_human_review(
        self, *, now: datetime | None = None,
    ) -> FreezeReport:
        now_dt = self._resolve_now(now)
        in_window = self.adjustments_in_window(now=now_dt)
        frozen = len(in_window) > self._max_adjustments
        earliest = in_window[0].timestamp if in_window else None
        latest = in_window[-1].timestamp if in_window else None
        reason = self.freeze_reason(now=now_dt)
        return FreezeReport(
            is_frozen=frozen,
            n_adjustments_in_window=len(in_window),
            max_adjustments=self._max_adjustments,
            window_days=self._window_days,
            earliest_recorded_at=earliest,
            latest_recorded_at=latest,
            freeze_reason=reason,
            requires_human_review=frozen,
            generated_at=_to_iso(now_dt),
        )

    def human_reset(self, operator_id: str) -> bool:
        if not operator_id:
            raise ValueError("operator_id must be a non-empty string")
        now_dt = self._now()
        in_window_before = self.adjustments_in_window(now=now_dt)
        n_cleared = len(in_window_before)

        ts_iso = _to_iso(now_dt)
        self._reset_cursor = now_dt
        self._reset_history = [
            *self._reset_history,
            {
                "timestamp": ts_iso,
                "operator_id": operator_id,
                "n_records_cleared": n_cleared,
            },
        ]
        self._append_jsonl(
            {
                "kind": "reset",
                "timestamp": ts_iso,
                "operator_id": operator_id,
                "n_records_cleared": n_cleared,
            }
        )
        return True

    @property
    def reset_history(self) -> tuple[dict, ...]:
        # Return frozen tuple of shallow-copied dicts so callers cannot
        # mutate the breaker's internal state.
        return tuple(dict(entry) for entry in self._reset_history)

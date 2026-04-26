"""Persist aggregator gate diagnostics to a per-day jsonl file.

R10 P1.2 — observability for the SMC gate funnel.
R10 P4 calibration commit 0 — schema_version=2 adds optional per-trade ``flags``
snapshot so the calibration runner can split arms by R10 feature-flag state
without reconstructing flags from process boot-time alone (catches operator
mid-run env mutation).

The aggregator (``MultiTimeframeAggregator.generate_setups``) builds a rich
``_last_setup_diagnostic`` dict on every call.  Without persistence the
diagnostic is overwritten each cycle, so we cannot answer "100% RangeTrader,
0% SMC trend — where do setups die?".  This module appends one jsonl line per
``generate_setups`` call to ``data/diagnostics/aggregator_gates_<UTC-DATE>.jsonl``
so the funnel can be reconstructed offline by ``scripts/build_gate_funnel.py``.

Design notes
------------
- The aggregator stays decoupled — it only exposes ``_last_setup_diagnostic``
  via attribute access.  Callers (live_demo, backtest scripts) decide whether
  to persist and pass the ops-layer context (magic, symbol, journal_dir).
- ``enabled=False`` lets backtest harnesses share the call site without
  polluting the production ``data/diagnostics/`` directory.  Live demo passes
  the default ``True``.
- The filename's UTC date comes strictly from ``bar_ts`` (NOT ``datetime.now``)
  so backtest replays are reproducible — the same fixture always lands in the
  same file, allowing safe re-run.
- Persistence is fail-closed: any IO error is swallowed and logged via the
  stdlib ``logging`` module with ``exc_info=True`` so ops sees the traceback.
  If the same UTC day sees ``_ESCALATE_AFTER`` consecutive failures, the next
  log is escalated to ERROR ("disk full?") so the issue surfaces in alerting.
- Per-UTC-day rotation by filename ensures the funnel builder can glob files
  without parsing line timestamps to discover the day boundary.
- ``zone_details`` is excluded from the persisted payload — it can grow to
  dozens of entries per call (one per scanned zone) and is redundant with
  the ``zone_rejects`` aggregate counters.
- ``schema_version`` is written into every record.  v1 records (pre-calibration)
  have no ``flags`` key; v2 records may carry a ``flags`` dict snapshotted from
  SMCConfig.  Calibration consumers read ``flags=None`` for legacy v1 records
  and fall back to boot-time snapshot, no migration script needed.
- ``flags`` is captured at call-time, NOT process-start, so an operator mutating
  an env var mid-run is still classifiable correctly.  Boot-time snapshot is a
  separate layer in the calibration pipeline.
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import date as _date
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_JOURNAL_DIR = Path("data") / "diagnostics"
FILENAME_TEMPLATE = "aggregator_gates_{date}.jsonl"
_VERBOSE_FIELDS = ("zone_details",)

# Schema version for the persisted record.  v1 = R10 P1.2 baseline (no ``flags``).
# v2 = R10 P4 calibration commit 0 (optional ``flags`` dict snapshotted from
# SMCConfig).  Bump when you add or remove top-level record fields; readers
# keyed on this version can migrate gracefully.
SCHEMA_VERSION = 2

# R10 feature-flag fields snapshotted from SMCConfig into each record's
# ``flags`` dict when a config is supplied.  Order is preserved for stability
# but readers should index by name, not position.  These match the field names
# on ``SMCConfig`` (see src/smc/config.py field_validator at line 77).
_FLAG_FIELDS: tuple[str, ...] = (
    "macro_enabled",
    "ai_regime_enabled",
    "ai_mode_router_enabled",
    "ai_regime_trust_threshold",
    "range_trend_filter_enabled",
    "range_ai_regime_gate_enabled",
    "range_require_regime_valid",
    "range_reversal_confirm_enabled",
    "range_ai_direction_entry_gate_enabled",
    "mode_router_trending_dominance_enabled",
    "spread_gate_enabled",
    "max_concurrent_per_symbol",
    "anti_stack_cooldown_minutes",
)

# Number of consecutive failures on a UTC day before we escalate the next
# warning to ERROR.  Three is enough to flag persistent disk problems
# (full / readonly / permissions) without being noisy on transient blips.
_ESCALATE_AFTER = 3

_logger = logging.getLogger(__name__)
# Per-process per-UTC-day failure counter.  Guarded by a lock so concurrent
# legs (control + treatment) on the same host can't race the counter.
_failure_lock = threading.Lock()
_failure_counter: dict[_date, int] = {}


def _utc_date(ts: datetime) -> _date:
    """Return ``ts``'s UTC ``date``.

    Naive datetimes are assumed to be UTC (matches aggregator convention).
    """
    if ts.tzinfo is None:
        return ts.date()
    return ts.astimezone(timezone.utc).date()


def _utc_date_str(ts: datetime) -> str:
    """Return YYYY-MM-DD for ``ts`` in UTC."""
    return _utc_date(ts).strftime("%Y-%m-%d")


def _strip_verbose(diagnostic: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``diagnostic`` without bulky fields.

    Pure function — does not mutate the input.  Currently drops
    ``zone_details`` (one entry per scanned H1 zone), keeping the
    aggregate ``zone_rejects`` counters intact.
    """
    return {k: v for k, v in diagnostic.items() if k not in _VERBOSE_FIELDS}


def _serialize_flags(config: Any) -> dict[str, Any]:
    """Snapshot the R10 feature-flag fields from ``config`` into a flat dict.

    Pure function — reads attributes via ``getattr`` so it works against any
    object exposing the named fields (typically an ``SMCConfig`` instance).
    Missing fields are silently omitted rather than logged, since calibration
    consumers tolerate partial snapshots (matching ``flags=None`` for legacy
    v1 records).  Returns an empty dict if ``config`` is ``None``.
    """
    if config is None:
        return {}
    snapshot: dict[str, Any] = {}
    for field in _FLAG_FIELDS:
        if hasattr(config, field):
            snapshot[field] = getattr(config, field)
    return snapshot


def journal_path_for(
    bar_ts: datetime,
    journal_dir: Path | str = DEFAULT_JOURNAL_DIR,
) -> Path:
    """Compute the per-UTC-day jsonl path for ``bar_ts``."""
    return Path(journal_dir) / FILENAME_TEMPLATE.format(date=_utc_date_str(bar_ts))


def reset_failure_counter() -> None:
    """Clear the per-day failure counter — exposed for tests."""
    with _failure_lock:
        _failure_counter.clear()


def _record_failure(day: _date, event: str, **fields: Any) -> None:
    """Increment failure count for ``day`` and log warn / error accordingly."""
    with _failure_lock:
        _failure_counter[day] = _failure_counter.get(day, 0) + 1
        count = _failure_counter[day]
    if count >= _ESCALATE_AFTER:
        _logger.error(
            "%s — gate diagnostic write failed %dx today (disk full or readonly?)",
            event,
            count,
            extra={"event": event, "failure_count": count, "utc_date": str(day), **fields},
            exc_info=True,
        )
    else:
        _logger.warning(
            "%s — gate diagnostic write failed (%d/%d before escalation)",
            event,
            count,
            _ESCALATE_AFTER,
            extra={"event": event, "failure_count": count, "utc_date": str(day), **fields},
            exc_info=True,
        )


def _record_success(day: _date) -> None:
    """Reset the failure counter for ``day`` after a successful write."""
    with _failure_lock:
        if day in _failure_counter:
            del _failure_counter[day]


def append_gate_diagnostic(
    diagnostic: dict[str, Any],
    *,
    bar_ts: datetime,
    symbol: str,
    magic: int,
    journal_dir: Path | str = DEFAULT_JOURNAL_DIR,
    enabled: bool = True,
    config: Any = None,
) -> Path | None:
    """Append a single gate diagnostic record to the per-day jsonl.

    Parameters
    ----------
    diagnostic:
        The ``_last_setup_diagnostic`` dict from the aggregator.  Verbose
        fields (``zone_details``) are stripped before serialization.
    bar_ts:
        Timestamp used both for the row's ``bar_ts`` field and for picking
        the per-day file (UTC date of ``bar_ts``).  Naive values are
        treated as UTC.  Must be derived from the call site, NOT ``now()``,
        so backtest replay is reproducible.
    symbol:
        Trading symbol (e.g. ``"XAUUSD"``).
    magic:
        Effective magic number for this leg (control vs treatment).  Stored
        per-row so the funnel builder can split by magic without inspecting
        the filename.
    journal_dir:
        Directory holding ``aggregator_gates_<date>.jsonl`` files.  Created
        if missing.  Defaults to ``data/diagnostics``.
    enabled:
        When ``False`` the call is a no-op — backtest harnesses pass
        ``enabled=False`` to share the same call site without polluting the
        production diagnostics directory.  Default ``True``.
    config:
        Optional ``SMCConfig`` (or any object exposing the R10 flag fields).
        When provided, a snapshot of ``_FLAG_FIELDS`` is written into the
        record's ``flags`` key.  When ``None`` (default — preserves backward
        compat with R10 P1.2 callers) no ``flags`` key is written; calibration
        consumers tolerate the absence.

    Returns
    -------
    Path | None
        The file path that was written, ``None`` if persistence was
        disabled or failed.  Failures are logged via the stdlib ``logging``
        module with ``exc_info=True`` and a per-day failure counter; once a
        UTC day accumulates ``_ESCALATE_AFTER`` failures the next log is
        emitted at ERROR.  Never raises — telemetry must not break the loop.
    """
    if not enabled:
        return None

    day = _utc_date(bar_ts)
    target = journal_path_for(bar_ts, journal_dir)

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        _record_failure(
            day,
            "gate_diag_mkdir_failed",
            path=str(target.parent),
        )
        return None

    if bar_ts.tzinfo is None:
        bar_ts_iso = bar_ts.replace(tzinfo=timezone.utc).isoformat()
    else:
        bar_ts_iso = bar_ts.astimezone(timezone.utc).isoformat()

    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "bar_ts": bar_ts_iso,
        "symbol": symbol,
        "magic": int(magic),
        "diagnostic": _strip_verbose(diagnostic),
    }
    if config is not None:
        record["flags"] = _serialize_flags(config)
    try:
        with target.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
    except Exception:
        _record_failure(
            day,
            "gate_diag_write_failed",
            path=str(target),
        )
        return None

    _record_success(day)
    return target


__all__ = [
    "DEFAULT_JOURNAL_DIR",
    "FILENAME_TEMPLATE",
    "SCHEMA_VERSION",
    "append_gate_diagnostic",
    "journal_path_for",
    "reset_failure_counter",
]

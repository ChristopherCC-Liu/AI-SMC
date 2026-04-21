"""Per-cycle ``health_probe`` event for measurement-lead SLA digest (R2).

Each main-loop cycle emits exactly one ``health_probe`` INFO-severity
structured event so the daily digest can compute uptime, tick
availability, and AI latency percentiles.  The event is the single
audit-trail anchor for "was this process healthy at cycle N?".

Design
------
The helper is a pure function that takes already-computed values and
returns the event payload.  This keeps the caller (live_demo.py cycle
loop) simple — it owns the ``data`` dict, the watchdog state, the
macro layer handle — and keeps the helper unit-testable without
mocking MT5.

A thin ``emit()`` wrapper calls ``structured_log.info`` with the
assembled payload; that is the call site in live_demo.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

# The canonical set of timeframes a healthy cycle must have fetched.
# Anything missing → data_ok=False so the digest knows a cycle is
# "limping" even though it did not exit.
_REQUIRED_TIMEFRAMES: tuple[str, ...] = ("M15", "H1", "H4", "D1")


@dataclass(frozen=True)
class HealthProbeFields:
    """Immutable snapshot of the fields that make up one health_probe event.

    Kept as a frozen dataclass so tests can assert on a plain object
    rather than a dict, and callers can pass it as ``**kw`` to
    ``structured_log.info``.

    ``leg_suffix`` carries ``_path_cfg.journal_suffix`` (``""`` for
    XAU Control + BTC, ``"_macro"`` for XAU Treatment) so the
    measurement-lead daily digest can aggregate uptime per leg.
    ``equity_usd`` / ``floating_usd`` feed the ops-lead dashboard
    P&L card — they are sibling fields to ``balance_usd`` and all
    three come from a single ``mt5.account_info()`` probe.
    """

    cycle: int
    leg_suffix: str
    tick_ok: bool
    data_ok: bool
    handle_age_sec: int | None
    debate_elapsed_ms_last: int | None
    macro_bias_fresh: bool
    balance_usd: float | None
    equity_usd: float | None
    floating_usd: float | None

    def to_event_kwargs(self) -> dict[str, Any]:
        """Return a dict suitable for ``structured_log.info("health_probe", **kw)``."""
        return {
            "cycle": self.cycle,
            "leg_suffix": self.leg_suffix,
            "tick_ok": self.tick_ok,
            "data_ok": self.data_ok,
            "handle_age_sec": self.handle_age_sec,
            "debate_elapsed_ms_last": self.debate_elapsed_ms_last,
            "macro_bias_fresh": self.macro_bias_fresh,
            "balance_usd": self.balance_usd,
            "equity_usd": self.equity_usd,
            "floating_usd": self.floating_usd,
        }


def _compute_data_ok(data: Mapping[Any, Any] | None) -> bool:
    """True iff the timeframe-keyed ``data`` dict has non-empty frames for
    all of M15, H1, H4, D1.

    Accepts either ``Timeframe`` enum keys or str keys (so tests can
    feed plain dicts).  An entry is considered present when it is truthy
    and either: has ``len() > 0`` OR does not support ``len()`` at all
    (some adapters may pass opaque objects).
    """
    if not data:
        return False
    present: set[str] = set()
    for key, frame in data.items():
        if frame is None:
            continue
        key_str = getattr(key, "name", None) or getattr(key, "value", None) or str(key)
        try:
            length = len(frame)  # type: ignore[arg-type]
        except TypeError:
            length = 1  # opaque object, treat as present
        if length > 0:
            present.add(str(key_str))
    return all(tf in present for tf in _REQUIRED_TIMEFRAMES)


def build_probe(
    *,
    cycle: int,
    leg_suffix: str = "",
    tick_ok: bool,
    data: Mapping[Any, Any] | None,
    handle_age_sec: int | None,
    debate_elapsed_ms_last: int | None,
    macro_bias_fresh: bool,
    balance_usd: float | None,
    equity_usd: float | None = None,
    floating_usd: float | None = None,
) -> HealthProbeFields:
    """Return the canonical ``HealthProbeFields`` snapshot for this cycle.

    Callers pass already-resolved values.  The function is side-effect
    free so tests can assert on the exact shape.
    """
    return HealthProbeFields(
        cycle=int(cycle),
        leg_suffix=str(leg_suffix or ""),
        tick_ok=bool(tick_ok),
        data_ok=_compute_data_ok(data),
        handle_age_sec=handle_age_sec,
        debate_elapsed_ms_last=debate_elapsed_ms_last,
        macro_bias_fresh=bool(macro_bias_fresh),
        balance_usd=(
            float(balance_usd) if balance_usd is not None else None
        ),
        equity_usd=(
            float(equity_usd) if equity_usd is not None else None
        ),
        floating_usd=(
            float(floating_usd) if floating_usd is not None else None
        ),
    )


def emit(probe: HealthProbeFields) -> None:
    """Emit the probe via ``structured_log.info("health_probe", …)``.

    Split from :func:`build_probe` so unit tests can inspect the
    payload without monkey-patching stderr.  Never raises.
    """
    try:
        from smc.monitor.structured_log import info as log_info

        log_info("health_probe", **probe.to_event_kwargs())
    except Exception:
        pass


__all__ = [
    "HealthProbeFields",
    "build_probe",
    "emit",
]

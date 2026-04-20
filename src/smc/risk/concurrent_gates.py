"""Pre-write gates protecting against position stacking disasters.

Round 4 v5 — created after the 2026-04-20 02:46 UTC incident where 5
XAUUSD BUYs opened in a 30-minute window got wiped out simultaneously
at the same SL level (-$212.25 total). Two independent gates:

- ``check_concurrent_cap``: hard cap on open positions per (symbol, magic).
  Prevents stacking more than N positions regardless of timing.

- ``check_anti_stack_cooldown``: time-based cooldown between same-direction
  entries on (symbol, magic). Prevents rapid-fire entries even below cap.

Both functions are pure (no MT5 I/O) so callers pass pre-fetched MT5
objects or mocks. This makes them trivial to unit test and keeps the
live_demo main loop free of additional MT5 calls outside the existing
pre-write gate block.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Protocol


__all__ = [
    "GateResult",
    "PositionLike",
    "DealLike",
    "check_concurrent_cap",
    "check_anti_stack_cooldown",
]


@dataclass(frozen=True)
class GateResult:
    """Outcome of a pre-write gate evaluation.

    When ``can_trade`` is False, ``reason`` is a short tag suitable for
    logging and a structured log event; ``detail`` carries the numeric
    context (e.g., current count / cap).
    """

    can_trade: bool
    reason: str = ""
    detail: str = ""


class PositionLike(Protocol):
    """Minimal shape of an MT5 position record we need."""

    magic: int
    symbol: str


class DealLike(Protocol):
    """Minimal shape of an MT5 deal record we need.

    ``entry`` follows MT5 semantics: 0 = IN (opening deal), 1 = OUT,
    2 = INOUT, 3 = OUT_BY.
    ``type`` follows MT5 semantics: 0 = BUY, 1 = SELL.
    ``time`` is Unix seconds.
    """

    symbol: str
    magic: int
    entry: int
    type: int
    time: int


def check_concurrent_cap(
    positions: Iterable[PositionLike],
    *,
    magic: int,
    max_concurrent: int,
) -> GateResult:
    """Block the entry when open positions matching ``magic`` hit the cap.

    Callers pass the full ``mt5.positions_get(symbol=...)`` iterable;
    this function filters by magic so Control and Treatment legs stay
    independent under Round 4 v5 Option B dual-magic routing.
    """
    if max_concurrent < 1:
        raise ValueError("max_concurrent must be >= 1")

    own_count = sum(1 for p in positions if p.magic == magic)
    if own_count >= max_concurrent:
        return GateResult(
            can_trade=False,
            reason="concurrent_cap",
            detail=f"{own_count}/{max_concurrent}",
        )
    return GateResult(can_trade=True)


def check_anti_stack_cooldown(
    deals: Iterable[DealLike],
    *,
    symbol: str,
    magic: int,
    direction: str,
    now: datetime,
    cooldown_minutes: int,
) -> GateResult:
    """Block same-direction entry if one was opened within cooldown window.

    ``direction`` must be ``"long"`` or ``"short"``. Any other value
    short-circuits to can_trade=True (the caller didn't know the
    direction, so there's nothing to protect).
    """
    if cooldown_minutes <= 0:
        return GateResult(can_trade=True)
    if direction not in ("long", "short"):
        return GateResult(can_trade=True)
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")

    mt5_type = 0 if direction == "long" else 1
    last_entry_dt: datetime | None = None
    for d in deals:
        if d.symbol != symbol:
            continue
        if d.magic != magic:
            continue
        if d.type != mt5_type:
            continue
        if d.entry != 0:  # 0 = IN (opening deal)
            continue
        d_dt = datetime.fromtimestamp(d.time, tz=timezone.utc)
        if last_entry_dt is None or d_dt > last_entry_dt:
            last_entry_dt = d_dt

    if last_entry_dt is None:
        return GateResult(can_trade=True)

    elapsed_min = (now - last_entry_dt).total_seconds() / 60.0
    if elapsed_min < cooldown_minutes:
        return GateResult(
            can_trade=False,
            reason="anti_stack",
            detail=f"{direction}_{elapsed_min:.1f}min<{cooldown_minutes}",
        )
    return GateResult(can_trade=True)

"""Position reconciliation between local tracking and broker state.

Compares the positions the OrderManager thinks are open against what the
broker actually reports. Detects three classes of divergence:

1. **Closed by broker** — we think a position is open, but the broker
   closed it (e.g. SL/TP hit, margin call, manual intervention).
2. **Untracked** — the broker has a position we don't know about (e.g.
   opened via the MT5 GUI or another EA).
3. **PnL mismatch** — both sides agree the position exists, but the
   unrealised PnL differs beyond a tolerance (indicates stale pricing
   or lot-size discrepancy).
"""

from __future__ import annotations

import logging

from smc.execution.types import PositionState, ReconciliationResult

__all__ = ["reconcile"]

logger = logging.getLogger(__name__)

# Tolerance for PnL comparison — below this USD delta we consider it "in sync".
_PNL_TOLERANCE_USD: float = 0.50


def reconcile(
    local_positions: dict[int, PositionState],
    broker_positions: tuple[PositionState, ...],
    *,
    pnl_tolerance: float = _PNL_TOLERANCE_USD,
) -> ReconciliationResult:
    """Compare local position tracking against broker-reported positions.

    Args:
        local_positions: Positions the OrderManager believes are open,
            keyed by ticket number.
        broker_positions: Positions the broker reports as currently open.
        pnl_tolerance: Maximum acceptable PnL difference in USD before
            flagging a mismatch.

    Returns:
        A frozen :class:`ReconciliationResult` describing any divergences.
    """
    broker_by_ticket: dict[int, PositionState] = {p.ticket: p for p in broker_positions}

    local_tickets = set(local_positions.keys())
    broker_tickets = set(broker_by_ticket.keys())

    # Positions we think are open but broker doesn't have
    closed_by_broker = tuple(sorted(local_tickets - broker_tickets))

    # Positions broker has that we don't track
    untracked = tuple(sorted(broker_tickets - local_tickets))

    # PnL mismatches on positions both sides agree exist
    shared_tickets = local_tickets & broker_tickets
    pnl_mismatches: list[int] = []
    for ticket in sorted(shared_tickets):
        local_pnl = local_positions[ticket].pnl_usd
        broker_pnl = broker_by_ticket[ticket].pnl_usd
        if abs(local_pnl - broker_pnl) > pnl_tolerance:
            pnl_mismatches.append(ticket)

    in_sync = not closed_by_broker and not untracked and not pnl_mismatches

    # Build human-readable summary
    parts: list[str] = []
    if closed_by_broker:
        parts.append(f"closed_by_broker={list(closed_by_broker)}")
    if untracked:
        parts.append(f"untracked={list(untracked)}")
    if pnl_mismatches:
        parts.append(f"pnl_mismatches={pnl_mismatches}")
    summary = "IN_SYNC" if in_sync else f"DIVERGED: {', '.join(parts)}"

    if not in_sync:
        logger.warning("Reconciliation divergence: %s", summary)

    return ReconciliationResult(
        in_sync=in_sync,
        closed_by_broker=closed_by_broker,
        untracked=untracked,
        pnl_mismatches=tuple(pnl_mismatches),
        summary=summary,
    )

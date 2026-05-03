"""Ticket 3 Step 3 — frozen sidecar simulator state.

**Sidecar reimplementation.** The production simulator
(``phase_d_walk_forward._step``) is **NOT imported** at runtime.
This module reimplements the simplified grid+martingale+hedge sim
mechanics in sidecar code so:

  1. baseline and candidate replays each get their own SimState
     instance (no shared mutable list / dict)
  2. apply_overlay never reaches a production module
  3. mirror drift in production sim mechanics is bounded by
     parity tests against ``_run_dynamic`` (out of scope for v1
     — sidecar replay is the ground truth here for shadow purposes)

Mirrored behaviour (per Ticket 3 plan §R4):
  - hedge open when flat (in hedgerock mode): 1 buy + 1 sell at close
  - TP per position: high ≥ entry+tp_usd → close buy at entry+tp_usd;
    low ≤ entry-tp_usd → close sell at entry-tp_usd
  - grid martingale add: when last_buy/sell_entry exists, n positions
    < max_orders, AND price crossed last_entry ± grid_spacing
  - lots = min(last_lots × gear, max_next_lot); skip if <= 0.001
  - blowup at dd ≥ 0.80 (closeall + blowup = True)
  - near_stopout count at dd ≥ 0.30
  - halt mode: closeall on entry, no new opens
  - observe mode: no new opens; existing TP still fires

Explicitly not mirrored (per Ticket 3 plan §R4):
  - broker fills / slippage / spread spike modelling
  - EA-side soft-close timer / SMLO / ADX confirms
  - margin call cascade (only blowup at 0.80 DD)
  - commission / swap / overnight financing
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


__all__ = [
    "Position",
    "SimState",
    "PIP_PNL_PER_LOT",
    "POINT",
    "STATIC_GRID_BASE_USD",
    "STATIC_ATR_MULT",
    "MAX_DD_BLOWUP",
    "NEAR_STOPOUT_DD",
    "fresh_sim_state",
    "open_position",
    "close_all",
    "step",
    "synthesize_ea_state",
]


# Mirror constants from production phase_d_walk_forward (verified by
# read-only inspection at design time; not imported).
POINT: float = 0.1
PIP_PNL_PER_LOT: float = 100.0
STATIC_GRID_BASE_USD: float = 25.0
STATIC_ATR_MULT: float = 4.0
MAX_DD_BLOWUP: float = 0.80
NEAR_STOPOUT_DD: float = 0.30


# ---------------------------------------------------------------------------
# Position + SimState
# ---------------------------------------------------------------------------


@dataclass
class Position:
    direction: int          # +1 buy, -1 sell
    entry_price: float
    lots: float
    open_ts: datetime


@dataclass
class SimState:
    """Per-replay mutable state. Two replays MUST instantiate this
    independently (use :func:`fresh_sim_state` per replay)."""

    equity: float
    cash: float
    high_watermark: float
    init_equity: float
    spread_pts: int
    positions: list[Position] = field(default_factory=list)
    last_buy_entry: float | None = None
    last_sell_entry: float | None = None
    closed_pnl_total: float = 0.0
    blowup: bool = False
    margin_stopout_count: int = 0
    near_stopout_count: int = 0
    halt_event_count: int = 0
    halt_streak_active: bool = False
    n_trades: int = 0           # closed positions
    n_opens: int = 0            # new entries
    max_dd_pct: float = 0.0
    max_open_lots: float = 0.0
    max_grid_density: int = 0   # max positions on one side at once
    recent_closed_pnls: list[float] = field(default_factory=list)


def fresh_sim_state(*, init_equity: float = 10_000.0,
                    spread_pts: int = 20) -> SimState:
    return SimState(
        equity=init_equity, cash=init_equity, high_watermark=init_equity,
        init_equity=init_equity, spread_pts=spread_pts,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_RECENT_PNL_SAMPLE: int = 20  # mirrors phase_d's _RECENT_PNL_SAMPLE


def _close_position(state: SimState, pos: Position, exit_price: float) -> None:
    pnl = (exit_price - pos.entry_price) * pos.direction * pos.lots * PIP_PNL_PER_LOT
    state.cash += pnl
    state.closed_pnl_total += pnl
    state.n_trades += 1
    state.recent_closed_pnls.append(pnl)
    if len(state.recent_closed_pnls) > _RECENT_PNL_SAMPLE:
        state.recent_closed_pnls.pop(0)


def open_position(*, state: SimState, ts: datetime, price: float,
                  direction: int, lots: float) -> None:
    pos = Position(direction=direction, entry_price=price, lots=lots, open_ts=ts)
    state.positions.append(pos)
    state.n_opens += 1
    if direction == 1:
        state.last_buy_entry = price
    else:
        state.last_sell_entry = price
    open_lots_now = sum(p.lots for p in state.positions)
    state.max_open_lots = max(state.max_open_lots, open_lots_now)
    n_buys = sum(1 for p in state.positions if p.direction == 1)
    n_sells = sum(1 for p in state.positions if p.direction == -1)
    state.max_grid_density = max(state.max_grid_density, n_buys, n_sells)


def close_all(*, state: SimState, exit_price: float) -> None:
    for pos in state.positions:
        _close_position(state, pos, exit_price)
    state.positions = []
    state.last_buy_entry = None
    state.last_sell_entry = None


def synthesize_ea_state(state: SimState) -> dict[str, Any]:
    """Synthesise an EA-state dict from the simulator's bookkeeping.

    Mirrors phase_d_walk_forward._build_synthetic_ea_state exactly:
      - consec_losses: walk recent_closed_pnls newest-first; count
        consecutive negatives
      - recent_closed_pnl: sum of recent_closed_pnls when n>0;
        None when n==0 (NOT 0.0 — that would silently say "we have
        evidence and it's neutral")
      - dd_pct: (high_watermark - equity) / high_watermark when hwm>0
    """
    consec = 0
    for pnl in reversed(state.recent_closed_pnls):
        if pnl < 0:
            consec += 1
        else:
            break

    n = len(state.recent_closed_pnls)
    if n == 0:
        recent_pnl: float | None = None
        recent_n: int | None = None
        consec_emit: int | None = None
    else:
        recent_pnl = float(sum(state.recent_closed_pnls))
        recent_n = n
        consec_emit = consec

    dd = 0.0
    if state.high_watermark > 0:
        dd = max(0.0, (state.high_watermark - state.equity) / state.high_watermark)

    open_lots = sum(p.lots for p in state.positions)
    floating = state.equity - state.cash

    return {
        "equity": state.equity,
        "balance": state.cash,
        "dd_pct": dd,
        "free_margin": state.equity,
        "margin_level": 999.0 if open_lots == 0 else 200.0,
        "open_lots": open_lots,
        "open_positions": len(state.positions),
        "floating_pnl": floating,
        "spread_pts": state.spread_pts,
        "consec_losses": consec_emit,
        "recent_closed_pnl": recent_pnl,
        "recent_sample_count": recent_n,
    }


# ---------------------------------------------------------------------------
# step() — process one H1 bar
# ---------------------------------------------------------------------------


def step(
    *,
    state: SimState,
    ts: datetime,
    high: float, low: float, close: float,
    grid_spacing: float,
    tp_usd: float,
    gear: float,
    max_next_lot: float,
    start_lots: float,
    max_orders_buy: int,
    max_orders_sell: int,
    mode: str,           # "hedgerock" | "observe" | "halt" | "momentum"
) -> None:
    """Process one H1 bar. Mutates state in place. Mirrors
    phase_d_walk_forward._step semantics; not imported."""
    if state.blowup:
        return

    # halt → closeall.
    if mode == "halt" and state.positions:
        close_all(state=state, exit_price=close)
        if not state.halt_streak_active:
            state.halt_event_count += 1
            state.halt_streak_active = True
    elif mode != "halt":
        state.halt_streak_active = False

    # TP per position.
    new_positions: list[Position] = []
    for pos in state.positions:
        if pos.direction == 1 and high >= pos.entry_price + tp_usd:
            _close_position(state, pos, pos.entry_price + tp_usd)
            continue
        if pos.direction == -1 and low <= pos.entry_price - tp_usd:
            _close_position(state, pos, pos.entry_price - tp_usd)
            continue
        new_positions.append(pos)
    state.positions = new_positions
    if not any(p.direction == 1 for p in state.positions):
        state.last_buy_entry = None
    if not any(p.direction == -1 for p in state.positions):
        state.last_sell_entry = None

    # Open initial hedge / grid add (only in hedgerock).
    if mode == "hedgerock" and len(state.positions) == 0:
        open_position(state=state, ts=ts, price=close, direction=+1, lots=start_lots)
        open_position(state=state, ts=ts, price=close, direction=-1, lots=start_lots)
    elif mode == "hedgerock":
        n_buys = sum(1 for p in state.positions if p.direction == 1)
        n_sells = sum(1 for p in state.positions if p.direction == -1)
        if state.last_buy_entry is not None and n_buys < max_orders_buy:
            if low <= state.last_buy_entry - grid_spacing:
                last_buy_lots = max(p.lots for p in state.positions if p.direction == 1)
                new_lots = min(last_buy_lots * gear, max_next_lot)
                if new_lots > 0.001:
                    open_position(
                        state=state, ts=ts,
                        price=state.last_buy_entry - grid_spacing,
                        direction=+1, lots=new_lots,
                    )
        if state.last_sell_entry is not None and n_sells < max_orders_sell:
            if high >= state.last_sell_entry + grid_spacing:
                last_sell_lots = max(p.lots for p in state.positions if p.direction == -1)
                new_lots = min(last_sell_lots * gear, max_next_lot)
                if new_lots > 0.001:
                    open_position(
                        state=state, ts=ts,
                        price=state.last_sell_entry + grid_spacing,
                        direction=-1, lots=new_lots,
                    )

    # Mark-to-market + DD + blowup.
    floating = sum(
        (close - p.entry_price) * p.direction * p.lots * PIP_PNL_PER_LOT
        for p in state.positions
    )
    state.equity = state.cash + floating
    state.high_watermark = max(state.high_watermark, state.equity)
    if state.high_watermark > 0:
        dd = (state.high_watermark - state.equity) / state.high_watermark
        state.max_dd_pct = max(state.max_dd_pct, dd)
        if dd >= NEAR_STOPOUT_DD:
            state.near_stopout_count += 1
        if dd >= MAX_DD_BLOWUP and not state.blowup:
            state.margin_stopout_count += 1
            close_all(state=state, exit_price=close)
            state.blowup = True

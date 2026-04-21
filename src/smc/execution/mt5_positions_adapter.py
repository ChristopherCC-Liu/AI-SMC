"""MT5 → PositionState adapter — Round 5 T1 F2.

Thin wrapper over the MetaTrader5 Python API that projects live broker state
into the domain types defined in :mod:`smc.execution.types`. Keeps mt5-specific
field naming out of the rest of the system so ``reconciler.reconcile`` and the
dashboard can consume typed, immutable values.

Functions:

- :func:`fetch_broker_positions` — currently open positions for our magic number.
- :func:`fetch_closed_pnl_since` — realised PnL on deals closed in a window.

Both return empty containers rather than raising when MT5 is unreachable or
the account has no matching records — callers treat "no data" as a benign
state, not a crash.

Round 5 T1 context: live_demo was bypassing OrderManager and calling
``mt5.order_send`` directly, which meant :mod:`smc.execution.reconciler` and
:class:`smc.risk.DrawdownGuard` never saw a single position. This adapter is
the missing bridge so cycle-end reconciliation + daily-loss accounting can run
against the real broker state.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from smc.execution.types import PositionState

__all__ = [
    "fetch_broker_positions",
    "fetch_closed_pnl_since",
    "XAUUSD_MAGIC",
]

# Must match the magic used in scripts/live_demo.py order_send request.
XAUUSD_MAGIC = 19760418


def fetch_broker_positions(
    mt5: Any,
    *,
    symbol: str = "XAUUSD",
    magic: int = XAUUSD_MAGIC,
) -> tuple[PositionState, ...]:
    """Return all positions opened by AI-SMC for *symbol*, as PositionState tuple.

    Filters by ``magic`` to exclude manual trades / other EAs on the same account.
    Returns an empty tuple when mt5 is unavailable, the call fails, or nothing
    matches — never raises.
    """
    try:
        positions = mt5.positions_get(symbol=symbol)
    except Exception:
        return ()
    if not positions:
        return ()

    out: list[PositionState] = []
    for p in positions:
        if getattr(p, "magic", None) != magic:
            continue
        try:
            out.append(_project_position(p, mt5))
        except Exception:
            continue
    return tuple(out)


def fetch_closed_pnl_since(
    mt5: Any,
    from_time: datetime,
    *,
    to_time: datetime | None = None,
    magic: int = XAUUSD_MAGIC,
) -> list[dict[str, Any]]:
    """Return realised PnL for deals that closed between *from_time* and *to_time*.

    A position close on MT5 is represented by a deal with ``entry=DEAL_ENTRY_OUT``
    (or ``DEAL_ENTRY_INOUT`` for partial closes). We sum ``profit + commission +
    swap`` to get the cash P&L including costs.

    Returns a list of ``{"ticket": int, "pnl_usd": float, "close_time":
    datetime}`` dicts, empty on any failure.
    """
    if to_time is None:
        to_time = datetime.now(tz=timezone.utc)
    try:
        deals = mt5.history_deals_get(from_time, to_time)
    except Exception:
        return []
    if not deals:
        return []

    # DEAL_ENTRY_OUT = 1, DEAL_ENTRY_INOUT = 2 per MT5 docs. Fall back to the
    # literals so this still works when mt5 is mocked in unit tests.
    exit_entries = {
        getattr(mt5, "DEAL_ENTRY_OUT", 1),
        getattr(mt5, "DEAL_ENTRY_INOUT", 2),
    }

    out: list[dict[str, Any]] = []
    for d in deals:
        if getattr(d, "magic", None) != magic:
            continue
        if getattr(d, "entry", None) not in exit_entries:
            continue
        pnl = (
            float(getattr(d, "profit", 0.0))
            + float(getattr(d, "commission", 0.0))
            + float(getattr(d, "swap", 0.0))
        )
        ts_raw = getattr(d, "time", None)
        try:
            close_time = (
                datetime.fromtimestamp(int(ts_raw), tz=timezone.utc)
                if ts_raw is not None
                else to_time
            )
        except (TypeError, ValueError, OverflowError):
            close_time = to_time
        # Round 5 O3: surface exit_price + magic + volume so the
        # trade_closed Telegram enrichment can compute rr_realized and map
        # the ticket to its A/B leg. All fields are optional — callers
        # that only need (ticket, pnl_usd, close_time) ignore the extras.
        exit_price = _safe_float(getattr(d, "price", None))
        volume = _safe_float(getattr(d, "volume", None))
        out.append(
            {
                "ticket": int(getattr(d, "position_id", 0) or getattr(d, "ticket", 0)),
                "pnl_usd": round(pnl, 2),
                "close_time": close_time,
                "exit_price": exit_price,
                "magic": int(getattr(d, "magic", magic)),
                "volume": volume,
            }
        )
    return out


def _safe_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _project_position(p: Any, mt5: Any) -> PositionState:
    """Map an mt5.positions_get() row to our :class:`PositionState` DTO."""
    # mt5 encodes direction via type: POSITION_TYPE_BUY=0 / SELL=1. Fall back
    # to literals for mocks that don't expose the constants.
    pos_type_buy = getattr(mt5, "POSITION_TYPE_BUY", 0)
    direction = "long" if getattr(p, "type", pos_type_buy) == pos_type_buy else "short"

    # Prefer the live tick for current price — p.price_current can lag.
    current_price = float(getattr(p, "price_current", 0.0) or 0.0)
    try:
        tick = mt5.symbol_info_tick(getattr(p, "symbol", "XAUUSD"))
        if tick is not None:
            current_price = float(tick.bid if direction == "long" else tick.ask)
    except Exception:
        pass  # keep whatever MT5 reported on the position row

    open_time_raw = getattr(p, "time", None)
    try:
        open_time = (
            datetime.fromtimestamp(int(open_time_raw), tz=timezone.utc)
            if open_time_raw is not None
            else datetime.now(tz=timezone.utc)
        )
    except (TypeError, ValueError, OverflowError):
        open_time = datetime.now(tz=timezone.utc)

    return PositionState(
        ticket=int(getattr(p, "ticket", 0)),
        instrument=str(getattr(p, "symbol", "XAUUSD")),
        direction=direction,
        lots=float(getattr(p, "volume", 0.0)),
        open_price=float(getattr(p, "price_open", 0.0)),
        current_price=current_price,
        sl=float(getattr(p, "sl", 0.0)),
        tp=float(getattr(p, "tp", 0.0)),
        pnl_usd=round(float(getattr(p, "profit", 0.0)), 2),
        open_time=open_time,
    )

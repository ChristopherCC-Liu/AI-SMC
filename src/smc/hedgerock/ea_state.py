"""EA → Decision Center runtime state ingest.

The MQL5 EA appends `&key=value&...` query params to `/signal?symbol=...`
on every OnTimer poll. These params describe the EA's *current* runtime
state (equity, drawdown, open exposure, recent realized PnL, ...) and
are consumed by the dynamic rule engine to derive `mode`, `lot_factor`,
`cooldown_until`, etc.

Phase A-closeout responsibilities of this module:
    1. Define a frozen :class:`EAState` dataclass.
    2. Provide a parser (`build_ea_state`) that maps the EA's `-1`
       sentinel for ``consec_losses`` / ``recent_closed_pnl`` /
       ``recent_sample_count`` to ``None``. Phase C rule engine MUST
       check for ``None`` before reading those fields.
    3. Provide a thread-safe :class:`EAStateStore` for the latest state
       per symbol. The /signal endpoint writes; /status reads.
    4. Recognize "no EA state sent" (legacy EA, missing params) and
       return ``None`` from the parser so the endpoint can decide not
       to clobber the previous good state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from threading import Lock

__all__ = [
    "EA_STATE_UNAVAILABLE",
    "EAState",
    "EAStateRecord",
    "EAStateStore",
    "build_ea_state",
]


# Sentinel value the EA emits when HistorySelect failed.
#
# Phase A-closeout mini-patch: ``recent_sample_count == -1`` is the
# CANONICAL history-unavailable signal. ``recent_closed_pnl == -1.0``
# alone is a perfectly legal real loss (e.g. a 1-USD loss on a small
# loser) and MUST NOT be treated as unavailable.
#
# When ``recent_sample_count == -1`` we fan the unavailable mark out
# to the other two history-derived fields:
#   - consec_losses
#   - recent_closed_pnl
#   - recent_sample_count itself
# All three become ``None`` so the rule engine cannot accidentally
# treat them as zero.
#
# Equity / balance / spread etc. always come from direct
# AccountInfoDouble calls and never carry the sentinel.
EA_STATE_UNAVAILABLE: int = -1


@dataclass(frozen=True)
class EAState:
    """Snapshot of the EA's runtime state at the moment of /signal poll.

    All fields with the ``| None`` annotation can be ``None`` either
    because the EA didn't send them (older EA / partial query) or
    because the EA sent the unavailable sentinel ``-1``.

    Phase C rule engine must defensively check for ``None`` before
    reading ``consec_losses`` / ``recent_closed_pnl`` /
    ``recent_sample_count``. Treating ``None`` as 0 would silently
    encode "no losses" / "no recent activity" which is the opposite of
    "we have no data".
    """

    equity: float | None
    balance: float | None
    dd_pct: float | None
    free_margin: float | None
    margin_level: float | None
    open_lots: float | None
    open_positions: int | None
    floating_pnl: float | None
    spread_pts: int | None
    # History-derived: -1 from EA → None here.
    consec_losses: int | None
    recent_closed_pnl: float | None
    recent_sample_count: int | None

    def to_dict(self) -> dict:
        return asdict(self)


def build_ea_state(
    *,
    equity: float | None = None,
    balance: float | None = None,
    dd_pct: float | None = None,
    free_margin: float | None = None,
    margin_level: float | None = None,
    open_lots: float | None = None,
    open_positions: int | None = None,
    floating_pnl: float | None = None,
    spread_pts: int | None = None,
    consec_losses: int | None = None,
    recent_closed_pnl: float | None = None,
    recent_sample_count: int | None = None,
) -> EAState | None:
    """Parse raw query-param values into an :class:`EAState`.

    Returns ``None`` ONLY when **every** field is ``None`` — i.e. the
    caller is a legacy EA / curl health-check that didn't send any
    state. Any single non-``None`` field is enough to record a partial
    state (Phase A-closeout mini-patch — drops the previous "equity
    OR balance" precondition).

    History-unavailable handling (mini-patch):
      - ``recent_sample_count == -1`` is the canonical sentinel.
        When seen, we fan the ``None`` mark out across all three
        history-derived fields.
      - ``recent_closed_pnl == -1.0`` ALONE is a real loss; we keep it
        verbatim. Same for ``consec_losses == -1`` (which the EA never
        actually emits but defensive parsing tolerates anyway).
    """
    fields = (
        equity, balance, dd_pct, free_margin, margin_level, open_lots,
        open_positions, floating_pnl, spread_pts,
        consec_losses, recent_closed_pnl, recent_sample_count,
    )
    if all(f is None for f in fields):
        return None

    # History-unavailable fan-out: only triggered by recent_sample_count == -1.
    if recent_sample_count == EA_STATE_UNAVAILABLE:
        consec_losses = None
        recent_closed_pnl = None
        recent_sample_count = None

    return EAState(
        equity=equity,
        balance=balance,
        dd_pct=dd_pct,
        free_margin=free_margin,
        margin_level=margin_level,
        open_lots=open_lots,
        open_positions=open_positions,
        floating_pnl=floating_pnl,
        spread_pts=spread_pts,
        consec_losses=consec_losses,
        recent_closed_pnl=recent_closed_pnl,
        recent_sample_count=recent_sample_count,
    )


@dataclass(frozen=True)
class EAStateRecord:
    """Latest :class:`EAState` plus the wall-clock time it was written.

    The store creates this on every ``set`` so :class:`MarketState` can
    compute ``ea_state_age_seconds`` / ``ea_state_stale`` without a
    parallel timestamp dict on the wiring layer.
    """

    state: EAState
    recorded_at: datetime  # tz-aware UTC

    def to_dict(self) -> dict:
        return {
            "state": self.state.to_dict(),
            "recorded_at": self.recorded_at.isoformat(),
        }


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


class EAStateStore:
    """Thread-safe in-memory store of the latest EA state per symbol.

    Each entry is an :class:`EAStateRecord` (state + recorded_at).
    ``set`` defaults ``recorded_at`` to ``datetime.now(timezone.utc)``;
    callers / tests may pass an explicit ``recorded_at`` to seed
    deterministic timestamps.

    Symbol case-insensitivity (Phase B-closeout #2): every symbol passed
    in or out is canonicalized to UPPERCASE. ``set("xauusd", ...)`` and
    ``get("XAUUSD")`` are guaranteed to round-trip; mixed case anywhere
    in the call chain (e.g. Phase C wiring layer using ``sym.lower()``)
    cannot create a phantom shadow record the rule engine then misses.

    Mirrors the :class:`PrevRegimeStore` pattern. The decision_server
    is single-process so an in-memory dict suffices; persistence can be
    layered later if cross-restart durability becomes required.

    Backward compatibility:
        - ``get(symbol)`` still returns the bare :class:`EAState`
          (callers that only need the snapshot don't have to migrate).
        - ``snapshot()`` still returns ``dict[str, EAState]`` for the
          same reason.
        - New methods ``get_record`` / ``get_recorded_at`` /
          ``snapshot_records`` expose the timestamp.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._cache: dict[str, EAStateRecord] = {}

    @staticmethod
    def _canon(symbol: str) -> str:
        return symbol.upper()

    def set(
        self,
        symbol: str,
        state: EAState,
        recorded_at: datetime | None = None,
    ) -> None:
        if recorded_at is None:
            recorded_at = _now_utc()
        if recorded_at.tzinfo is None:
            raise ValueError("recorded_at must be tz-aware (UTC)")
        record = EAStateRecord(state=state, recorded_at=recorded_at)
        with self._lock:
            self._cache[self._canon(symbol)] = record

    def get(self, symbol: str) -> EAState | None:
        with self._lock:
            rec = self._cache.get(self._canon(symbol))
            return rec.state if rec is not None else None

    def get_record(self, symbol: str) -> EAStateRecord | None:
        with self._lock:
            return self._cache.get(self._canon(symbol))

    def get_recorded_at(self, symbol: str) -> datetime | None:
        with self._lock:
            rec = self._cache.get(self._canon(symbol))
            return rec.recorded_at if rec is not None else None

    def snapshot(self) -> dict[str, EAState]:
        """Decoupled ``dict[str, EAState]`` snapshot — legacy shape."""
        with self._lock:
            return {sym: rec.state for sym, rec in self._cache.items()}

    def snapshot_records(self) -> dict[str, EAStateRecord]:
        """Decoupled ``dict[str, EAStateRecord]`` snapshot.

        /status uses this so the JSON includes ``recorded_at``.
        """
        with self._lock:
            return dict(self._cache)

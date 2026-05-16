"""Multi-timeframe state machine + trading-session overlay.

Two orthogonal dimensions, fused into one :class:`TimeframeConsensus`:

  * Vertical (timeframe): D1 / H4 / H1 / M5, each with an
    independent :class:`TimeframeState`.
  * Horizontal (session): ASIA / LONDON / NEWYORK / OVERLAP, each
    with a :class:`SessionProfile` that scales recommended position
    sizing.

Consensus rules:
  * Weighted vote: D1 40%, H4 35%, H1 20%, M5 5%.
  * D1 in ACCUMULATING → ``can_recommend = False`` regardless of the
    short-frame states.
  * ``consensus_score`` = sum of weights for timeframes at or above
    VALIDATING; ``can_recommend`` requires ``score >= 0.7`` AND D1 is
    not ACCUMULATING.

Isolation: this module does NOT import ``rule_engine`` or the
Tier-1 unsealed prod modules. Pure inputs in, frozen state out.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence


__all__ = [
    "CONSENSUS_THRESHOLD",
    "SESSION_PROFILES",
    "SessionProfile",
    "TIMEFRAME_WEIGHTS",
    "TimeframeConsensus",
    "TimeframeState",
    "TradingSession",
    "compute_consensus",
    "detect_session",
    "detect_timeframe_state",
]


class TimeframeState(str, Enum):
    ACCUMULATING = "ACCUMULATING"
    VALIDATING = "VALIDATING"
    READY = "READY"
    ACTIVE = "ACTIVE"


class TradingSession(str, Enum):
    ASIA = "ASIA"
    LONDON = "LONDON"
    NEWYORK = "NEWYORK"
    OVERLAP = "OVERLAP"  # London + New York concurrent (13–17 UTC)


_STATE_RANK: Mapping[TimeframeState, int] = {
    TimeframeState.ACCUMULATING: 0,
    TimeframeState.VALIDATING: 1,
    TimeframeState.READY: 2,
    TimeframeState.ACTIVE: 3,
}


def _state_rank(s: TimeframeState) -> int:
    return _STATE_RANK[s]


# Per spec: D1=40%, H4=35%, H1=20%, M5=5%.
TIMEFRAME_WEIGHTS: Mapping[str, float] = MappingProxyType({
    "D1": 0.40, "H4": 0.35, "H1": 0.20, "M5": 0.05,
})


# Threshold for ``can_recommend = True``.
CONSENSUS_THRESHOLD: float = 0.70


@dataclass(frozen=True)
class SessionProfile:
    session: TradingSession
    vol_multiplier: float
    atr_stop_multiplier: float
    position_scale: float


# Calibrated against XAUUSD typical profile:
# - ASIA: thin liquidity, narrower ranges, low size.
# - LONDON open: trend dominant, deeper books.
# - NEWYORK: macro events, swings.
# - OVERLAP: fattest tape, biggest size.
SESSION_PROFILES: Mapping[TradingSession, SessionProfile] = MappingProxyType({
    TradingSession.ASIA: SessionProfile(
        session=TradingSession.ASIA,
        vol_multiplier=0.5,
        atr_stop_multiplier=1.5,
        position_scale=0.7,
    ),
    TradingSession.LONDON: SessionProfile(
        session=TradingSession.LONDON,
        vol_multiplier=1.5,
        atr_stop_multiplier=2.5,
        position_scale=1.2,
    ),
    TradingSession.NEWYORK: SessionProfile(
        session=TradingSession.NEWYORK,
        vol_multiplier=1.2,
        atr_stop_multiplier=2.2,
        position_scale=1.1,
    ),
    TradingSession.OVERLAP: SessionProfile(
        session=TradingSession.OVERLAP,
        vol_multiplier=1.8,
        atr_stop_multiplier=2.8,
        position_scale=1.3,
    ),
})


@dataclass(frozen=True)
class TimeframeConsensus:
    d1_state: TimeframeState
    h4_state: TimeframeState
    h1_state: TimeframeState
    m5_state: TimeframeState
    active_session: TradingSession
    session_profile: SessionProfile
    consensus_score: float
    can_recommend: bool
    blocking_conditions: tuple[str, ...]
    n_bars_per_timeframe: Mapping[str, int]
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ---------------------------------------------------------------------------
# Session detection
# ---------------------------------------------------------------------------


def detect_session(now: datetime | None = None) -> TradingSession:
    """Map a UTC timestamp onto a trading session.

    Rules (UTC hour):

      * 13–17 → OVERLAP (London + New York both open)
      * 17–22 → NEWYORK
      * 8–13  → LONDON
      * 0–8 and 22–24 → ASIA (Tokyo opens late evening UTC)
    """
    ts = now or datetime.now(timezone.utc)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    else:
        ts = ts.astimezone(timezone.utc)
    h = ts.hour
    if 13 <= h < 17:
        return TradingSession.OVERLAP
    if 17 <= h < 22:
        return TradingSession.NEWYORK
    if 8 <= h < 13:
        return TradingSession.LONDON
    return TradingSession.ASIA


# ---------------------------------------------------------------------------
# Per-timeframe state detection
# ---------------------------------------------------------------------------


def _ohlc_to_log_returns(bars: Sequence[Mapping[str, float]]) -> list[float]:
    out: list[float] = []
    prev_close: float | None = None
    for b in bars:
        try:
            close = float(b.get("close", b.get("c", 0.0)))
        except (TypeError, ValueError):
            continue
        if close <= 0.0:
            prev_close = None
            continue
        if prev_close is not None and prev_close > 0.0:
            out.append(math.log(close / prev_close))
        prev_close = close
    return out


# Minimum bars needed for confident classification per timeframe.
_MIN_BARS_FOR_VALIDATION: Mapping[str, int] = MappingProxyType({
    "D1": 20,  # ~1 month
    "H4": 30,
    "H1": 48,
    "M5": 60,
})


def detect_timeframe_state(
    *,
    timeframe: str,
    bars: Iterable[Mapping[str, float]],
    has_active_candidate: bool = False,
) -> TimeframeState:
    """Classify the per-timeframe state from raw OHLC bars.

    ``has_active_candidate`` lets the caller stamp ACTIVE without
    needing to invent a synthetic OHLC pattern that the rule engine
    would label READY.

    Heuristic:
      * not enough bars → ACCUMULATING
      * absolute trend (cumulative log return) ≥ 1.5× short-window
        stddev AND short stddev not exploding → READY
      * trend forming (≥ 0.75× stddev) → VALIDATING
      * else → ACCUMULATING
    """
    bar_list = list(bars)
    min_bars = _MIN_BARS_FOR_VALIDATION.get(timeframe, 20)

    if has_active_candidate:
        return TimeframeState.ACTIVE

    if len(bar_list) < min_bars:
        return TimeframeState.ACCUMULATING

    returns = _ohlc_to_log_returns(bar_list)
    if len(returns) < 5:
        return TimeframeState.ACCUMULATING

    short_n = max(5, min(20, len(returns) // 2))
    short_returns = returns[-short_n:]
    long_returns = returns

    short_stddev = statistics.pstdev(short_returns) if len(short_returns) >= 2 else 0.0
    cumulative_trend = abs(sum(long_returns[-min_bars:]))

    # Exploding short-window vol relative to long-window stddev:
    # treat as ACCUMULATING (no clear trend, just chop).
    long_stddev = statistics.pstdev(long_returns) if len(long_returns) >= 2 else 0.0
    if long_stddev > 0 and short_stddev > long_stddev * 2.5:
        return TimeframeState.ACCUMULATING

    if short_stddev <= 0:
        return TimeframeState.ACCUMULATING

    trend_ratio = cumulative_trend / short_stddev

    if trend_ratio >= 1.5:
        return TimeframeState.READY
    if trend_ratio >= 0.75:
        return TimeframeState.VALIDATING
    return TimeframeState.ACCUMULATING


# ---------------------------------------------------------------------------
# Consensus
# ---------------------------------------------------------------------------


def _at_or_above_validating(state: TimeframeState) -> bool:
    return _state_rank(state) >= _state_rank(TimeframeState.VALIDATING)


def compute_consensus(
    *,
    d1_bars: Iterable[Mapping[str, float]] | None = None,
    h4_bars: Iterable[Mapping[str, float]] | None = None,
    h1_bars: Iterable[Mapping[str, float]] | None = None,
    m5_bars: Iterable[Mapping[str, float]] | None = None,
    active_timeframes: tuple[str, ...] = (),
    now: datetime | None = None,
    d1_state: TimeframeState | None = None,
    h4_state: TimeframeState | None = None,
    h1_state: TimeframeState | None = None,
    m5_state: TimeframeState | None = None,
) -> TimeframeConsensus:
    """Fuse per-timeframe states + session into a frozen consensus.

    Caller may either pass raw bars (and let the rule engine classify)
    or pass explicit per-timeframe states (useful for tests + for
    callers that already know the live position state).

    ``active_timeframes`` (e.g. ``("H1",)``) marks the named
    timeframes as ACTIVE regardless of bar input — mirrors the
    "we already have a live candidate on H1" case.
    """
    bar_inputs: dict[str, list[dict]] = {
        "D1": list(d1_bars or []),
        "H4": list(h4_bars or []),
        "H1": list(h1_bars or []),
        "M5": list(m5_bars or []),
    }

    states: dict[str, TimeframeState] = {}
    explicit: dict[str, TimeframeState | None] = {
        "D1": d1_state, "H4": h4_state, "H1": h1_state, "M5": m5_state,
    }
    for tf in ("D1", "H4", "H1", "M5"):
        if explicit[tf] is not None:
            states[tf] = explicit[tf]  # type: ignore[assignment]
            continue
        states[tf] = detect_timeframe_state(
            timeframe=tf,
            bars=bar_inputs[tf],
            has_active_candidate=tf in active_timeframes,
        )

    # Score = sum of weights for timeframes at-or-above VALIDATING.
    score = 0.0
    for tf, w in TIMEFRAME_WEIGHTS.items():
        if _at_or_above_validating(states[tf]):
            score += w
    score = round(score, 6)

    blockers: list[str] = []
    if states["D1"] == TimeframeState.ACCUMULATING:
        blockers.append("d1_accumulating")
    if score < CONSENSUS_THRESHOLD:
        blockers.append(
            f"consensus_score_below_threshold:{score:.4f}<"
            f"{CONSENSUS_THRESHOLD:.2f}"
        )

    can_recommend = (
        states["D1"] != TimeframeState.ACCUMULATING
        and score >= CONSENSUS_THRESHOLD
    )

    session = detect_session(now)
    profile = SESSION_PROFILES[session]

    return TimeframeConsensus(
        d1_state=states["D1"],
        h4_state=states["H4"],
        h1_state=states["H1"],
        m5_state=states["M5"],
        active_session=session,
        session_profile=profile,
        consensus_score=score,
        can_recommend=can_recommend,
        blocking_conditions=tuple(blockers),
        n_bars_per_timeframe=MappingProxyType({
            tf: len(bar_inputs[tf]) for tf in ("D1", "H4", "H1", "M5")
        }),
    )

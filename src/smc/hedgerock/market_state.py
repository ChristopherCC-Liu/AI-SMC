"""MarketState — Phase B aggregator for Phase C rule_engine.

A :class:`MarketState` is the single composite the rule engine reads
on each tick. It merges three orthogonal sources:

    1. :class:`MarketFeatures` — derived from the data lake by
       :class:`ForexDataLakeMarketFeaturesProvider`.
    2. :class:`RegimeAssessmentV2` — output of
       :func:`classify_regime_v2` on top of those features (plus
       optional news / spread).
    3. :class:`EAState` — runtime equity / DD / spread / open lots /
       recent realized PnL pushed by the EA via /signal query params.

Phase B does NOT yet plug this into ``decision_server.build_envelope``;
the aggregator is the *Phase C contract*. We land it now (with tests)
so Phase C can drop in a ``rule_engine.derive_envelope(state)`` and
flip the wiring.

Design rules:
    1. Frozen dataclass — rule engine reads, never mutates.
    2. ``aggregate_market_state`` is pure: given the three inputs +
       a clock, returns a deterministic ``MarketState``. No I/O.
    3. ``stale`` flag captures "EA hasn't reported in a while" — the
       rule engine should treat that as a signal to harden safe-mode,
       not silently keep using old EAState numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from smc.hedgerock.decision_server import MarketFeatures
from smc.hedgerock.ea_state import EAState
from smc.hedgerock.regime_classifier_v2 import RegimeAssessmentV2

__all__ = [
    "DEFAULT_EA_STATE_FRESHNESS_SECONDS",
    "MarketState",
    "aggregate_market_state",
]


DEFAULT_EA_STATE_FRESHNESS_SECONDS: int = 60
"""How recently the EA must have reported state for it to count as
``fresh`` for risk-control decisions. With OnTimer at 10 s the EA
should report at least 6 times per minute; 60 s gives a 6× safety
margin before the rule engine should fall back to safe-mode-style
decisions on stale state."""


@dataclass(frozen=True)
class MarketState:
    """Composite read by the Phase C rule engine.

    Attributes:
        symbol: e.g. "XAUUSD".
        generated_at: UTC time the aggregation ran.
        features: lake-derived features (volatility_rank, hh/ll, trend bars).
        regime_assessment: v2 regime + confidence + reason.
        ea_state: latest EA-reported runtime state, or None if never
            reported (or stripped because stale_ea_state is True).
        ea_state_age_seconds: seconds since the EAState was reported
            (None when ea_state is None).
        ea_state_stale: True when the EAState arrived more than
            ``ea_state_freshness_seconds`` ago. Phase C must treat
            ``stale=True`` as "trust features over EAState".
    """

    symbol: str
    generated_at: datetime
    features: MarketFeatures
    regime_assessment: RegimeAssessmentV2
    ea_state: EAState | None
    ea_state_age_seconds: float | None
    ea_state_stale: bool


def aggregate_market_state(
    *,
    symbol: str,
    now: datetime,
    features: MarketFeatures,
    regime_assessment: RegimeAssessmentV2,
    ea_state: EAState | None,
    ea_state_recorded_at: datetime | None,
    freshness_seconds: int = DEFAULT_EA_STATE_FRESHNESS_SECONDS,
) -> MarketState:
    """Build a :class:`MarketState` snapshot.

    Args:
        symbol: Market symbol the snapshot is for.
        now: UTC clock — caller-supplied so tests are deterministic.
        features: Lake-derived MarketFeatures.
        regime_assessment: Output of classify_regime_v2.
        ea_state: Most recent EAState from the EAStateStore, or None
            if the EA has never reported.
        ea_state_recorded_at: When ``ea_state`` was set in the store.
            None when ``ea_state`` is None.
        freshness_seconds: Threshold above which ``ea_state_stale``
            is True. Defaults to ``DEFAULT_EA_STATE_FRESHNESS_SECONDS``.

    Returns:
        A frozen :class:`MarketState`.
    """
    if now.tzinfo is None:
        raise ValueError("'now' must be tz-aware (UTC)")

    age: float | None = None
    stale = False
    if ea_state is not None and ea_state_recorded_at is not None:
        if ea_state_recorded_at.tzinfo is None:
            raise ValueError("'ea_state_recorded_at' must be tz-aware (UTC)")
        delta = now - ea_state_recorded_at
        age = max(0.0, delta.total_seconds())
        stale = age > freshness_seconds
    elif ea_state is not None and ea_state_recorded_at is None:
        # Defensive: state present but no recorded timestamp — treat as
        # stale so the rule engine doesn't trust unbounded-age input.
        stale = True

    return MarketState(
        symbol=symbol.upper(),
        generated_at=now,
        features=features,
        regime_assessment=regime_assessment,
        ea_state=ea_state,
        ea_state_age_seconds=age,
        ea_state_stale=stale,
    )


def utc_now() -> datetime:
    """Trivial UTC-now helper. Re-exported so callers / tests don't
    need to remember the ``tzinfo=timezone.utc`` boilerplate."""
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Convenience: derive everything from a EAStateStore + lake provider.
#
# This is what production wiring will call once per /signal poll. It
# is *not* part of the rule_engine yet — it just builds the state. The
# rule engine reads MarketState; here we just glue the inputs.
# ---------------------------------------------------------------------------


def aggregate_from_stores(
    *,
    symbol: str,
    now: datetime,
    market_features: MarketFeatures,
    regime_assessment: RegimeAssessmentV2,
    ea_state_store,
    freshness_seconds: int = DEFAULT_EA_STATE_FRESHNESS_SECONDS,
) -> MarketState:
    """Read EAStateRecord from an :class:`EAStateStore` and aggregate.

    Phase B-closeout #1: the timestamp now lives in the store itself,
    so the wiring layer no longer maintains a parallel
    ``ea_recorded_at_store`` dict — eliminating an entire class of
    bugs where the two go out of sync.

    Returns a frozen :class:`MarketState`; ``ea_state_stale`` reflects
    the actual store age vs ``freshness_seconds``.
    """
    record = ea_state_store.get_record(symbol.upper())
    if record is None:
        return aggregate_market_state(
            symbol=symbol,
            now=now,
            features=market_features,
            regime_assessment=regime_assessment,
            ea_state=None,
            ea_state_recorded_at=None,
            freshness_seconds=freshness_seconds,
        )
    return aggregate_market_state(
        symbol=symbol,
        now=now,
        features=market_features,
        regime_assessment=regime_assessment,
        ea_state=record.state,
        ea_state_recorded_at=record.recorded_at,
        freshness_seconds=freshness_seconds,
    )

"""Phase B step 4 — MarketState aggregator tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from smc.hedgerock.decision_server import MarketFeatures
from smc.hedgerock.ea_state import EAStateStore, build_ea_state
from smc.hedgerock.market_state import (
    DEFAULT_EA_STATE_FRESHNESS_SECONDS,
    aggregate_from_stores,
    aggregate_market_state,
)
from smc.hedgerock.regime_classifier_v2 import classify_regime_v2


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def now() -> datetime:
    return datetime(2026, 4, 30, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def features() -> MarketFeatures:
    return MarketFeatures(
        volatility_rank=0.55, hh_count=8, ll_count=2, h4_trend_bars=5,
        regime="TREND_UP",
    )


@pytest.fixture
def regime_assessment(features: MarketFeatures):
    return classify_regime_v2(
        volatility_rank=features.volatility_rank,
        h4_trend_bars=features.h4_trend_bars,
        hh_count=features.hh_count, ll_count=features.ll_count,
    )


# ---------------------------------------------------------------------------
# Pure aggregator
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_aggregate_with_no_ea_state_marks_age_none(
    now: datetime, features: MarketFeatures, regime_assessment,
) -> None:
    state = aggregate_market_state(
        symbol="XAUUSD", now=now,
        features=features, regime_assessment=regime_assessment,
        ea_state=None, ea_state_recorded_at=None,
    )
    assert state.symbol == "XAUUSD"
    assert state.ea_state is None
    assert state.ea_state_age_seconds is None
    assert state.ea_state_stale is False  # no state at all ≠ stale


@pytest.mark.unit
def test_aggregate_with_fresh_ea_state_is_not_stale(
    now: datetime, features: MarketFeatures, regime_assessment,
) -> None:
    ea = build_ea_state(equity=10000.0, balance=10100.0)
    recorded_at = now - timedelta(seconds=5)  # 5s old → very fresh
    state = aggregate_market_state(
        symbol="xauusd", now=now,
        features=features, regime_assessment=regime_assessment,
        ea_state=ea, ea_state_recorded_at=recorded_at,
    )
    assert state.symbol == "XAUUSD"  # canonicalized
    assert state.ea_state == ea
    assert state.ea_state_age_seconds == pytest.approx(5.0)
    assert state.ea_state_stale is False


@pytest.mark.unit
def test_aggregate_with_stale_ea_state_marks_stale(
    now: datetime, features: MarketFeatures, regime_assessment,
) -> None:
    ea = build_ea_state(equity=10000.0)
    recorded_at = now - timedelta(seconds=DEFAULT_EA_STATE_FRESHNESS_SECONDS + 30)
    state = aggregate_market_state(
        symbol="XAUUSD", now=now,
        features=features, regime_assessment=regime_assessment,
        ea_state=ea, ea_state_recorded_at=recorded_at,
    )
    assert state.ea_state_stale is True
    assert state.ea_state_age_seconds is not None
    assert state.ea_state_age_seconds > DEFAULT_EA_STATE_FRESHNESS_SECONDS


@pytest.mark.unit
def test_aggregate_with_state_but_missing_timestamp_treats_as_stale(
    now: datetime, features: MarketFeatures, regime_assessment,
) -> None:
    """Defensive: present-state-but-no-recorded-at is suspicious."""
    ea = build_ea_state(equity=10000.0)
    state = aggregate_market_state(
        symbol="XAUUSD", now=now,
        features=features, regime_assessment=regime_assessment,
        ea_state=ea, ea_state_recorded_at=None,
    )
    assert state.ea_state_stale is True


@pytest.mark.unit
def test_aggregate_rejects_naive_clock(
    features: MarketFeatures, regime_assessment,
) -> None:
    with pytest.raises(ValueError, match="tz-aware"):
        aggregate_market_state(
            symbol="XAUUSD", now=datetime(2026, 4, 30, 12, 0),
            features=features, regime_assessment=regime_assessment,
            ea_state=None, ea_state_recorded_at=None,
        )


@pytest.mark.unit
def test_aggregate_rejects_naive_recorded_at(
    now: datetime, features: MarketFeatures, regime_assessment,
) -> None:
    with pytest.raises(ValueError, match="tz-aware"):
        aggregate_market_state(
            symbol="XAUUSD", now=now,
            features=features, regime_assessment=regime_assessment,
            ea_state=build_ea_state(equity=10000.0),
            ea_state_recorded_at=datetime(2026, 4, 30, 11, 59, 0),
        )


# ---------------------------------------------------------------------------
# Convenience wrapper from EAStateStore
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_aggregate_from_stores_pulls_state_with_explicit_recorded_at(
    now: datetime, features: MarketFeatures, regime_assessment,
) -> None:
    """Phase B-closeout #1: timestamp lives in the store; no parallel dict."""
    ea_store = EAStateStore()
    ea = build_ea_state(equity=10000.0, balance=10100.0)
    ea_store.set("XAUUSD", ea, recorded_at=now - timedelta(seconds=10))

    state = aggregate_from_stores(
        symbol="XAUUSD", now=now,
        market_features=features, regime_assessment=regime_assessment,
        ea_state_store=ea_store,
    )
    assert state.ea_state == ea
    assert state.ea_state_age_seconds == pytest.approx(10.0)
    assert state.ea_state_stale is False


@pytest.mark.unit
def test_aggregate_from_stores_uses_default_recorded_at_now(
    features: MarketFeatures, regime_assessment,
) -> None:
    """When set() is called without recorded_at, the store stamps now()
    automatically. Aggregate-from-stores should treat such a record as fresh."""
    ea_store = EAStateStore()
    ea = build_ea_state(equity=10000.0)
    ea_store.set("XAUUSD", ea)  # default recorded_at = now()

    now = datetime.now(timezone.utc) + timedelta(seconds=2)
    state = aggregate_from_stores(
        symbol="XAUUSD", now=now,
        market_features=features, regime_assessment=regime_assessment,
        ea_state_store=ea_store,
    )
    assert state.ea_state_stale is False
    assert state.ea_state_age_seconds is not None
    assert state.ea_state_age_seconds < 60


@pytest.mark.unit
def test_aggregate_from_stores_marks_stale_when_age_exceeds_freshness(
    now: datetime, features: MarketFeatures, regime_assessment,
) -> None:
    """End-to-end through real EAStateStore — confirm freshness check
    works without an external timestamp dict."""
    from smc.hedgerock.market_state import DEFAULT_EA_STATE_FRESHNESS_SECONDS

    ea_store = EAStateStore()
    ea = build_ea_state(equity=10000.0)
    ea_store.set("XAUUSD", ea,
                 recorded_at=now - timedelta(seconds=DEFAULT_EA_STATE_FRESHNESS_SECONDS + 30))

    state = aggregate_from_stores(
        symbol="XAUUSD", now=now,
        market_features=features, regime_assessment=regime_assessment,
        ea_state_store=ea_store,
    )
    assert state.ea_state_stale is True
    assert state.ea_state_age_seconds is not None
    assert state.ea_state_age_seconds > DEFAULT_EA_STATE_FRESHNESS_SECONDS


@pytest.mark.unit
def test_aggregate_from_stores_handles_missing_record(
    now: datetime, features: MarketFeatures, regime_assessment,
) -> None:
    ea_store = EAStateStore()  # empty
    state = aggregate_from_stores(
        symbol="XAUUSD", now=now,
        market_features=features, regime_assessment=regime_assessment,
        ea_state_store=ea_store,
    )
    assert state.ea_state is None
    assert state.ea_state_age_seconds is None
    assert state.ea_state_stale is False


@pytest.mark.unit
def test_market_state_is_frozen(
    now: datetime, features: MarketFeatures, regime_assessment,
) -> None:
    state = aggregate_market_state(
        symbol="XAUUSD", now=now,
        features=features, regime_assessment=regime_assessment,
        ea_state=None, ea_state_recorded_at=None,
    )
    with pytest.raises(Exception):
        state.symbol = "BTCUSD"  # type: ignore[misc]


@pytest.mark.unit
def test_market_state_carries_through_regime_assessment_diagnostics(
    now: datetime, features: MarketFeatures, regime_assessment,
) -> None:
    """The rule engine relies on rule_votes for decision_log dumps."""
    state = aggregate_market_state(
        symbol="XAUUSD", now=now,
        features=features, regime_assessment=regime_assessment,
        ea_state=None, ea_state_recorded_at=None,
    )
    assert state.regime_assessment.regime in {"trend_up", "range", "unknown"}
    assert state.regime_assessment.reason  # non-empty
    assert isinstance(state.regime_assessment.rule_votes, tuple)

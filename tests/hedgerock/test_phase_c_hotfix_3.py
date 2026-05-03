"""Phase C-hotfix-3 targeted tests.

Patch 1 — tri-state transition_lock_override semantics
Patch 2 — aggressive step-up requires recent_sample_count ≥ MIN
          AND recent_closed_pnl ≥ 0
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from smc.hedgerock.decision_server import (
    EAStateStore,
    MarketFeatures,
    PrevRegimeStore,
    PrevRegimeV2Store,
    build_envelope,
    create_app,
)
from smc.hedgerock.ea_state import build_ea_state
from smc.hedgerock.market_state import aggregate_market_state
from smc.hedgerock.regime_classifier_v2 import RegimeAssessmentV2
from smc.hedgerock.rule_engine import derive_envelope_params


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now() -> datetime:
    return datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)


def _features_v2_range_legacy(legacy_regime: str) -> MarketFeatures:
    """Features that classify_regime_v2 → "range" (vol=0.45, balanced HH/LL,
    h4_trend_bars=1, range#2 rule fires at 0.65 confidence). Caller picks
    the legacy regime so the test can drive features.regime independently
    of what the v2 classifier actually says."""
    return MarketFeatures(
        volatility_rank=0.45, hh_count=3, ll_count=3,
        h4_trend_bars=1, regime=legacy_regime,  # type: ignore[arg-type]
    )


class _SwitchableProvider:
    """Returns one MarketFeatures up to switch_after, then another."""

    def __init__(self, first: MarketFeatures, then: MarketFeatures, *,
                 switch_after: int = 1) -> None:
        self._first = first
        self._then = then
        self._switch_after = switch_after
        self.call_count = 0

    def get_features(self, symbol: str) -> MarketFeatures:
        self.call_count += 1
        if self.call_count <= self._switch_after:
            return self._first
        return self._then


def _state(*, regime_v2: str, confidence: float, ea_state, ea_recorded_at, now):
    assessment = RegimeAssessmentV2(
        regime=regime_v2,  # type: ignore[arg-type]
        confidence=confidence,
        reason=f"test regime={regime_v2}",
        rule_votes=(),
    )
    # Pass minimal MarketFeatures — the rule_engine doesn't read it.
    return aggregate_market_state(
        symbol="XAUUSD", now=now,
        features=MarketFeatures(
            volatility_rank=0.45, hh_count=3, ll_count=3,
            h4_trend_bars=1, regime="CONSOLIDATION",
        ),
        regime_assessment=assessment,
        ea_state=ea_state,
        ea_state_recorded_at=ea_recorded_at,
    )


# ---------------------------------------------------------------------------
# Patch 1 — tri-state transition_lock semantics
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_build_envelope_override_not_provided_falls_back_to_legacy() -> None:
    """When wiring does NOT pass the override flag, build_envelope uses
    the legacy compute_lock_until on features.regime + prev_regime."""
    now = _now()
    feats = MarketFeatures(
        volatility_rank=0.5, hh_count=4, ll_count=2,
        h4_trend_bars=3, regime="CONSOLIDATION",
    )
    # prev_regime=TREND_UP → CONSOLIDATION = legacy distance 2 = 3600 s
    env = build_envelope(
        "XAUUSD", feats, prev_regime="TREND_UP", now=now,
        # transition_lock_override_provided default False
    )
    assert env.transition_lock_until_ts is not None  # legacy fired


@pytest.mark.unit
def test_build_envelope_override_provided_with_none_means_no_lock() -> None:
    """Phase C-hotfix-3 #1: tri-state ``provided=True + override=None``
    must produce ``transition_lock_until_ts=None`` — NOT fall back to
    legacy."""
    now = _now()
    feats = MarketFeatures(
        volatility_rank=0.5, hh_count=4, ll_count=2,
        h4_trend_bars=3, regime="CONSOLIDATION",
    )
    # legacy would lock; override says no lock.
    env = build_envelope(
        "XAUUSD", feats, prev_regime="TREND_UP", now=now,
        transition_lock_until_override=None,
        transition_lock_override_provided=True,
    )
    assert env.transition_lock_until_ts is None


@pytest.mark.unit
def test_build_envelope_override_provided_with_datetime_used_verbatim() -> None:
    now = _now()
    feats = MarketFeatures(
        volatility_rank=0.5, hh_count=4, ll_count=2,
        h4_trend_bars=3, regime="CONSOLIDATION",
    )
    target = now + timedelta(minutes=42)
    env = build_envelope(
        "XAUUSD", feats, prev_regime="TREND_UP", now=now,
        transition_lock_until_override=target,
        transition_lock_override_provided=True,
    )
    assert env.transition_lock_until_ts == target


@pytest.mark.unit
def test_v2_lock_none_does_not_inherit_legacy_lock() -> None:
    """End-to-end: rule_engine wiring fires; v2 says same regime (no lock)
    but legacy MarketRegimeAI changed → legacy compute_lock_until would
    set 3600 s. Wiring must use v2's None and NOT fall back to legacy.

    Setup:
        Poll 1 features: legacy=TREND_UP, classifier_v2=range
        Poll 2 features: legacy=CONSOLIDATION, classifier_v2=range
    Expected:
        Poll 2 envelope.transition_lock_until_ts is None
        (legacy distance TREND_UP→CONSOLIDATION=2 → 3600s would have
         been set without the tri-state fix)
    """
    poll1 = _features_v2_range_legacy("TREND_UP")
    poll2 = _features_v2_range_legacy("CONSOLIDATION")
    market = _SwitchableProvider(poll1, poll2, switch_after=1)
    legacy_store = PrevRegimeStore()
    v2_store = PrevRegimeV2Store()
    app = create_app(
        market, store=legacy_store,
        prev_regime_v2_store=v2_store,
        enable_debate=False, enable_rule_engine=True,
    )
    with TestClient(app) as client:
        b1 = client.get(
            "/signal",
            params={
                "symbol": "XAUUSD",
                "equity": 10000.0, "balance": 10000.0,
                "dd_pct": 0.0, "spread_pts": 20,
            },
        ).json()
        # Poll 1: prev legacy=None and prev_v2=None → no lock either way.
        assert b1["regime"] == "range"
        assert b1["transition_lock_until_ts"] is None

        b2 = client.get(
            "/signal",
            params={
                "symbol": "XAUUSD",
                "equity": 10000.0, "balance": 10000.0,
                "dd_pct": 0.0, "spread_pts": 20,
            },
        ).json()
        assert b2["regime"] == "range"
        # Legacy: TREND_UP → CONSOLIDATION distance = 2 = 3600 s.
        # v2:     range    → range            distance = 0 = no lock.
        # tri-state ensures the v2 None wins.
        assert b2["transition_lock_until_ts"] is None


# ---------------------------------------------------------------------------
# Patch 2 — aggressive step-up sample threshold
# ---------------------------------------------------------------------------


def _build_ea(*, recent_n: int | None, recent_pnl: float | None = 0.0):
    """EA state with full risk snapshot but variable history sample size."""
    kwargs = dict(
        equity=10000.0, balance=10000.0, dd_pct=0.0, spread_pts=20,
        consec_losses=0,
    )
    if recent_n is not None:
        kwargs["recent_sample_count"] = recent_n
        kwargs["recent_closed_pnl"] = recent_pnl
    return build_ea_state(**kwargs)


@pytest.mark.parametrize("sample", [0, 1, 3, 4])
@pytest.mark.unit
def test_sample_count_below_minimum_caps_at_normal_tier(sample: int) -> None:
    """Phase C-hotfix-3 #2: 0 ≤ recent_sample_count < MIN(5) → insufficient
    evidence → cap at normal even at high confidence."""
    now = _now()
    ea = _build_ea(recent_n=sample, recent_pnl=10.0)  # positive but not enough samples
    state = _state(regime_v2="range", confidence=0.85,
                   ea_state=ea, ea_recorded_at=now - timedelta(seconds=5), now=now)
    p = derive_envelope_params(state, prev_envelope=None)
    assert p.mode == "hedgerock"
    assert p.risk_tier == "normal"
    assert p.lot_factor == pytest.approx(1.0)
    assert "insufficient_sample" in p.reason


@pytest.mark.unit
def test_sample_count_at_minimum_with_positive_pnl_allows_aggressive() -> None:
    """sample_count == MIN (5) with non-negative pnl → aggressive."""
    now = _now()
    ea = _build_ea(recent_n=5, recent_pnl=0.0)  # exactly the threshold, neutral pnl
    state = _state(regime_v2="range", confidence=0.85,
                   ea_state=ea, ea_recorded_at=now - timedelta(seconds=5), now=now)
    p = derive_envelope_params(state, prev_envelope=None)
    assert p.mode == "hedgerock"
    assert p.risk_tier == "aggressive"
    assert p.lot_factor == pytest.approx(1.5)


@pytest.mark.unit
def test_sample_count_above_minimum_with_positive_pnl_aggressive() -> None:
    now = _now()
    ea = _build_ea(recent_n=20, recent_pnl=42.0)
    state = _state(regime_v2="range", confidence=0.85,
                   ea_state=ea, ea_recorded_at=now - timedelta(seconds=5), now=now)
    p = derive_envelope_params(state, prev_envelope=None)
    assert p.risk_tier == "aggressive"


@pytest.mark.unit
def test_sample_count_at_minimum_with_negative_pnl_caps_at_normal() -> None:
    """Existing recent-loss gate still applies at sample == MIN."""
    now = _now()
    ea = _build_ea(recent_n=5, recent_pnl=-1.0)
    state = _state(regime_v2="range", confidence=0.85,
                   ea_state=ea, ea_recorded_at=now - timedelta(seconds=5), now=now)
    p = derive_envelope_params(state, prev_envelope=None)
    assert p.risk_tier == "normal"
    # The recent-pnl blocker should fire (more specific than insufficient_sample).
    assert "recent_pnl" in p.reason


@pytest.mark.unit
def test_sample_count_none_caps_at_normal_with_history_unavailable_reason() -> None:
    """recent_sample_count=None → history_unavailable bucket (not
    insufficient_sample)."""
    now = _now()
    ea = _build_ea(recent_n=None)
    state = _state(regime_v2="range", confidence=0.85,
                   ea_state=ea, ea_recorded_at=now - timedelta(seconds=5), now=now)
    p = derive_envelope_params(state, prev_envelope=None)
    assert p.risk_tier == "normal"
    assert "history_unavailable" in p.reason

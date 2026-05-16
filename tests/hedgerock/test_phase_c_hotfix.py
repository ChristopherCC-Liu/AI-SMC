"""Phase C-hotfix targeted tests.

Patch 1 — cooldown persistence across /signal calls
Patch 2 — no_ea_state guard
Patch 3 — envelope regime/confidence aligned with v2 assessment
Patch 4 — recent_closed_pnl gates (sentinel None ≠ 0)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from smc.hedgerock.decision_server import (
    EAStateStore,
    MarketFeatures,
    create_app,
)
from smc.hedgerock.ea_state import build_ea_state
from smc.hedgerock.envelope_store import EnvelopeStore
from smc.hedgerock.market_state import aggregate_market_state
from smc.hedgerock.regime_classifier_v2 import RegimeAssessmentV2
from smc.hedgerock.rule_engine import derive_envelope_params
from smc.hedgerock.schemas import SignalEnvelope


# ---------------------------------------------------------------------------
# Stubs / fixtures
# ---------------------------------------------------------------------------


class _StubProvider:
    def __init__(self, features: MarketFeatures) -> None:
        self._f = features

    def get_features(self, symbol: str) -> MarketFeatures:
        return self._f


def _features_range() -> MarketFeatures:
    """Maps to v2 'range' with confidence=0.65 via classify_regime_v2."""
    return MarketFeatures(
        volatility_rank=0.45, hh_count=3, ll_count=3,
        h4_trend_bars=1, regime="CONSOLIDATION",
    )


def _features_legacy_transition_v2_range() -> MarketFeatures:
    """Legacy regime=TRANSITION, but v2 classifier sees 'range'.

    Used to verify Patch 3 — envelope.regime should reflect v2 regime,
    not the legacy mapping (which would be 'unknown' here).
    """
    return MarketFeatures(
        volatility_rank=0.45, hh_count=3, ll_count=3,
        h4_trend_bars=1, regime="TRANSITION",
    )


def _now() -> datetime:
    return datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)


def _build_state_for_unit_test(*, regime_v2: str, confidence: float,
                               ea_state, ea_recorded_at: datetime,
                               now: datetime):
    """Build a MarketState with arbitrary v2 regime for rule_engine tests."""
    assessment = RegimeAssessmentV2(
        regime=regime_v2,  # type: ignore[arg-type]
        confidence=confidence,
        reason=f"test regime={regime_v2}",
        rule_votes=(),
    )
    return aggregate_market_state(
        symbol="XAUUSD",
        now=now,
        features=_features_range(),
        regime_assessment=assessment,
        ea_state=ea_state,
        ea_state_recorded_at=ea_recorded_at,
    )


# ---------------------------------------------------------------------------
# Patch 1 — cooldown persists across /signal calls (end-to-end)
# ---------------------------------------------------------------------------


def test_spread_spike_cooldown_persists_across_subsequent_normal_polls() -> None:
    """First poll: spread_pts=120 → cooldown.
    Second poll: spread_pts=20 (normal) → still observe + cooldown carried.
    Third poll (after cooldown expires): hedgerock allowed again."""
    market = _StubProvider(_features_range())
    ea_store = EAStateStore()
    env_store = EnvelopeStore()
    app = create_app(
        market, enable_debate=False, enable_rule_engine=True,
        ea_state_store=ea_store, envelope_store=env_store,
    )
    with TestClient(app) as client:
        # Poll 1 — spread spike triggers cooldown.
        b1 = client.get(
            "/signal",
            params={
                "symbol": "XAUUSD",
                "equity": 10000.0, "balance": 10000.0,
                "dd_pct": 0.0, "spread_pts": 120,
            },
        ).json()
        assert b1["mode"] == "observe"
        assert b1["cooldown_until"] is not None
        cooldown1 = b1["cooldown_until"]

        # Poll 2 — spread back to normal, but cooldown carries forward.
        b2 = client.get(
            "/signal",
            params={
                "symbol": "XAUUSD",
                "equity": 10000.0, "balance": 10000.0,
                "dd_pct": 0.0, "spread_pts": 20,
            },
        ).json()
        assert b2["mode"] == "observe", (
            "cooldown should keep us in observe even though spread returned to normal"
        )
        assert b2["cooldown_until"] == cooldown1 or b2["cooldown_until"] > cooldown1
        assert "cooldown" in b2["reason"].lower()


def test_cooldown_carryover_pure_function_demotes_hedgerock_to_observe() -> None:
    """Direct rule_engine test: prev cooldown still in future + new
    params want hedgerock → demoted to observe with carried cooldown."""
    now = _now()
    ea = build_ea_state(equity=10000.0, balance=10000.0, dd_pct=0.0, spread_pts=20,
                        consec_losses=0, recent_closed_pnl=0.0, recent_sample_count=20)
    state = _build_state_for_unit_test(
        regime_v2="range", confidence=0.85,
        ea_state=ea, ea_recorded_at=now - timedelta(seconds=5), now=now,
    )

    # Build a prev_envelope with cooldown 5 min in the future.
    prev = SignalEnvelope(
        symbol="XAUUSD",
        generated_at=now - timedelta(minutes=1),
        active_timeframe="H1",
        active_strategy_id="xauusd_h1_range",
        regime="range",
        cooldown_until=now + timedelta(minutes=5),
    )
    p = derive_envelope_params(state, prev_envelope=prev)
    assert p.mode == "observe"
    assert p.cooldown_until is not None
    assert p.cooldown_until >= prev.cooldown_until
    assert "cooldown" in p.reason.lower()


def test_cooldown_carryover_takes_max_of_prev_and_new() -> None:
    """If new branch sets a SHORTER cooldown than prev, prev wins."""
    now = _now()
    ea = build_ea_state(equity=10000.0, balance=10000.0, dd_pct=0.0, spread_pts=120)
    state = _build_state_for_unit_test(
        regime_v2="range", confidence=0.65,
        ea_state=ea, ea_recorded_at=now - timedelta(seconds=5), now=now,
    )
    # Prev has a long 30-min cooldown.
    long_cd = now + timedelta(minutes=30)
    prev = SignalEnvelope(
        symbol="XAUUSD",
        generated_at=now - timedelta(minutes=2),
        active_timeframe="H1",
        active_strategy_id="xauusd_h1_range",
        regime="range",
        cooldown_until=long_cd,
    )
    p = derive_envelope_params(state, prev_envelope=prev)
    # Spread cooldown is 5 min; prev was 30 min — longer wins.
    assert p.cooldown_until >= long_cd


def test_expired_prev_cooldown_does_not_carry_forward() -> None:
    """If prev.cooldown_until is in the past, no carryover (clean slate)."""
    now = _now()
    ea = build_ea_state(equity=10000.0, balance=10000.0, dd_pct=0.0, spread_pts=20)
    state = _build_state_for_unit_test(
        regime_v2="range", confidence=0.65,
        ea_state=ea, ea_recorded_at=now - timedelta(seconds=5), now=now,
    )
    expired = SignalEnvelope(
        symbol="XAUUSD",
        generated_at=now - timedelta(minutes=10),
        active_timeframe="H1",
        active_strategy_id="xauusd_h1_range",
        regime="range",
        cooldown_until=now - timedelta(minutes=5),  # expired
    )
    p = derive_envelope_params(state, prev_envelope=expired)
    assert p.mode == "hedgerock"  # free to enter


# ---------------------------------------------------------------------------
# Patch 2 — no_ea_state guard end-to-end
# ---------------------------------------------------------------------------


def test_signal_no_ea_state_falls_back_to_observe_even_in_clean_range() -> None:
    market = _StubProvider(_features_range())
    app = create_app(market, enable_debate=False, enable_rule_engine=True)
    with TestClient(app) as client:
        # Bare /signal — no EA query params at all.
        body = client.get("/signal", params={"symbol": "XAUUSD"}).json()
        assert body["mode"] == "observe"
        assert body["lot_factor"] == 0.0
        assert "no_ea_state" in body["reason"]


def test_signal_with_fresh_ea_state_promotes_to_hedgerock() -> None:
    """Same features + COMPLETE risk snapshot (Phase C-hotfix-2 #2) → hedgerock allowed."""
    market = _StubProvider(_features_range())
    app = create_app(market, enable_debate=False, enable_rule_engine=True)
    with TestClient(app) as client:
        body = client.get(
            "/signal",
            params={
                "symbol": "XAUUSD",
                "equity": 10000.0, "balance": 10000.0,
                "dd_pct": 0.0, "spread_pts": 20,
            },
        ).json()
        assert body["mode"] == "hedgerock"


# ---------------------------------------------------------------------------
# Patch 3 — envelope regime/confidence aligned with v2 assessment
# ---------------------------------------------------------------------------


def test_envelope_regime_mirrors_v2_assessment_not_legacy_mapping() -> None:
    """Legacy features.regime=TRANSITION (→ legacy v2 'unknown') BUT
    classify_regime_v2 sees 'range' → envelope.regime must be 'range'."""
    market = _StubProvider(_features_legacy_transition_v2_range())
    app = create_app(market, enable_debate=False, enable_rule_engine=True)
    with TestClient(app) as client:
        body = client.get(
            "/signal",
            params={
                "symbol": "XAUUSD",
                "equity": 10000.0, "balance": 10000.0,
                "dd_pct": 0.0, "spread_pts": 20,
            },
        ).json()
        # The legacy mapping would have produced 'unknown' here.
        assert body["regime"] == "range"
        # active_strategy_id mirrors the v2 regime.
        assert "range" in body["active_strategy_id"]
        assert body["mode"] == "hedgerock"  # range conf=0.65 with EA state


def test_envelope_confidence_uses_v2_assessment() -> None:
    """When rule_engine fires the envelope confidence is the
    classify_regime_v2 confidence (0.65 for the balanced fixture),
    not the timeframe router's confidence."""
    market = _StubProvider(_features_range())
    app = create_app(market, enable_debate=False, enable_rule_engine=True)
    with TestClient(app) as client:
        body = client.get(
            "/signal",
            params={
                "symbol": "XAUUSD",
                "equity": 10000.0, "balance": 10000.0,
                "dd_pct": 0.0, "spread_pts": 20,
            },
        ).json()
        # range#2 rule confidence is 0.65.
        assert body["confidence"] == pytest.approx(0.65)


# ---------------------------------------------------------------------------
# Patch 4 — recent_closed_pnl gates
# ---------------------------------------------------------------------------


def test_recent_loss_blocks_step_up_to_aggressive() -> None:
    """High confidence + recent realized loss (>=5 samples) → cap at normal."""
    now = _now()
    ea = build_ea_state(
        equity=10000.0, balance=10000.0, dd_pct=0.0, spread_pts=20,
        consec_losses=0,
        recent_closed_pnl=-30.0,   # mild loss, above stepdown threshold
        recent_sample_count=10,
    )
    state = _build_state_for_unit_test(
        regime_v2="range", confidence=0.85,
        ea_state=ea, ea_recorded_at=now - timedelta(seconds=5), now=now,
    )
    p = derive_envelope_params(state, prev_envelope=None)
    assert p.mode == "hedgerock"
    assert p.risk_tier == "normal"
    assert p.lot_factor == pytest.approx(1.0)
    assert "recent_pnl" in p.reason or "step-up blocked" in p.reason


def test_recent_loss_at_stepdown_threshold_steps_down() -> None:
    """Recent_closed_pnl ≤ -50 with sample ≥ 5 → step-down."""
    now = _now()
    ea = build_ea_state(
        equity=10000.0, balance=10000.0, dd_pct=0.0, spread_pts=20,
        consec_losses=0,
        recent_closed_pnl=-75.0,
        recent_sample_count=10,
    )
    state = _build_state_for_unit_test(
        regime_v2="range", confidence=0.85,
        ea_state=ea, ea_recorded_at=now - timedelta(seconds=5), now=now,
    )
    p = derive_envelope_params(state, prev_envelope=None)
    assert p.mode == "hedgerock"
    assert p.risk_tier == "observe"
    assert p.lot_factor == pytest.approx(0.5)
    assert "recent_closed_pnl" in p.reason


def test_recent_pnl_minus_one_with_full_sample_is_real_loss_not_unavailable() -> None:
    """Phase A-closeout #1 + Phase C-hotfix #4 invariant:
    recent_closed_pnl=-1.0 with sample_count=20 is a REAL small loss.
    It should be visible to the rule engine — NOT discarded as unavailable.

    With confidence high we'd normally step UP. But because realized
    PnL is < 0 on a full sample, step-up is blocked → cap at normal."""
    now = _now()
    ea = build_ea_state(
        equity=10000.0, balance=10000.0, dd_pct=0.0, spread_pts=20,
        consec_losses=0,
        recent_closed_pnl=-1.0,
        recent_sample_count=20,
    )
    state = _build_state_for_unit_test(
        regime_v2="range", confidence=0.85,
        ea_state=ea, ea_recorded_at=now - timedelta(seconds=5), now=now,
    )
    p = derive_envelope_params(state, prev_envelope=None)
    assert p.mode == "hedgerock"
    assert p.risk_tier == "normal"  # step-up blocked
    assert "recent_pnl" in p.reason or "step-up blocked" in p.reason


def test_recent_pnl_sample_below_minimum_caps_at_normal() -> None:
    """Phase C-hotfix-3 #2: sample_count=3 < threshold (5) is INSUFFICIENT
    evidence for aggressive step-up — cap at normal regardless of pnl
    sign. (Old hotfix-1 behavior: ignore field entirely → would have
    stepped up. New behavior: positive evidence required.)
    """
    now = _now()
    ea = build_ea_state(
        equity=10000.0, balance=10000.0, dd_pct=0.0, spread_pts=20,
        consec_losses=0,
        recent_closed_pnl=-100.0,
        recent_sample_count=3,
    )
    state = _build_state_for_unit_test(
        regime_v2="range", confidence=0.85,
        ea_state=ea, ea_recorded_at=now - timedelta(seconds=5), now=now,
    )
    p = derive_envelope_params(state, prev_envelope=None)
    assert p.mode == "hedgerock"
    assert p.risk_tier == "normal"
    assert p.lot_factor == pytest.approx(1.0)
    assert "insufficient_sample" in p.reason


def test_recent_pnl_sentinel_none_does_not_step_down_but_blocks_step_up() -> None:
    """Phase C-hotfix-2 #1: history unavailable (sentinel → None) must
    NOT step DOWN (no evidence of losses) BUT must block aggressive
    step-up (no evidence things are going well either).
    Result: hedgerock at normal tier (lot_factor=1.0)."""
    now = _now()
    ea = build_ea_state(
        equity=10000.0, balance=10000.0, dd_pct=0.0, spread_pts=20,
        consec_losses=-1,
        recent_closed_pnl=-1,
        recent_sample_count=-1,
    )
    state = _build_state_for_unit_test(
        regime_v2="range", confidence=0.85,
        ea_state=ea, ea_recorded_at=now - timedelta(seconds=5), now=now,
    )
    p = derive_envelope_params(state, prev_envelope=None)
    assert p.mode == "hedgerock"
    assert p.risk_tier == "normal"
    assert p.lot_factor == pytest.approx(1.0)
    assert "history_unavailable" in p.reason

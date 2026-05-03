"""Phase C-hotfix-4 — history completeness gate for aggressive step-up.

Aggressive step-up now requires the FULL history triple, not just one or
two of the three fields:
    - consec_losses        is not None
    - recent_sample_count  is not None  AND  >= _RECENT_PNL_MIN_SAMPLE
    - recent_closed_pnl    is not None  AND  >= 0

Any single missing field → cap at normal tier with a reason that names
the absent piece of evidence so an operator reading the decision_log
can tell which producer side dropped the field.
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
from smc.hedgerock.ea_state import EAState, build_ea_state
from smc.hedgerock.market_state import aggregate_market_state
from smc.hedgerock.regime_classifier_v2 import RegimeAssessmentV2
from smc.hedgerock.rule_engine import derive_envelope_params


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now() -> datetime:
    return datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)


def _features_v2_range() -> MarketFeatures:
    """Features that classify_regime_v2 → 'range' at conf=0.65 (range#2 rule).

    For the aggressive-step-up tests we'll feed the rule_engine an
    explicit RegimeAssessmentV2(regime='range', confidence=0.85) so we
    bypass the classifier and isolate the history-completeness gate.
    """
    return MarketFeatures(
        volatility_rank=0.45, hh_count=3, ll_count=3,
        h4_trend_bars=1, regime="CONSOLIDATION",
    )


class _StubProvider:
    def __init__(self, features: MarketFeatures) -> None:
        self._f = features

    def get_features(self, symbol: str) -> MarketFeatures:
        return self._f


def _state(*, ea_state, now: datetime, confidence: float = 0.85):
    """Build a MarketState with v2='range' at the given confidence."""
    assessment = RegimeAssessmentV2(
        regime="range", confidence=confidence,
        reason="test fixture", rule_votes=(),
    )
    return aggregate_market_state(
        symbol="XAUUSD", now=now,
        features=_features_v2_range(),
        regime_assessment=assessment,
        ea_state=ea_state,
        ea_state_recorded_at=now - timedelta(seconds=5),
    )


# ---------------------------------------------------------------------------
# Pure rule_engine — completeness gate
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_full_history_triple_with_positive_pnl_allows_aggressive() -> None:
    """consec=0 + sample=5 + pnl=0 → aggressive (per user spec)."""
    now = _now()
    ea = build_ea_state(
        equity=10000.0, balance=10000.0, dd_pct=0.0, spread_pts=20,
        consec_losses=0, recent_closed_pnl=0.0, recent_sample_count=5,
    )
    p = derive_envelope_params(_state(ea_state=ea, now=now), prev_envelope=None)
    assert p.mode == "hedgerock"
    assert p.risk_tier == "aggressive"
    assert p.lot_factor == pytest.approx(1.5)


@pytest.mark.unit
def test_missing_recent_closed_pnl_caps_at_normal() -> None:
    """Sample size sufficient + consec_losses present BUT
    recent_closed_pnl is None → cap at normal tier.

    EAState dataclass is frozen so we use dataclasses.replace via
    build_ea_state's full kwargs, then None-out one field via direct
    construction.
    """
    now = _now()
    # Build a full state then replace recent_closed_pnl with None to
    # simulate a producer that dropped the field.
    full = build_ea_state(
        equity=10000.0, balance=10000.0, dd_pct=0.0, spread_pts=20,
        consec_losses=0, recent_closed_pnl=10.0, recent_sample_count=20,
    )
    assert full is not None
    ea = EAState(
        equity=full.equity, balance=full.balance, dd_pct=full.dd_pct,
        free_margin=full.free_margin, margin_level=full.margin_level,
        open_lots=full.open_lots, open_positions=full.open_positions,
        floating_pnl=full.floating_pnl, spread_pts=full.spread_pts,
        consec_losses=full.consec_losses,
        recent_closed_pnl=None,                     # ← dropped
        recent_sample_count=full.recent_sample_count,
    )
    p = derive_envelope_params(_state(ea_state=ea, now=now), prev_envelope=None)
    assert p.risk_tier == "normal"
    assert p.lot_factor == pytest.approx(1.0)
    assert "history_incomplete" in p.reason
    assert "recent_closed_pnl" in p.reason


@pytest.mark.unit
def test_missing_consec_losses_caps_at_normal() -> None:
    now = _now()
    full = build_ea_state(
        equity=10000.0, balance=10000.0, dd_pct=0.0, spread_pts=20,
        consec_losses=0, recent_closed_pnl=10.0, recent_sample_count=20,
    )
    assert full is not None
    ea = EAState(
        equity=full.equity, balance=full.balance, dd_pct=full.dd_pct,
        free_margin=full.free_margin, margin_level=full.margin_level,
        open_lots=full.open_lots, open_positions=full.open_positions,
        floating_pnl=full.floating_pnl, spread_pts=full.spread_pts,
        consec_losses=None,                         # ← dropped
        recent_closed_pnl=full.recent_closed_pnl,
        recent_sample_count=full.recent_sample_count,
    )
    p = derive_envelope_params(_state(ea_state=ea, now=now), prev_envelope=None)
    assert p.risk_tier == "normal"
    assert p.lot_factor == pytest.approx(1.0)
    assert "history_incomplete" in p.reason
    assert "consec_losses" in p.reason


@pytest.mark.unit
def test_missing_history_does_not_step_down() -> None:
    """history_incomplete must NOT trigger step-down (no evidence of losses)."""
    now = _now()
    full = build_ea_state(
        equity=10000.0, balance=10000.0, dd_pct=0.0, spread_pts=20,
        consec_losses=0, recent_closed_pnl=10.0, recent_sample_count=20,
    )
    assert full is not None
    ea = EAState(
        equity=full.equity, balance=full.balance, dd_pct=full.dd_pct,
        free_margin=full.free_margin, margin_level=full.margin_level,
        open_lots=full.open_lots, open_positions=full.open_positions,
        floating_pnl=full.floating_pnl, spread_pts=full.spread_pts,
        consec_losses=None,
        recent_closed_pnl=None,
        recent_sample_count=full.recent_sample_count,
    )
    # confidence below aggressive floor — anyway normal, just assert
    # no step-down spilled out.
    p = derive_envelope_params(
        _state(ea_state=ea, now=now, confidence=0.65),
        prev_envelope=None,
    )
    assert p.mode == "hedgerock"  # NOT downgraded to observe
    assert p.risk_tier == "normal"
    assert "step-down" not in p.reason


# ---------------------------------------------------------------------------
# /signal end-to-end — query params control completeness
# ---------------------------------------------------------------------------


def _build_app_for_range_aggressive():
    """App with classifier features wired so confidence=0.85 in v2."""
    # Classifier vol=0.55 + h4_trend=4 + hh-ll=4 → trend_up rule fires.
    # We want range with high confidence — use vol=0.45, hh-ll=0,
    # h4_trend=1 → range#2 fires at 0.65, which is < aggressive_floor.
    # Bumping confidence to 0.85 requires a different fixture; the simplest
    # way is to drive a 'range' assessment that the classifier won't actually
    # produce. The /signal wiring uses the real classifier, so for end-to-end
    # we instead choose a fixture that classifies with confidence ≥ 0.80.
    #
    # Trick: vol_rank<0.30 → range#1 fires at 0.80. That's the cleanest
    # way to land 'range' with conf ≥ aggressive floor.
    feats = MarketFeatures(
        volatility_rank=0.20, hh_count=3, ll_count=3,
        h4_trend_bars=1, regime="CONSOLIDATION",
    )
    market = _StubProvider(feats)
    return create_app(
        market, enable_debate=False, enable_rule_engine=True,
    )


@pytest.mark.unit
def test_signal_full_history_triple_aggressive() -> None:
    """User spec scenario 3 — consec_losses=0, recent_pnl=0, sample=5 → aggressive."""
    app = _build_app_for_range_aggressive()
    with TestClient(app) as client:
        body = client.get(
            "/signal",
            params={
                "symbol": "XAUUSD",
                "equity": 10000.0, "balance": 10000.0,
                "dd_pct": 0.0, "spread_pts": 20,
                "consec_losses": 0,
                "recent_closed_pnl": 0.0,
                "recent_sample_count": 5,
            },
        ).json()
        assert body["mode"] == "hedgerock"
        assert body["risk_tier"] == "aggressive"
        assert body["lot_factor"] == pytest.approx(1.5)


@pytest.mark.unit
def test_signal_sample_20_with_recent_pnl_omitted_caps_at_normal() -> None:
    """User spec scenario 1 — sample=20 but recent_closed_pnl omitted → normal."""
    app = _build_app_for_range_aggressive()
    with TestClient(app) as client:
        body = client.get(
            "/signal",
            params={
                "symbol": "XAUUSD",
                "equity": 10000.0, "balance": 10000.0,
                "dd_pct": 0.0, "spread_pts": 20,
                "consec_losses": 0,
                # recent_closed_pnl deliberately omitted
                "recent_sample_count": 20,
            },
        ).json()
        assert body["mode"] == "hedgerock"
        assert body["risk_tier"] == "normal"
        assert body["lot_factor"] == pytest.approx(1.0)
        assert "history_incomplete" in body["reason"]
        assert "recent_closed_pnl" in body["reason"]


@pytest.mark.unit
def test_signal_recent_pnl_with_consec_losses_omitted_caps_at_normal() -> None:
    """User spec scenario 2 — recent_pnl=10/sample=20 but consec_losses omitted → normal."""
    app = _build_app_for_range_aggressive()
    with TestClient(app) as client:
        body = client.get(
            "/signal",
            params={
                "symbol": "XAUUSD",
                "equity": 10000.0, "balance": 10000.0,
                "dd_pct": 0.0, "spread_pts": 20,
                # consec_losses deliberately omitted
                "recent_closed_pnl": 10.0,
                "recent_sample_count": 20,
            },
        ).json()
        assert body["mode"] == "hedgerock"
        assert body["risk_tier"] == "normal"
        assert body["lot_factor"] == pytest.approx(1.0)
        assert "history_incomplete" in body["reason"]
        assert "consec_losses" in body["reason"]


@pytest.mark.unit
def test_signal_full_triple_with_positive_pnl_aggressive_end_to_end() -> None:
    """Mirror of scenario 3 but with the production-likely sample size."""
    app = _build_app_for_range_aggressive()
    with TestClient(app) as client:
        body = client.get(
            "/signal",
            params={
                "symbol": "XAUUSD",
                "equity": 10000.0, "balance": 10000.0,
                "dd_pct": 0.0, "spread_pts": 20,
                "consec_losses": 0,
                "recent_closed_pnl": 25.0,
                "recent_sample_count": 20,
            },
        ).json()
        assert body["mode"] == "hedgerock"
        assert body["risk_tier"] == "aggressive"

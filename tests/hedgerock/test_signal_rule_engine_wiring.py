"""Phase C wiring tests — /signal end-to-end through rule_engine.

Pinned behaviour:
    1. With ``enable_rule_engine=False`` (the default), envelope keeps
       schema-default dynamic fields. This protects the 600+ existing
       unit tests that don't expect rule_engine to fire.
    2. With ``enable_rule_engine=True``, /signal:
        - calls classify_regime_v2 over MarketFeatures
        - aggregates EAStateStore via aggregate_from_stores
        - feeds the MarketState into rule_engine.derive_envelope_params
        - passes the resulting kwargs into build_envelope
    3. EA query params (equity, dd_pct, ...) → EAStateStore →
       rule_engine response visible in the next /signal envelope.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from smc.hedgerock.decision_server import (
    EAStateStore,
    MarketFeatures,
    create_app,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _StubProvider:
    def __init__(self, features: MarketFeatures) -> None:
        self._f = features

    def get_features(self, symbol: str) -> MarketFeatures:
        return self._f


def _features_for_range() -> MarketFeatures:
    """Features that classify_regime_v2 maps to ``range`` with high confidence.

    vol_rank=0.45, hh_count=3, ll_count=3, h4_trend_bars=1
        - range#1 (vol<0.30) NO
        - range#2 (|hh-ll|<=1 AND vol<0.65) → conf=0.65 ✓
        - trend (h4_trend<3) NO
        - breakout (vol<0.85) NO
    """
    return MarketFeatures(
        volatility_rank=0.45, hh_count=3, ll_count=3,
        h4_trend_bars=1, regime="CONSOLIDATION",
    )


def _features_for_trend_up() -> MarketFeatures:
    """Features that classify to ``trend_up`` (HedgeRock disabled)."""
    return MarketFeatures(
        volatility_rank=0.55, hh_count=8, ll_count=2,
        h4_trend_bars=5, regime="TREND_UP",
    )


# ---------------------------------------------------------------------------
# enable_rule_engine=False (default) — envelope unchanged
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_default_off_envelope_keeps_schema_defaults() -> None:
    market = _StubProvider(_features_for_range())
    app = create_app(market, enable_debate=False)  # default enable_rule_engine=False
    with TestClient(app) as client:
        body = client.get("/signal", params={"symbol": "XAUUSD"}).json()
        # Schema defaults — rule_engine NOT invoked.
        assert body["mode"] == "observe"
        assert body["hedgerock_enabled"] is False
        assert body["lot_factor"] == 1.0          # schema default
        assert body["max_next_lot"] == 0.05       # schema default


@pytest.mark.unit
def test_status_reports_rule_engine_off_by_default() -> None:
    market = _StubProvider(_features_for_range())
    app = create_app(market, enable_debate=False)
    with TestClient(app) as client:
        body = client.get("/status").json()
        assert body["rule_engine_enabled"] is False


# ---------------------------------------------------------------------------
# enable_rule_engine=True — production wiring
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_status_reports_rule_engine_on_when_enabled() -> None:
    market = _StubProvider(_features_for_range())
    app = create_app(market, enable_debate=False, enable_rule_engine=True)
    with TestClient(app) as client:
        body = client.get("/status").json()
        assert body["rule_engine_enabled"] is True


@pytest.mark.unit
def test_range_features_drive_envelope_to_hedgerock_mode() -> None:
    """Range regime + fresh EAState + classify_regime_v2 conf=0.65 →
    mode=hedgerock, risk_tier=normal (conf < aggressive_floor 0.80).

    EAState is required (Phase C-hotfix #2) — without it the wiring
    falls back to observe.
    """
    market = _StubProvider(_features_for_range())
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
        assert body["hedgerock_enabled"] is True
        assert body["risk_tier"] == "normal"
        assert body["lot_factor"] == pytest.approx(1.0)
        assert body["regime"] == "range"


@pytest.mark.unit
def test_trend_up_features_force_observe_mode() -> None:
    """trend_up regime → rule_engine forces mode=observe, lot_factor=0."""
    market = _StubProvider(_features_for_trend_up())
    app = create_app(market, enable_debate=False, enable_rule_engine=True)
    with TestClient(app) as client:
        body = client.get("/signal", params={"symbol": "XAUUSD"}).json()
        assert body["mode"] == "observe"
        assert body["hedgerock_enabled"] is False
        assert body["lot_factor"] == 0.0
        assert body["regime"] == "trend_up"


@pytest.mark.unit
def test_ea_dd_step_down_visible_in_envelope() -> None:
    """Send equity + dd_pct via query → EAStateStore records → next
    /signal call's rule_engine sees DD ≥ 2% → step DOWN risk_tier."""
    market = _StubProvider(_features_for_range())
    ea_store = EAStateStore()
    app = create_app(
        market, enable_debate=False, enable_rule_engine=True,
        ea_state_store=ea_store,
    )
    with TestClient(app) as client:
        body = client.get(
            "/signal",
            params={
                "symbol": "XAUUSD",
                "equity": 9700.0, "balance": 10000.0, "dd_pct": 0.03,
                "spread_pts": 20,
                "consec_losses": 0, "recent_closed_pnl": 0.0, "recent_sample_count": 20,
            },
        ).json()
        # DD 3% ≥ 2% threshold → step-down → risk_tier=observe + smaller lot.
        assert body["mode"] == "hedgerock"  # range regime + good conf still trades
        assert body["risk_tier"] == "observe"
        assert body["lot_factor"] == pytest.approx(0.5)
        assert "step-down" in body["reason"]


@pytest.mark.unit
def test_ea_severe_dd_halts_envelope() -> None:
    """6% DD triggers halt regardless of regime."""
    market = _StubProvider(_features_for_range())
    ea_store = EAStateStore()
    app = create_app(
        market, enable_debate=False, enable_rule_engine=True,
        ea_state_store=ea_store,
    )
    with TestClient(app) as client:
        body = client.get(
            "/signal",
            params={
                "symbol": "XAUUSD",
                "equity": 9400.0, "balance": 10000.0, "dd_pct": 0.06,
            },
        ).json()
        assert body["mode"] == "halt"
        assert body["lot_factor"] == 0.0
        assert body["cooldown_until"] is not None


@pytest.mark.unit
def test_ea_consec_losses_5_triggers_cooldown() -> None:
    """consec_losses=5 → cooldown."""
    market = _StubProvider(_features_for_range())
    ea_store = EAStateStore()
    app = create_app(
        market, enable_debate=False, enable_rule_engine=True,
        ea_state_store=ea_store,
    )
    with TestClient(app) as client:
        body = client.get(
            "/signal",
            params={
                "symbol": "XAUUSD",
                "equity": 10000.0, "balance": 10000.0,
                "consec_losses": 6, "recent_sample_count": 20,
            },
        ).json()
        assert body["mode"] == "observe"
        assert body["cooldown_until"] is not None
        assert "consec_losses" in body["reason"]


@pytest.mark.unit
def test_minus_one_sentinel_does_not_trigger_step_down() -> None:
    """Phase A-closeout #1 mini-patch sentinel: history unavailable
    must NOT be treated as 'no losses'. After fan-out the rule_engine
    should see consec_losses=None and not step down."""
    market = _StubProvider(_features_for_range())
    ea_store = EAStateStore()
    app = create_app(
        market, enable_debate=False, enable_rule_engine=True,
        ea_state_store=ea_store,
    )
    with TestClient(app) as client:
        body = client.get(
            "/signal",
            params={
                "symbol": "XAUUSD",
                "equity": 10000.0, "balance": 10000.0, "dd_pct": 0.0,
                "spread_pts": 20,
                "consec_losses": -1, "recent_closed_pnl": -1,
                "recent_sample_count": -1,
            },
        ).json()
        # Sentinel → None → no step-down trigger but step-up blocked
        # (Phase C-hotfix-2 #1) → cap at normal tier.
        assert body["mode"] == "hedgerock"
        assert body["risk_tier"] == "normal"
        assert body["lot_factor"] == pytest.approx(1.0)


@pytest.mark.unit
def test_envelope_carries_rule_engine_reason_string() -> None:
    """With EAState present the reason mentions the regime that fired."""
    market = _StubProvider(_features_for_range())
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
        assert body["reason"]  # non-empty
        assert "range" in body["reason"]

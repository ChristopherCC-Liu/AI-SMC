"""Phase C-hotfix-2 targeted tests.

Patch 1 — history_unavailable (recent_sample_count is None) blocks step-up
Patch 2 — incomplete_ea_state guard (dd_pct AND spread_pts required)
Patch 3 — v2 transition_lock from PrevRegimeV2Store
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from smc.hedgerock.decision_server import (
    EAStateStore,
    MarketFeatures,
    PrevRegimeV2Store,
    create_app,
)
from smc.hedgerock.ea_state import build_ea_state
from smc.hedgerock.market_state import aggregate_market_state
from smc.hedgerock.regime_classifier_v2 import RegimeAssessmentV2
from smc.hedgerock.rule_engine import derive_envelope_params
from smc.hedgerock.transition_lock import (
    REGIME_V2_DISTANCE,
    compute_lock_seconds_v2,
    compute_lock_until_v2,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now() -> datetime:
    return datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)


def _features_range() -> MarketFeatures:
    return MarketFeatures(
        volatility_rank=0.45, hh_count=3, ll_count=3,
        h4_trend_bars=1, regime="CONSOLIDATION",
    )


def _features_legacy_consolidation_v2_range() -> MarketFeatures:
    """Legacy regime stays CONSOLIDATION (legacy v2 mapping = 'range').

    Used for transition lock test where the LEGACY regime is unchanged
    poll-to-poll but the v2 regime transitions.
    """
    return MarketFeatures(
        volatility_rank=0.45, hh_count=3, ll_count=3,
        h4_trend_bars=1, regime="CONSOLIDATION",
    )


def _features_clean_trend_up() -> MarketFeatures:
    return MarketFeatures(
        volatility_rank=0.55, hh_count=8, ll_count=2,
        h4_trend_bars=5, regime="TREND_UP",
    )


class _StubProvider:
    def __init__(self, features: MarketFeatures) -> None:
        self._f = features
        self._call = 0

    def get_features(self, symbol: str) -> MarketFeatures:
        self._call += 1
        return self._f


class _SwitchableProvider:
    """Returns one MarketFeatures up to ``switch_after`` calls, then another."""

    def __init__(self, first: MarketFeatures, then: MarketFeatures, *, switch_after: int = 1) -> None:
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
    return aggregate_market_state(
        symbol="XAUUSD", now=now,
        features=_features_range(),
        regime_assessment=assessment,
        ea_state=ea_state,
        ea_state_recorded_at=ea_recorded_at,
    )


# ---------------------------------------------------------------------------
# Patch 1 — history_unavailable blocks aggressive step-up
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_recent_sample_count_minus_one_caps_at_normal_tier() -> None:
    """Phase C-hotfix-2 #1: recent_sample_count=-1 sentinel
    (→ recent_n=None) → block step-up → risk_tier=normal, lot_factor=1.0.
    """
    now = _now()
    ea = build_ea_state(
        equity=10000.0, balance=10000.0, dd_pct=0.0, spread_pts=20,
        consec_losses=-1, recent_closed_pnl=-1, recent_sample_count=-1,
    )
    state = _state(regime_v2="range", confidence=0.85,
                   ea_state=ea, ea_recorded_at=now - timedelta(seconds=5), now=now)
    p = derive_envelope_params(state, prev_envelope=None)
    assert p.mode == "hedgerock"
    assert p.risk_tier == "normal"
    assert p.lot_factor == pytest.approx(1.0)
    assert "history_unavailable" in p.reason


@pytest.mark.unit
def test_history_present_with_full_sample_allows_aggressive_step_up() -> None:
    """Sanity: with sample present + non-negative pnl, step-up is allowed."""
    now = _now()
    ea = build_ea_state(
        equity=10000.0, balance=10000.0, dd_pct=0.0, spread_pts=20,
        consec_losses=0, recent_closed_pnl=15.0, recent_sample_count=20,
    )
    state = _state(regime_v2="range", confidence=0.85,
                   ea_state=ea, ea_recorded_at=now - timedelta(seconds=5), now=now)
    p = derive_envelope_params(state, prev_envelope=None)
    assert p.risk_tier == "aggressive"
    assert p.lot_factor == pytest.approx(1.5)


@pytest.mark.unit
def test_history_unavailable_does_not_step_down() -> None:
    """history_unavailable must NOT trigger step-down (no evidence of losses)."""
    now = _now()
    ea = build_ea_state(
        equity=10000.0, balance=10000.0, dd_pct=0.0, spread_pts=20,
        consec_losses=-1, recent_closed_pnl=-1, recent_sample_count=-1,
    )
    state = _state(regime_v2="range", confidence=0.65,
                   ea_state=ea, ea_recorded_at=now - timedelta(seconds=5), now=now)
    p = derive_envelope_params(state, prev_envelope=None)
    # confidence < aggressive_floor anyway → normal tier; step-down NOT triggered.
    assert p.risk_tier == "normal"
    assert p.lot_factor == pytest.approx(1.0)
    assert "step-down" not in p.reason


# ---------------------------------------------------------------------------
# Patch 2 — incomplete_ea_state guard
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_only_equity_balance_blocks_hedgerock_via_incomplete_ea_state() -> None:
    """User spec: 只传 equity/balance 不得 hedgerock"""
    market = _StubProvider(_features_range())
    app = create_app(market, enable_debate=False, enable_rule_engine=True)
    with TestClient(app) as client:
        body = client.get(
            "/signal",
            params={"symbol": "XAUUSD", "equity": 10000.0, "balance": 10000.0},
        ).json()
        assert body["mode"] == "observe"
        assert body["lot_factor"] == 0.0
        assert "incomplete_ea_state" in body["reason"]
        assert "spread_pts" in body["reason"] or "dd_pct" in body["reason"]


@pytest.mark.unit
def test_only_dd_pct_missing_blocks_hedgerock() -> None:
    """spread_pts present but dd_pct missing → still observe."""
    market = _StubProvider(_features_range())
    app = create_app(market, enable_debate=False, enable_rule_engine=True)
    with TestClient(app) as client:
        body = client.get(
            "/signal",
            params={
                "symbol": "XAUUSD",
                "equity": 10000.0, "balance": 10000.0,
                "spread_pts": 20,
            },
        ).json()
        assert body["mode"] == "observe"
        assert "incomplete_ea_state" in body["reason"]
        assert "dd_pct" in body["reason"]


@pytest.mark.unit
def test_only_spread_pts_missing_blocks_hedgerock() -> None:
    """dd_pct present but spread_pts missing → still observe."""
    market = _StubProvider(_features_range())
    app = create_app(market, enable_debate=False, enable_rule_engine=True)
    with TestClient(app) as client:
        body = client.get(
            "/signal",
            params={
                "symbol": "XAUUSD",
                "equity": 10000.0, "balance": 10000.0,
                "dd_pct": 0.0,
            },
        ).json()
        assert body["mode"] == "observe"
        assert "incomplete_ea_state" in body["reason"]
        assert "spread_pts" in body["reason"]


@pytest.mark.unit
def test_full_risk_snapshot_allows_hedgerock() -> None:
    """User spec: 传 equity/balance/dd_pct/spread_pts 才能 hedgerock"""
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
        assert body["lot_factor"] == pytest.approx(1.0)


@pytest.mark.unit
def test_history_can_be_unavailable_with_complete_risk_snapshot() -> None:
    """incomplete_ea_state guard does NOT require history fields —
    they may be -1/sentinel/missing. Only dd_pct + spread_pts are
    the minimum mandatory risk-snapshot fields."""
    market = _StubProvider(_features_range())
    app = create_app(market, enable_debate=False, enable_rule_engine=True)
    with TestClient(app) as client:
        body = client.get(
            "/signal",
            params={
                "symbol": "XAUUSD",
                "equity": 10000.0, "balance": 10000.0,
                "dd_pct": 0.0, "spread_pts": 20,
                "consec_losses": -1, "recent_closed_pnl": -1,
                "recent_sample_count": -1,
            },
        ).json()
        # History unavailable but risk-snapshot complete → hedgerock at normal.
        assert body["mode"] == "hedgerock"
        assert body["risk_tier"] == "normal"


# ---------------------------------------------------------------------------
# Patch 3 — v2 transition_lock (compute_lock_until_v2 + PrevRegimeV2Store)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_compute_lock_seconds_v2_same_regime_returns_zero() -> None:
    assert compute_lock_seconds_v2("range", "range") == 0
    assert compute_lock_seconds_v2("trend_up", "trend_up") == 0


@pytest.mark.unit
def test_compute_lock_seconds_v2_first_call_returns_zero() -> None:
    assert compute_lock_seconds_v2(None, "range") == 0


@pytest.mark.unit
def test_compute_lock_seconds_v2_extreme_reversal_max_lock() -> None:
    # trend_up → trend_down = distance 3 = 7200 s
    assert compute_lock_seconds_v2("trend_up", "trend_down") == 7200
    # crisis → range = distance 3 = 7200 s
    assert compute_lock_seconds_v2("crisis", "range") == 7200


@pytest.mark.unit
def test_compute_lock_until_v2_returns_none_for_zero_distance() -> None:
    now = _now()
    assert compute_lock_until_v2("range", "range", now) is None
    assert compute_lock_until_v2(None, "range", now) is None


@pytest.mark.unit
def test_compute_lock_until_v2_rejects_naive_clock() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        compute_lock_until_v2("range", "trend_up", datetime(2026, 5, 1))


@pytest.mark.unit
def test_v2_distance_matrix_covers_all_pairs() -> None:
    """Sanity: every v2-regime pair must have a defined distance,
    otherwise compute_lock_seconds_v2 raises KeyError at runtime."""
    from smc.hedgerock.schemas import REGIMES_V2
    for a in REGIMES_V2:
        for b in REGIMES_V2:
            assert (a, b) in REGIME_V2_DISTANCE, f"missing distance for ({a}, {b})"


@pytest.mark.unit
def test_prev_regime_v2_store_canonicalizes_symbol() -> None:
    s = PrevRegimeV2Store()
    s.set("xauusd", "range")
    assert s.get("XAUUSD") == "range"
    assert s.get("XaUuSd") == "range"


# ---------------------------------------------------------------------------
# Patch 3 — wiring: v2 transition lock fires when legacy regime unchanged
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_v2_transition_lock_fires_when_legacy_regime_unchanged() -> None:
    """Phase C-hotfix-2 #3: features.regime stays CONSOLIDATION (legacy)
    across two polls. classify_regime_v2 returns 'range' on poll 1
    (balanced HH/LL features) and 'trend_up' on poll 2 (HH dominance
    features). transition_lock_until_ts must be set on poll 2 because
    the v2 regime changed range→trend_up, even though legacy is identical.
    """
    range_features = _features_legacy_consolidation_v2_range()  # legacy=CONSOLIDATION, v2=range
    trend_up_features = _features_clean_trend_up()              # legacy=TREND_UP, v2=trend_up

    # Force legacy=CONSOLIDATION on BOTH calls so legacy compute_lock_until
    # would not fire — but vary v2 regime by changing other fields.
    poll1 = MarketFeatures(
        volatility_rank=0.45, hh_count=3, ll_count=3,
        h4_trend_bars=1, regime="CONSOLIDATION",  # v2 → range
    )
    poll2 = MarketFeatures(
        volatility_rank=0.55, hh_count=8, ll_count=2,
        h4_trend_bars=5, regime="CONSOLIDATION",  # legacy stays CONSOLIDATION!
                                                  # v2 → trend_up (HH-dominant features)
    )

    market = _SwitchableProvider(poll1, poll2, switch_after=1)
    app = create_app(market, enable_debate=False, enable_rule_engine=True)
    with TestClient(app) as client:
        # Poll 1 — establishes prev_regime_v2 = "range"
        b1 = client.get(
            "/signal",
            params={
                "symbol": "XAUUSD",
                "equity": 10000.0, "balance": 10000.0,
                "dd_pct": 0.0, "spread_pts": 20,
            },
        ).json()
        assert b1["regime"] == "range"
        # No prior v2 → no v2 lock yet (None).
        assert b1["transition_lock_until_ts"] is None

        # Poll 2 — v2 regime flips to trend_up while legacy stays CONSOLIDATION.
        b2 = client.get(
            "/signal",
            params={
                "symbol": "XAUUSD",
                "equity": 10000.0, "balance": 10000.0,
                "dd_pct": 0.0, "spread_pts": 20,
            },
        ).json()
        assert b2["regime"] == "trend_up"
        # Legacy mapping (CONSOLIDATION → "range", same as poll 1) wouldn't fire.
        # v2 mapping (range → trend_up, distance=2) fires → 3600 s lock.
        assert b2["transition_lock_until_ts"] is not None


@pytest.mark.unit
def test_status_exposes_tracked_v2_regimes() -> None:
    market = _StubProvider(_features_range())
    app = create_app(market, enable_debate=False, enable_rule_engine=True)
    with TestClient(app) as client:
        # Trigger a poll so the v2 store gets populated.
        client.get(
            "/signal",
            params={
                "symbol": "XAUUSD",
                "equity": 10000.0, "balance": 10000.0,
                "dd_pct": 0.0, "spread_pts": 20,
            },
        )
        body = client.get("/status").json()
        assert "tracked_v2_regimes" in body
        assert body["tracked_v2_regimes"]["XAUUSD"] == "range"


@pytest.mark.unit
def test_status_tracked_v2_regimes_empty_before_first_poll() -> None:
    market = _StubProvider(_features_range())
    app = create_app(market, enable_debate=False, enable_rule_engine=True)
    with TestClient(app) as client:
        body = client.get("/status").json()
        assert body["tracked_v2_regimes"] == {}

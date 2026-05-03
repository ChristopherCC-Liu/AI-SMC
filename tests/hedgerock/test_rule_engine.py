"""Phase C — rule_engine.derive_envelope_params tests.

Each test pins down one branch of the decision tree. The contract:

    - Numeric outputs are within schema bounds (build_envelope's
      Pydantic validator catches violations; we assert on the
      DynamicParams field values directly).
    - Reason string contains enough breadcrumb to debug from logs.
    - mode/hedgerock_enabled/risk_tier are consistent (never
      mode=hedgerock + hedgerock_enabled=False, etc).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from smc.hedgerock.decision_server import MarketFeatures
from smc.hedgerock.ea_state import build_ea_state
from smc.hedgerock.market_state import aggregate_market_state
from smc.hedgerock.regime_classifier_v2 import RegimeAssessmentV2, classify_regime_v2
from smc.hedgerock.rule_engine import (
    DynamicParams,
    derive_envelope_params,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def now() -> datetime:
    return datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)


def _features(regime: str = "TREND_UP", *, vol_rank: float = 0.5) -> MarketFeatures:
    return MarketFeatures(
        volatility_rank=vol_rank, hh_count=4, ll_count=2,
        h4_trend_bars=3, regime=regime,
    )


def _state(
    *,
    now: datetime,
    regime_v2: str,
    confidence: float = 0.85,
    ea_state=None,
    ea_recorded_at: datetime | None = None,
    stale: bool = False,
):
    """Build a MarketState by hand (don't go through classify_regime_v2 —
    we want to drive specific regimes for branch coverage)."""
    assessment = RegimeAssessmentV2(
        regime=regime_v2,  # type: ignore[arg-type]
        confidence=confidence,
        reason=f"test-fixture regime={regime_v2}",
        rule_votes=(),
    )
    if stale:
        # Force the stale path by giving an old recorded_at.
        ea_recorded_at = now - timedelta(seconds=600)
    return aggregate_market_state(
        symbol="XAUUSD",
        now=now,
        features=_features(),
        regime_assessment=assessment,
        ea_state=ea_state,
        ea_state_recorded_at=ea_recorded_at,
    )


# ---------------------------------------------------------------------------
# Decision-tree branches
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_stale_ea_state_forces_observe(now: datetime) -> None:
    ea = build_ea_state(equity=10000.0)
    state = _state(now=now, regime_v2="range", ea_state=ea, stale=True)
    p = derive_envelope_params(state, prev_envelope=None)

    assert p.mode == "observe"
    assert p.hedgerock_enabled is False
    assert p.lot_factor == 0.0
    assert "stale" in p.reason


@pytest.mark.unit
def test_crisis_regime_halts_with_cooldown(now: datetime) -> None:
    state = _state(now=now, regime_v2="crisis")
    p = derive_envelope_params(state, prev_envelope=None)

    assert p.mode == "halt"
    assert p.hedgerock_enabled is False
    assert p.lot_factor == 0.0
    assert p.max_orders_buy == 0
    assert p.max_orders_sell == 0
    assert p.cooldown_until is not None
    assert p.cooldown_until > now
    assert "crisis" in p.reason


@pytest.mark.unit
def test_news_regime_observes_with_cooldown(now: datetime) -> None:
    state = _state(now=now, regime_v2="news")
    p = derive_envelope_params(state, prev_envelope=None)

    assert p.mode == "observe"
    assert p.lot_factor == 0.0
    assert p.cooldown_until is not None and p.cooldown_until > now
    assert "news" in p.reason


@pytest.mark.unit
def test_severe_dd_halts_even_in_range_regime(now: datetime) -> None:
    """5%+ DD is a hard halt regardless of how good the regime looks."""
    ea = build_ea_state(equity=9400.0, balance=10000.0, dd_pct=0.06)
    state = _state(
        now=now, regime_v2="range",
        ea_state=ea, ea_recorded_at=now - timedelta(seconds=5),
    )
    p = derive_envelope_params(state, prev_envelope=None)

    assert p.mode == "halt"
    assert p.cooldown_until is not None
    assert "DD" in p.reason


@pytest.mark.unit
def test_spread_anomaly_triggers_short_cooldown(now: datetime) -> None:
    ea = build_ea_state(equity=10000.0, balance=10000.0, spread_pts=120)
    state = _state(
        now=now, regime_v2="range",
        ea_state=ea, ea_recorded_at=now - timedelta(seconds=5),
    )
    p = derive_envelope_params(state, prev_envelope=None)

    assert p.mode == "observe"
    assert p.cooldown_until is not None
    assert (p.cooldown_until - now) <= timedelta(minutes=10)
    assert "spread" in p.reason


@pytest.mark.unit
def test_consec_losses_cooldown_threshold(now: datetime) -> None:
    """consec_losses ≥ 5 → cooldown."""
    ea = build_ea_state(equity=10000.0, balance=10000.0,
                        consec_losses=6, recent_sample_count=20)
    state = _state(
        now=now, regime_v2="range",
        ea_state=ea, ea_recorded_at=now - timedelta(seconds=5),
    )
    p = derive_envelope_params(state, prev_envelope=None)

    assert p.mode == "observe"
    assert p.cooldown_until is not None
    assert "consec_losses" in p.reason


@pytest.mark.unit
def test_trend_regime_disables_hedgerock(now: datetime) -> None:
    """HedgeRock isn't a trend strategy — observe in trend_up/down."""
    state = _state(now=now, regime_v2="trend_up")
    p = derive_envelope_params(state, prev_envelope=None)
    assert p.mode == "observe"
    assert p.hedgerock_enabled is False


@pytest.mark.unit
def test_unknown_regime_observes(now: datetime) -> None:
    state = _state(now=now, regime_v2="unknown")
    p = derive_envelope_params(state, prev_envelope=None)
    assert p.mode == "observe"


@pytest.mark.unit
def test_range_low_confidence_observes(now: datetime) -> None:
    """Range regime with confidence below the floor is still observe-only.

    Provide a fresh EAState — otherwise the no_ea_state guard
    (Phase C-hotfix #2) would short-circuit the test before reaching
    the confidence floor branch.
    """
    ea = build_ea_state(equity=10000.0, balance=10000.0, dd_pct=0.0, spread_pts=20)
    state = _state(now=now, regime_v2="range", confidence=0.50,
                   ea_state=ea, ea_recorded_at=now - timedelta(seconds=5))
    p = derive_envelope_params(state, prev_envelope=None)
    assert p.mode == "observe"
    assert "confidence" in p.reason


@pytest.mark.unit
def test_range_normal_confidence_runs_hedgerock_at_normal_tier(
    now: datetime,
) -> None:
    """Range + 0.55-0.79 confidence → mode=hedgerock + risk_tier=normal."""
    ea = build_ea_state(equity=10000.0, balance=10000.0, dd_pct=0.0, spread_pts=20,
                        consec_losses=0, recent_closed_pnl=0.0, recent_sample_count=20)
    state = _state(
        now=now, regime_v2="range", confidence=0.65,
        ea_state=ea, ea_recorded_at=now - timedelta(seconds=5),
    )
    p = derive_envelope_params(state, prev_envelope=None)

    assert p.mode == "hedgerock"
    assert p.hedgerock_enabled is True
    assert p.risk_tier == "normal"
    assert p.lot_factor == pytest.approx(1.0)
    assert p.cooldown_until is None


@pytest.mark.unit
def test_range_high_confidence_steps_up_to_aggressive(
    now: datetime,
) -> None:
    ea = build_ea_state(equity=10000.0, balance=10000.0, dd_pct=0.005, spread_pts=20,
                        consec_losses=0, recent_closed_pnl=0.0, recent_sample_count=20)
    state = _state(
        now=now, regime_v2="range", confidence=0.85,
        ea_state=ea, ea_recorded_at=now - timedelta(seconds=5),
    )
    p = derive_envelope_params(state, prev_envelope=None)

    assert p.mode == "hedgerock"
    assert p.risk_tier == "aggressive"
    assert p.lot_factor == pytest.approx(1.5)
    assert p.max_next_lot == pytest.approx(0.10)
    assert p.max_orders_buy == 3
    assert p.max_orders_sell == 3


@pytest.mark.unit
def test_range_with_dd_step_down_to_observe_tier(now: datetime) -> None:
    """DD ≥ 2% steps down sizing even in a clean range regime."""
    ea = build_ea_state(equity=9750.0, balance=10000.0, dd_pct=0.025, spread_pts=20,
                        consec_losses=0, recent_closed_pnl=0.0, recent_sample_count=20)
    state = _state(
        now=now, regime_v2="range", confidence=0.85,
        ea_state=ea, ea_recorded_at=now - timedelta(seconds=5),
    )
    p = derive_envelope_params(state, prev_envelope=None)

    assert p.mode == "hedgerock"
    assert p.hedgerock_enabled is True  # we still trade, just smaller
    assert p.risk_tier == "observe"
    assert p.lot_factor == pytest.approx(0.5)
    assert p.max_next_lot == pytest.approx(0.03)
    assert "step-down" in p.reason


@pytest.mark.unit
def test_consec_losses_3_steps_down_but_does_not_cooldown(now: datetime) -> None:
    """3 consec losses = step-down, not cooldown."""
    ea = build_ea_state(equity=10000.0, balance=10000.0, dd_pct=0.0, spread_pts=20,
                        consec_losses=3, recent_closed_pnl=0.0, recent_sample_count=20)
    state = _state(
        now=now, regime_v2="range", confidence=0.85,
        ea_state=ea, ea_recorded_at=now - timedelta(seconds=5),
    )
    p = derive_envelope_params(state, prev_envelope=None)

    assert p.mode == "hedgerock"
    assert p.cooldown_until is None
    assert p.lot_factor == pytest.approx(0.5)
    assert "step-down" in p.reason


@pytest.mark.unit
def test_no_ea_state_forces_observe_even_in_clean_range(now: datetime) -> None:
    """Phase C-hotfix #2: refuse to enter HedgeRock without EA runtime state.

    Even a perfect range + high-confidence regime must NOT step into
    HedgeRock when ea_state is None — the rule_engine has no DD /
    spread / loss-streak signal to react to, so the safe default is
    observe.
    """
    state = _state(now=now, regime_v2="range", confidence=0.85,
                   ea_state=None, ea_recorded_at=None)
    p = derive_envelope_params(state, prev_envelope=None)

    assert p.mode == "observe"
    assert p.hedgerock_enabled is False
    assert p.lot_factor == 0.0
    assert "no_ea_state" in p.reason


@pytest.mark.unit
def test_dynamic_params_to_kwargs_matches_build_envelope(now: datetime) -> None:
    """to_kwargs() output must be valid kwargs for build_envelope."""
    from smc.hedgerock.decision_server import build_envelope
    from smc.hedgerock.regime_filters import FilterInputs  # noqa: F401

    state = _state(now=now, regime_v2="range", confidence=0.65,
                   ea_state=build_ea_state(equity=10000.0),
                   ea_recorded_at=now - timedelta(seconds=5))
    p = derive_envelope_params(state, prev_envelope=None)

    # Build a real envelope using the kwargs — Pydantic validates.
    env = build_envelope(
        symbol="XAUUSD",
        features=_features(),
        prev_regime=None,
        now=now,
        enable_debate=False,
        **p.to_kwargs(),
    )
    assert env.mode == p.mode
    assert env.hedgerock_enabled == p.hedgerock_enabled
    assert env.risk_tier == p.risk_tier
    assert env.lot_factor == p.lot_factor
    assert env.max_next_lot == p.max_next_lot
    assert env.takeprofit_points == p.takeprofit_points

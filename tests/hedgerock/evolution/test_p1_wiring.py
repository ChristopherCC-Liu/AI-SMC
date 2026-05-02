"""Tests for P1 wiring — candidate_generator consults
``multi_timeframe_state`` (consensus) and ``adaptive_stops``
(advisory) when issuing recommendations.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from smc.hedgerock.evolution.adaptive_stops import (
    StopRecommendation,
    VolatilityRegime,
)
from smc.hedgerock.evolution.candidate_generator import (
    DECISION_NO_RECOMMENDATION,
    DECISION_RECOMMEND,
    REASON_EXTREME_VOLATILITY,
    REASON_TIMEFRAME_CONSENSUS_INSUFFICIENT,
    generate_candidate_proposals,
)
from smc.hedgerock.evolution.candidate_menu import CANDIDATE_MENU_V0
from smc.hedgerock.evolution.multi_timeframe_state import (
    TimeframeState,
    TradingSession,
    compute_consensus,
)
from smc.hedgerock.evolution.policy_manifest import (
    EvidenceBundle,
    GateStatus,
    PromotionGateResult,
)


def _bundle() -> EvidenceBundle:
    return EvidenceBundle(
        bundle_id="<test>",
        bundle_hash_sha256="0" * 64,
        atlas_report_path="<test>",
        atlas_report_hash_sha256="0" * 64,
        data_availability_report_path="<test>",
        data_availability_report_hash_sha256="0" * 64,
        walk_forward_run_paths=("<test>",),
        year_replication={
            "XAUUSD": {"years_total": 5, "years_passing": 5,
                       "negative_sign_years": ()},
        },
        cross_symbol_count=1,
        halt_event_count=42,
        no_strategy_change=True,
    )


def _g6_for(target: str) -> PromotionGateResult:
    return PromotionGateResult(
        gate_id="G6", status=GateStatus.FAIL,
        reason=f"safety_bound_undefined for {target}",
    )


def _stop(regime: VolatilityRegime) -> StopRecommendation:
    return StopRecommendation(
        vol_regime=regime,
        atr_multiplier=2.0,
        position_scale=1.0,
        parkinson_vol=0.0,
        garman_klass_vol=0.0,
        realized_vol_30=0.0,
        realized_vol_ma_60=0.0,
        sigma_ratio=1.0,
        reasoning="(test)",
        n_bars_observed=120,
        blocking_conditions=(),
    )


# ---------------------------------------------------------------------------
# Timeframe consensus gating
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_d1_accumulating_blocks_every_proposal_with_specific_reason() -> None:
    consensus = compute_consensus(
        d1_state=TimeframeState.ACCUMULATING,
        h4_state=TimeframeState.READY,
        h1_state=TimeframeState.READY,
        m5_state=TimeframeState.READY,
        now=datetime(2026, 5, 2, 15, 0, tzinfo=timezone.utc),
    )
    proposals = generate_candidate_proposals(
        candidate_menu=CANDIDATE_MENU_V0,
        bundle=_bundle(),
        gate_results_per_candidate={},
        blocking_reasons_per_candidate={},
        timeframe_consensus=consensus,
    )
    for p in proposals:
        assert p.decision == DECISION_NO_RECOMMENDATION
        assert p.decision_reason == REASON_TIMEFRAME_CONSENSUS_INSUFFICIENT


@pytest.mark.unit
def test_consensus_score_below_threshold_blocks() -> None:
    consensus = compute_consensus(
        d1_state=TimeframeState.READY,
        h4_state=TimeframeState.ACCUMULATING,
        h1_state=TimeframeState.ACCUMULATING,
        m5_state=TimeframeState.READY,
    )
    assert consensus.can_recommend is False
    proposals = generate_candidate_proposals(
        candidate_menu=CANDIDATE_MENU_V0,
        bundle=_bundle(),
        gate_results_per_candidate={},
        blocking_reasons_per_candidate={},
        timeframe_consensus=consensus,
    )
    for p in proposals:
        assert p.decision_reason == REASON_TIMEFRAME_CONSENSUS_INSUFFICIENT


@pytest.mark.unit
def test_consensus_pass_does_not_short_circuit() -> None:
    consensus = compute_consensus(
        d1_state=TimeframeState.READY,
        h4_state=TimeframeState.READY,
        h1_state=TimeframeState.READY,
        m5_state=TimeframeState.READY,
    )
    assert consensus.can_recommend is True
    proposals = generate_candidate_proposals(
        candidate_menu=CANDIDATE_MENU_V0,
        bundle=_bundle(),
        gate_results_per_candidate={
            "c1-lower-observe-floor-0.50": {
                "G6": _g6_for(
                    "smc.hedgerock.rule_engine._CONFIDENCE_OBSERVE_FLOOR"
                )
            },
        },
        blocking_reasons_per_candidate={},
        timeframe_consensus=consensus,
    )
    by_id = {p.candidate_id: p for p in proposals}
    assert by_id["c1-lower-observe-floor-0.50"].decision == DECISION_RECOMMEND


# ---------------------------------------------------------------------------
# Stop recommendation gating (EXTREME blocks aggressive_cap)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_extreme_vol_blocks_aggressive_floor_only() -> None:
    consensus = compute_consensus(
        d1_state=TimeframeState.READY,
        h4_state=TimeframeState.READY,
        h1_state=TimeframeState.READY,
        m5_state=TimeframeState.READY,
    )
    stop = _stop(VolatilityRegime.EXTREME)
    proposals = generate_candidate_proposals(
        candidate_menu=CANDIDATE_MENU_V0,
        bundle=_bundle(),
        gate_results_per_candidate={
            "c1-lower-observe-floor-0.50": {
                "G6": _g6_for(
                    "smc.hedgerock.rule_engine._CONFIDENCE_OBSERVE_FLOOR"
                )
            },
            "c3-aggressive-floor-0.78": {
                "G6": _g6_for(
                    "smc.hedgerock.rule_engine._CONFIDENCE_AGGRESSIVE_FLOOR"
                )
            },
        },
        blocking_reasons_per_candidate={},
        timeframe_consensus=consensus,
        stop_recommendation=stop,
    )
    by_id = {p.candidate_id: p for p in proposals}

    aggressive = by_id["c3-aggressive-floor-0.78"]
    # Aggressive proposals are blocked under EXTREME vol regardless
    # of the existing exposure-class veto.
    assert aggressive.decision == DECISION_NO_RECOMMENDATION
    assert aggressive.decision_reason == REASON_EXTREME_VOLATILITY

    # Observe-floor proposals (different class) still resolve normally.
    obs = by_id["c1-lower-observe-floor-0.50"]
    assert obs.decision_reason != REASON_EXTREME_VOLATILITY


@pytest.mark.unit
def test_normal_vol_does_not_block_aggressive_class() -> None:
    consensus = compute_consensus(
        d1_state=TimeframeState.READY,
        h4_state=TimeframeState.READY,
        h1_state=TimeframeState.READY,
        m5_state=TimeframeState.READY,
    )
    stop = _stop(VolatilityRegime.NORMAL)
    proposals = generate_candidate_proposals(
        candidate_menu=CANDIDATE_MENU_V0,
        bundle=_bundle(),
        gate_results_per_candidate={
            "c3-aggressive-floor-0.78": {
                "G6": _g6_for(
                    "smc.hedgerock.rule_engine._CONFIDENCE_AGGRESSIVE_FLOOR"
                )
            },
        },
        blocking_reasons_per_candidate={},
        timeframe_consensus=consensus,
        stop_recommendation=stop,
    )
    by_id = {p.candidate_id: p for p in proposals}
    assert by_id["c3-aggressive-floor-0.78"].decision_reason \
        != REASON_EXTREME_VOLATILITY


# ---------------------------------------------------------------------------
# Reason ids stable
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_p1_reason_ids_are_stable() -> None:
    assert REASON_TIMEFRAME_CONSENSUS_INSUFFICIENT == \
        "timeframe_consensus_insufficient"
    assert REASON_EXTREME_VOLATILITY == "extreme_volatility"


# ---------------------------------------------------------------------------
# Backward compat — omit the new kwargs
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_legacy_callers_without_p1_kwargs_still_work() -> None:
    proposals = generate_candidate_proposals(
        candidate_menu=CANDIDATE_MENU_V0,
        bundle=_bundle(),
        gate_results_per_candidate={
            "c1-lower-observe-floor-0.50": {
                "G6": _g6_for(
                    "smc.hedgerock.rule_engine._CONFIDENCE_OBSERVE_FLOOR"
                )
            },
        },
        blocking_reasons_per_candidate={},
    )
    by_id = {p.candidate_id: p for p in proposals}
    assert by_id["c1-lower-observe-floor-0.50"].decision == DECISION_RECOMMEND


# ---------------------------------------------------------------------------
# Combined precedence — anomaly LOCKDOWN beats consensus + stop
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_anomaly_lockdown_wins_over_consensus_block() -> None:
    """When both LOCKDOWN anomaly AND consensus insufficiency apply,
    the anomaly reason takes precedence (it sits first in the
    short-circuit chain)."""
    from smc.hedgerock.evolution.anomaly_shield import (
        AnomalyLevel, AnomalyState,
    )
    from smc.hedgerock.evolution.candidate_generator import (
        REASON_MARKET_ANOMALY,
    )
    anomaly = AnomalyState(
        level=AnomalyLevel.LOCKDOWN, triggers=("gap_lockdown:0.012",),
        short_window_vol=0.030, historical_vol_p90=0.001,
        historical_vol_p95=0.002, historical_vol_p99=0.003,
        max_gap_pct=0.012, n_bars_observed=60,
        last_anomaly_at=None, next_recovery_at=None,
        blocking_conditions=(),
    )
    consensus = compute_consensus(
        d1_state=TimeframeState.ACCUMULATING,
        h4_state=TimeframeState.READY,
        h1_state=TimeframeState.READY,
        m5_state=TimeframeState.READY,
    )
    proposals = generate_candidate_proposals(
        candidate_menu=CANDIDATE_MENU_V0,
        bundle=_bundle(),
        gate_results_per_candidate={},
        blocking_reasons_per_candidate={},
        anomaly_state=anomaly,
        timeframe_consensus=consensus,
    )
    for p in proposals:
        assert p.decision_reason == REASON_MARKET_ANOMALY


@pytest.mark.unit
def test_consensus_session_profile_passed_through() -> None:
    """Smoke that consensus carries session_profile; downstream
    callers (e.g. recommendation CLI) need to read this for the
    Stop-Loss Advisory section."""
    consensus = compute_consensus(
        d1_state=TimeframeState.READY,
        h4_state=TimeframeState.READY,
        h1_state=TimeframeState.READY,
        m5_state=TimeframeState.READY,
        now=datetime(2026, 5, 2, 15, 0, tzinfo=timezone.utc),
    )
    assert consensus.active_session == TradingSession.OVERLAP
    assert consensus.session_profile.position_scale == 1.3

"""P0-1 — Property-based invariant tests for the evolution sidecar.

Each test is a hypothesis ``@given`` that fuzzes its input strategy
and asserts ONE narrow safety invariant. The strategies stay inside
realistic XAUUSD-shaped OHLC ranges so the runtime is bounded;
``HealthCheck`` profiles are tightened to keep CI hermetic.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest

from hypothesis import HealthCheck, given, settings, strategies as st

from smc.hedgerock.evolution.adaptive_stops import (
    ATR_MULT_HI,
    ATR_MULT_LO,
    POSITION_SCALE_BY_REGIME,
    StopRecommendation,
    VolatilityRegime,
    atr_multiplier_for_ratio,
    compute_stop_recommendation,
)
from smc.hedgerock.evolution.adversarial_scenarios import (
    BUILTIN_SCENARIOS,
    generate_synthetic_stress,
)
from smc.hedgerock.evolution.anomaly_shield import (
    AnomalyDetector,
    AnomalyLevel,
    shield_action,
)
from smc.hedgerock.evolution.candidate_generator import (
    CandidateProposal,
    DECISION_NO_RECOMMENDATION,
    DECISION_RECOMMEND,
    REASON_MARKET_ANOMALY,
    SAFETY_CLAMPS,
    generate_candidate_proposals,
)
from smc.hedgerock.evolution.candidate_menu import CANDIDATE_MENU_V0
from smc.hedgerock.evolution.multi_timeframe_state import (
    CONSENSUS_THRESHOLD,
    SESSION_PROFILES,
    TimeframeState,
    TradingSession,
    compute_consensus,
)
from smc.hedgerock.evolution.policy_manifest import (
    EvidenceBundle,
    GateStatus,
    PromotionGateResult,
)
from smc.hedgerock.evolution.registry_audit import RegistryAuditState
from smc.hedgerock.evolution.queue_aging import (
    DEFAULT_STALE_AFTER_DAYS,
    mark_stale_entries,
)
from smc.hedgerock.evolution.regime_engine import (
    GATE_BASE_WEIGHTS,
    MarketRegime,
    RegimeDetector,
    regime_adaptive_weights,
)
from smc.hedgerock.evolution.shadow_test_queue import (
    ShadowTestQueue,
    build_queue_entry,
)
from smc.hedgerock.evolution.stress_tester import (
    StressTester,
    VERDICT_BREACHED,
    VERDICT_PARTIAL,
    VERDICT_SURVIVED,
)


# ---------------------------------------------------------------------------
# Shared strategies
# ---------------------------------------------------------------------------


_PROFILE = settings(
    max_examples=40,
    deadline=2000,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)


def _bar_strategy() -> st.SearchStrategy[dict]:
    """One OHLC bar with realistic XAUUSD price levels and the
    high≥max(open,close) ≥ min(open,close)≥low invariant."""

    def _build(open_: float, close: float, high_extra: float, low_extra: float):
        hi = max(open_, close) + high_extra
        lo = min(open_, close) - low_extra
        return {"open": open_, "close": close, "high": hi, "low": max(lo, 0.01)}

    return st.builds(
        _build,
        open_=st.floats(min_value=500.0, max_value=4000.0, allow_nan=False, allow_infinity=False),
        close=st.floats(min_value=500.0, max_value=4000.0, allow_nan=False, allow_infinity=False),
        high_extra=st.floats(min_value=0.0, max_value=20.0, allow_nan=False, allow_infinity=False),
        low_extra=st.floats(min_value=0.0, max_value=20.0, allow_nan=False, allow_infinity=False),
    )


def _bars_strategy(min_size: int = 30, max_size: int = 80) -> st.SearchStrategy[list[dict]]:
    return st.lists(_bar_strategy(), min_size=min_size, max_size=max_size)


def _macro_strategy() -> st.SearchStrategy[dict]:
    return st.fixed_dictionaries({
        "VIX": st.floats(min_value=8.0, max_value=80.0, allow_nan=False),
        "DXY": st.floats(min_value=80.0, max_value=120.0, allow_nan=False),
        "US10Y": st.floats(min_value=0.0, max_value=8.0, allow_nan=False),
    })


# ---------------------------------------------------------------------------
# 1. Regime Engine invariants
# ---------------------------------------------------------------------------


@given(bars=_bars_strategy(), macro=_macro_strategy())
@_PROFILE
def test_invariant_regime_detect_always_returns_enum_member(bars, macro) -> None:
    snap = RegimeDetector().detect(bars=bars, macro=macro)
    assert isinstance(snap.regime, MarketRegime)


@given(bars=_bars_strategy())
@_PROFILE
def test_invariant_regime_detect_never_raises(bars) -> None:
    """No OHLC sequence in the realistic band should raise — the
    detector falls back to NORMAL with blocking_conditions when the
    inputs are degenerate."""
    RegimeDetector().detect(bars=bars)


@given(regime=st.sampled_from(list(MarketRegime)))
@_PROFILE
def test_invariant_adaptive_weights_strictly_positive(regime) -> None:
    weights = regime_adaptive_weights(regime=regime)
    for gate, w in weights.items():
        assert w > 0, f"non-positive weight for {gate} in {regime}: {w}"


@given(
    regime=st.sampled_from(list(MarketRegime)),
    base=st.dictionaries(
        keys=st.sampled_from(list(GATE_BASE_WEIGHTS.keys())),
        values=st.floats(min_value=0.1, max_value=10.0, allow_nan=False),
        min_size=8, max_size=8,
    ),
)
@_PROFILE
def test_invariant_adaptive_weights_total_in_reasonable_band(regime, base) -> None:
    """Weighted total must stay within an order of magnitude of the
    base sum — never 0, never astronomically large."""
    weights = regime_adaptive_weights(regime=regime, base_weights=base)
    base_total = sum(base.values())
    total = sum(weights.values())
    assert total > 0
    # Weight multipliers are in [0.7, 1.5], so the total stays in
    # [0.7×base, 1.5×base].
    assert 0.5 * base_total <= total <= 2.0 * base_total


# ---------------------------------------------------------------------------
# 2. Anomaly Shield invariants
# ---------------------------------------------------------------------------


@given(bars=_bars_strategy())
@_PROFILE
def test_invariant_anomaly_detect_returns_enum(bars) -> None:
    state = AnomalyDetector().detect(bars=bars)
    assert isinstance(state.level, AnomalyLevel)


@given(bars=_bars_strategy())
@_PROFILE
def test_invariant_shield_threshold_multiplier_non_decreasing(bars) -> None:
    """confidence_threshold_multiplier is the FACTOR applied to the
    live confidence floor — by spec it tightens the floor (≥ 1.0)
    or stays unchanged (= 1.0). It can never *loosen* the floor."""
    state = AnomalyDetector().detect(bars=bars)
    action = shield_action(state)
    assert action.confidence_threshold_multiplier >= 1.0


@given(bars=_bars_strategy())
@_PROFILE
def test_invariant_lockdown_freezes_queue(bars) -> None:
    state = AnomalyDetector().detect(bars=bars)
    action = shield_action(state)
    if state.level == AnomalyLevel.LOCKDOWN:
        assert action.queue_frozen is True
        assert action.full_lockdown is True
        assert action.new_candidates_allowed is False


@given(level=st.sampled_from([
    AnomalyLevel.ELEVATED, AnomalyLevel.CRITICAL, AnomalyLevel.LOCKDOWN,
]))
@_PROFILE
def test_invariant_recovery_only_steps_down_by_one_per_window(level) -> None:
    """Synthesise a previous-state at ``level`` and verify a single
    quiet detection moves the level down by AT MOST one step (never
    skips a tier). NORMAL is excluded — there's no "below NORMAL"
    state, and a detector that re-fires on noisy quiet-tape would
    legitimately produce ELEVATED, which is an UPward move not a
    recovery step."""
    from datetime import timedelta
    from smc.hedgerock.evolution.anomaly_shield import (
        AnomalyState, RECOVERY_STEP_MINUTES,
    )
    t0 = datetime(2026, 5, 2, 12, 0, tzinfo=timezone.utc)
    seed = AnomalyState(
        level=level, triggers=("seed",),
        short_window_vol=0.001, historical_vol_p90=0.001,
        historical_vol_p95=0.002, historical_vol_p99=0.003,
        max_gap_pct=0.0, n_bars_observed=60,
        last_anomaly_at=t0.isoformat(),
        next_recovery_at=None,
        blocking_conditions=(),
    )
    quiet_bars = [
        {"open": 2000.0, "close": 2000.0,
         "high": 2000.5, "low": 1999.5}
        for _ in range(60)
    ]
    one_step_later = t0 + timedelta(minutes=RECOVERY_STEP_MINUTES)
    s = AnomalyDetector().detect(
        bars=quiet_bars, previous_state=seed, now=one_step_later,
    )
    rank = {l: i for i, l in enumerate(AnomalyLevel)}
    assert rank[s.level] <= rank[level]
    assert rank[level] - rank[s.level] <= 1


# ---------------------------------------------------------------------------
# 3. Candidate Generator invariants
# ---------------------------------------------------------------------------


def _evidence_bundle() -> EvidenceBundle:
    return EvidenceBundle(
        bundle_id="<inv>",
        bundle_hash_sha256="0" * 64,
        atlas_report_path="<inv>",
        atlas_report_hash_sha256="0" * 64,
        data_availability_report_path="<inv>",
        data_availability_report_hash_sha256="0" * 64,
        walk_forward_run_paths=("<inv>",),
        year_replication={
            "XAUUSD": {"years_total": 5, "years_passing": 5,
                       "negative_sign_years": ()},
        },
        cross_symbol_count=1,
        halt_event_count=42,
        no_strategy_change=True,
    )


def _audit_state(*, violation: bool, present: bool) -> RegistryAuditState:
    return RegistryAuditState(
        audit_log_path="/tmp/x",
        audit_log_present=present,
        stale_v030_deleted_during_this_session=violation,
        lost_sha_count=4 if violation else 0,
        lost_sha256=(),
        registry_append_only_violation=violation,
    )


@given(triggered=st.booleans())
@_PROFILE
def test_invariant_proposed_value_inside_safety_clamp(triggered) -> None:
    """Every RECOMMEND proposal's proposed_value must lie inside the
    SAFETY_CLAMPS band for its parameter class."""
    g6 = PromotionGateResult(
        gate_id="G6", status=GateStatus.FAIL,
        reason=(
            "safety_bound_undefined for "
            "smc.hedgerock.rule_engine._CONFIDENCE_OBSERVE_FLOOR"
        ),
    )
    proposals = generate_candidate_proposals(
        candidate_menu=CANDIDATE_MENU_V0,
        bundle=_evidence_bundle(),
        gate_results_per_candidate={
            "c1-lower-observe-floor-0.50": {"G6": g6},
        } if triggered else {},
        blocking_reasons_per_candidate={},
    )
    for p in proposals:
        if p.decision != DECISION_RECOMMEND:
            continue
        clamp = SAFETY_CLAMPS.get(p.parameter_class)
        if clamp is None:
            continue
        assert clamp.lo <= p.proposed_value <= clamp.hi


@given(
    violation=st.booleans(),
    audit_present=st.booleans(),
)
@_PROFILE
def test_invariant_violation_blocks_every_recommend(violation, audit_present) -> None:
    """Any registry append-only violation forces every candidate to
    NO_RECOMMENDATION/evidence_chain_invalid."""
    bundle = _evidence_bundle()
    if violation or not audit_present:
        from dataclasses import replace as _replace
        bundle = _replace(
            bundle,
            registry_audit=_audit_state(
                violation=violation, present=audit_present,
            ),
        )
    proposals = generate_candidate_proposals(
        candidate_menu=CANDIDATE_MENU_V0,
        bundle=bundle,
        gate_results_per_candidate={},
        blocking_reasons_per_candidate={},
    )
    if violation:
        assert all(p.decision == DECISION_NO_RECOMMENDATION for p in proposals)


@given(level=st.sampled_from([AnomalyLevel.CRITICAL, AnomalyLevel.LOCKDOWN]))
@_PROFILE
def test_invariant_critical_or_lockdown_blocks_every_recommend(level) -> None:
    from smc.hedgerock.evolution.anomaly_shield import AnomalyState
    anomaly = AnomalyState(
        level=level, triggers=("inv",),
        short_window_vol=0.05, historical_vol_p90=0.001,
        historical_vol_p95=0.002, historical_vol_p99=0.003,
        max_gap_pct=0.0, n_bars_observed=60,
        last_anomaly_at=None, next_recovery_at=None,
        blocking_conditions=(),
    )
    proposals = generate_candidate_proposals(
        candidate_menu=CANDIDATE_MENU_V0,
        bundle=_evidence_bundle(),
        gate_results_per_candidate={},
        blocking_reasons_per_candidate={},
        anomaly_state=anomaly,
    )
    for p in proposals:
        assert p.decision == DECISION_NO_RECOMMENDATION
        assert p.decision_reason == REASON_MARKET_ANOMALY


# ---------------------------------------------------------------------------
# 4. Adaptive Stops invariants
# ---------------------------------------------------------------------------


@given(
    sigma_ratio=st.floats(min_value=-5.0, max_value=20.0, allow_nan=False, allow_infinity=False),
)
@_PROFILE
def test_invariant_atr_multiplier_in_band(sigma_ratio) -> None:
    m = atr_multiplier_for_ratio(sigma_ratio)
    assert ATR_MULT_LO <= m <= ATR_MULT_HI


@given(bars=_bars_strategy(min_size=90, max_size=120))
@_PROFILE
def test_invariant_position_scale_in_band(bars) -> None:
    rec = compute_stop_recommendation(bars=bars)
    assert 0.3 <= rec.position_scale <= 1.5
    # Vol regime must be a real enum member.
    assert isinstance(rec.vol_regime, VolatilityRegime)


@given(bars=_bars_strategy(min_size=20, max_size=120))
@_PROFILE
def test_invariant_advisory_only_always_true(bars) -> None:
    rec = compute_stop_recommendation(bars=bars)
    assert rec.advisory_only is True


@given(regime=st.sampled_from(list(VolatilityRegime)))
@_PROFILE
def test_invariant_position_scale_table_in_band(regime) -> None:
    scale = POSITION_SCALE_BY_REGIME[regime]
    assert 0.3 <= scale <= 1.5


# ---------------------------------------------------------------------------
# 5. Multi-timeframe invariants
# ---------------------------------------------------------------------------


_STATES = list(TimeframeState)


@given(
    h4=st.sampled_from(_STATES),
    h1=st.sampled_from(_STATES),
    m5=st.sampled_from(_STATES),
)
@_PROFILE
def test_invariant_d1_accumulating_blocks_can_recommend(h4, h1, m5) -> None:
    cons = compute_consensus(
        d1_state=TimeframeState.ACCUMULATING,
        h4_state=h4, h1_state=h1, m5_state=m5,
    )
    assert cons.can_recommend is False


@given(
    d1=st.sampled_from(_STATES),
    h4=st.sampled_from(_STATES),
    h1=st.sampled_from(_STATES),
    m5=st.sampled_from(_STATES),
)
@_PROFILE
def test_invariant_consensus_score_in_unit_interval(d1, h4, h1, m5) -> None:
    cons = compute_consensus(d1_state=d1, h4_state=h4, h1_state=h1, m5_state=m5)
    assert 0.0 <= cons.consensus_score <= 1.0


# ---------------------------------------------------------------------------
# 6. Stress Tester invariants
# ---------------------------------------------------------------------------


@given(
    parameter_class=st.sampled_from([
        "confidence_threshold_observe",
        "confidence_threshold_aggressive",
        "confidence_threshold_range_2",
        "halt_expiry_observe_hours",
    ]),
    proposed=st.floats(min_value=0.05, max_value=20.0, allow_nan=False, allow_infinity=False),
)
@_PROFILE
def test_invariant_stress_verdict_is_enum(parameter_class, proposed) -> None:
    proposal = CandidateProposal(
        candidate_id="prop",
        parameter_target="smc.hedgerock.rule_engine._CONFIDENCE_OBSERVE_FLOOR",
        parameter_class=parameter_class,
        baseline_value=0.55, proposed_value=proposed,
        triggered_by=("inv",),
        expected_improvement="t", risks=(), next_validation=(),
        decision=DECISION_RECOMMEND, decision_reason="",
    )
    live = {
        "confidence_threshold_observe": 0.55,
        "confidence_threshold_aggressive": 0.80,
        "confidence_threshold_range_2": 0.65,
        "halt_expiry_observe_hours": 4.0,
    }
    results = StressTester().test_candidate(proposal, live)
    for r in results:
        assert r.verdict in (VERDICT_SURVIVED, VERDICT_PARTIAL, VERDICT_BREACHED)


@given(scenario_id=st.sampled_from(list(BUILTIN_SCENARIOS.keys())))
@_PROFILE
def test_invariant_stress_max_drawdown_non_negative(scenario_id) -> None:
    """``max_drawdown_pct`` is reported as a positive percentage of
    the running peak — invariant: ≥ 0."""
    scenario = BUILTIN_SCENARIOS[scenario_id]
    proposal = CandidateProposal(
        candidate_id="prop",
        parameter_target="smc.hedgerock.rule_engine._CONFIDENCE_OBSERVE_FLOOR",
        parameter_class="confidence_threshold_observe",
        baseline_value=0.55, proposed_value=0.55,
        triggered_by=("inv",),
        expected_improvement="t", risks=(), next_validation=(),
        decision=DECISION_RECOMMEND, decision_reason="",
    )
    live = {"confidence_threshold_observe": 0.55,
            "confidence_threshold_aggressive": 0.80,
            "confidence_threshold_range_2": 0.65,
            "halt_expiry_observe_hours": 4.0}
    results = StressTester(scenarios=(scenario,)).test_candidate(proposal, live)
    for r in results:
        assert r.max_drawdown_pct >= 0.0
        assert r.historical_max_drawdown_pct >= 0.0


# ---------------------------------------------------------------------------
# 7. Queue invariants
# ---------------------------------------------------------------------------


def _make_proposal(suffix: str) -> CandidateProposal:
    return CandidateProposal(
        candidate_id=f"c-prop-{suffix}",
        parameter_target="smc.hedgerock.rule_engine._CONFIDENCE_OBSERVE_FLOOR",
        parameter_class="confidence_threshold_observe",
        baseline_value=0.55, proposed_value=0.50,
        triggered_by=("inv",),
        expected_improvement="t", risks=(), next_validation=(),
        decision=DECISION_RECOMMEND, decision_reason="",
    )


@given(n=st.integers(min_value=1, max_value=8))
@_PROFILE
def test_invariant_queue_append_only(n, tmp_path_factory) -> None:
    tmp = tmp_path_factory.mktemp("inv_queue")
    queue = ShadowTestQueue(
        path=tmp / "queue.jsonl",
        audit_log_path=tmp / "_audit.md",
    )
    first_batch = [_make_proposal(f"first-{i}") for i in range(n)]
    queue.enqueue_proposals(first_batch)
    snapshot_first = queue.path.read_text(encoding="utf-8")

    second_batch = [_make_proposal(f"second-{i}") for i in range(n)]
    queue.enqueue_proposals(second_batch)
    snapshot_second = queue.path.read_text(encoding="utf-8")
    # Append-only: original lines preserved verbatim as a prefix of
    # the post-second-enqueue file.
    assert snapshot_second.startswith(snapshot_first)


@given(
    age_hours=st.integers(min_value=0, max_value=720),
    threshold=st.integers(min_value=1, max_value=72),
)
@_PROFILE
def test_invariant_aging_only_marks_stale_never_deletes(age_hours, threshold, tmp_path_factory) -> None:
    """``age_queue_entries`` may APPEND a STALE marker but must not
    remove or rewrite the original QUEUED line."""
    tmp = tmp_path_factory.mktemp("inv_aging")
    queue = ShadowTestQueue(
        path=tmp / "queue.jsonl",
        audit_log_path=tmp / "_audit.md",
    )
    queue.enqueue_proposals([_make_proposal("aging")])
    pre = queue.path.read_text(encoding="utf-8")

    from datetime import datetime as _dt, timedelta as _td, timezone as _tz
    now = _dt.now(_tz.utc) + _td(hours=age_hours)
    mark_stale_entries(
        queue_path=queue.path,
        stale_after_days=max(1, threshold // 24),
        now=now,
    )
    post = queue.path.read_text(encoding="utf-8")
    # Original lines preserved as prefix.
    assert post.startswith(pre)

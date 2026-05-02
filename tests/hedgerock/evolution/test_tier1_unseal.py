"""Tests for the Tier-1 unseal — read-only wiring of
``replay_validator`` ↔ ``phase_d_walk_forward`` and
``candidate_generator`` ↔ ``decision_server``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from smc.hedgerock import decision_server, phase_d_walk_forward
from smc.hedgerock.evolution.candidate_generator import (
    DECISION_RECOMMEND,
    get_live_parameter_snapshot,
    resolve_baseline_value,
)
from smc.hedgerock.evolution.replay_validator import (
    ReplayValidationReport,
    summarise_replay_with_backtest,
)


_REPO = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# 1. decision_server contract
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_decision_server_returns_known_keys() -> None:
    live = decision_server.get_live_parameters()
    assert isinstance(live, dict)
    assert set(live.keys()) == set(decision_server.LIVE_PARAMETER_KEYS)
    for v in live.values():
        assert isinstance(v, float)


@pytest.mark.unit
def test_decision_server_returns_defensive_copy() -> None:
    a = decision_server.get_live_parameters()
    a["confidence_threshold_observe"] = 999.0
    b = decision_server.get_live_parameters()
    assert b["confidence_threshold_observe"] != 999.0


@pytest.mark.unit
def test_decision_server_internal_mapping_is_immutable() -> None:
    """The internal mapping is a MappingProxy — assigning to it raises."""
    from types import MappingProxyType
    assert isinstance(
        decision_server._CURRENT_LIVE_PARAMETERS, MappingProxyType
    )
    with pytest.raises(TypeError):
        decision_server._CURRENT_LIVE_PARAMETERS[  # type: ignore[index]
            "confidence_threshold_observe"
        ] = 0.0


# ---------------------------------------------------------------------------
# 2. phase_d_walk_forward contract
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_walk_forward_run_is_pure_and_deterministic() -> None:
    history = [
        {
            "window_id": "y2024_q1", "pnl_pp": 1.0, "dd_pp": 0.5,
            "n_signals": 12, "regime_bucket": "range",
        },
        {
            "window_id": "y2024_q2", "pnl_pp": -0.5, "dd_pp": 0.2,
            "n_signals": 7, "regime_bucket": "trend",
        },
    ]
    a = phase_d_walk_forward.run_walk_forward_backtest(
        parameters={"confidence_threshold_observe": 0.55},
        history=history,
    )
    b = phase_d_walk_forward.run_walk_forward_backtest(
        parameters={"confidence_threshold_observe": 0.55},
        history=history,
    )
    assert a == b
    assert a.n_windows == 2


@pytest.mark.unit
def test_walk_forward_does_not_mutate_history() -> None:
    history = [
        {"window_id": "y2024_q1", "pnl_pp": 1.0, "dd_pp": 0.5},
    ]
    snapshot = [dict(h) for h in history]
    phase_d_walk_forward.run_walk_forward_backtest(
        parameters={"confidence_threshold_observe": 0.55},
        history=history,
    )
    assert history == snapshot


@pytest.mark.unit
def test_walk_forward_skips_items_without_window_id() -> None:
    history = [
        {"pnl_pp": 1.0, "dd_pp": 0.5},  # no window_id
        {"window_id": "y2024_q1", "pnl_pp": 1.0, "dd_pp": 0.5},
    ]
    res = phase_d_walk_forward.run_walk_forward_backtest(
        parameters={}, history=history,
    )
    assert res.n_windows == 1


@pytest.mark.unit
def test_walk_forward_results_are_frozen() -> None:
    res = phase_d_walk_forward.run_walk_forward_backtest(
        parameters={}, history=[],
    )
    with pytest.raises(Exception):
        res.aggregate_pnl_pp = 999.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 3. candidate_generator wiring through decision_server
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_get_live_parameter_snapshot_matches_decision_server() -> None:
    assert get_live_parameter_snapshot() == decision_server.get_live_parameters()


@pytest.mark.unit
def test_resolve_baseline_prefers_live_snapshot() -> None:
    live_observe = decision_server.get_live_parameters()[
        "confidence_threshold_observe"
    ]
    out = resolve_baseline_value(
        parameter_class="confidence_threshold_observe",
        manifest_baseline=99.99,
    )
    assert out == live_observe


@pytest.mark.unit
def test_resolve_baseline_falls_back_to_manifest_for_unknown_class() -> None:
    out = resolve_baseline_value(
        parameter_class="totally_unknown_class",
        manifest_baseline=1.23,
    )
    assert out == 1.23


@pytest.mark.unit
def test_resolve_baseline_handles_none_class() -> None:
    out = resolve_baseline_value(
        parameter_class=None, manifest_baseline=4.5,
    )
    assert out == 4.5


# ---------------------------------------------------------------------------
# 4. replay_validator wiring through phase_d_walk_forward
# ---------------------------------------------------------------------------


def _history(n: int) -> list[dict]:
    return [
        {
            "window_id": f"y2024_w{i:02d}",
            "pnl_pp": 0.5 + (i % 3) * 0.1,
            "dd_pp": 0.2 + (i % 4) * 0.05,
            "n_signals": 10 + i,
            "regime_bucket": "range" if i % 2 == 0 else "trend",
        }
        for i in range(n)
    ]


@pytest.mark.unit
def test_summarise_replay_with_backtest_returns_report() -> None:
    rep = summarise_replay_with_backtest(
        candidate_id="c1-lower-observe-floor-0.50",
        parameter_class="confidence_threshold_observe",
        proposed_value=0.50,
        history=_history(10),
    )
    assert isinstance(rep, ReplayValidationReport)
    assert rep.n_windows_replayed == 10
    assert rep.n_artefacts_read == 0
    assert "insufficient_window_coverage" not in "|".join(
        rep.blocking_conditions
    )


@pytest.mark.unit
def test_summarise_replay_with_backtest_flags_short_history() -> None:
    rep = summarise_replay_with_backtest(
        candidate_id="c1",
        parameter_class="confidence_threshold_observe",
        proposed_value=0.50,
        history=_history(3),
    )
    blockers = "|".join(rep.blocking_conditions)
    assert "insufficient_window_coverage" in blockers


@pytest.mark.unit
def test_summarise_replay_with_backtest_rejects_unknown_class() -> None:
    rep = summarise_replay_with_backtest(
        candidate_id="c1",
        parameter_class="not_a_real_class",
        proposed_value=1.0,
        history=_history(10),
    )
    assert any(
        "unsupported_parameter_class" in b for b in rep.blocking_conditions
    )
    assert rep.n_windows_replayed == 0


@pytest.mark.unit
def test_summarise_replay_with_backtest_at_baseline_gives_zero_delta() -> None:
    """Calling with the live baseline value should produce ~0 delta."""
    live = decision_server.get_live_parameters()
    rep = summarise_replay_with_backtest(
        candidate_id="c-no-op",
        parameter_class="confidence_threshold_observe",
        proposed_value=live["confidence_threshold_observe"],
        history=_history(10),
    )
    assert rep.delta_pnl_pp_mean == 0.0
    assert rep.delta_dd_pp_worst == 0.0


# ---------------------------------------------------------------------------
# 5. End-to-end: a generator-issued RECOMMEND → backtest-validated
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_recommend_flow_uses_live_baseline_end_to_end() -> None:
    from smc.hedgerock.evolution.candidate_generator import (
        generate_candidate_proposals,
    )
    from smc.hedgerock.evolution.candidate_menu import CANDIDATE_MENU_V0
    from smc.hedgerock.evolution.policy_manifest import (
        EvidenceBundle,
        GateStatus,
        PromotionGateResult,
    )

    bundle = EvidenceBundle(
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
    g6_fail_for_observe = PromotionGateResult(
        gate_id="G6", status=GateStatus.FAIL,
        reason=(
            "safety_bound_undefined for "
            "smc.hedgerock.rule_engine._CONFIDENCE_OBSERVE_FLOOR"
        ),
    )
    proposals = generate_candidate_proposals(
        candidate_menu=CANDIDATE_MENU_V0,
        bundle=bundle,
        gate_results_per_candidate={
            "c1-lower-observe-floor-0.50": {"G6": g6_fail_for_observe},
        },
        blocking_reasons_per_candidate={},
    )
    by_id = {p.candidate_id: p for p in proposals}
    obs = by_id["c1-lower-observe-floor-0.50"]
    assert obs.decision == DECISION_RECOMMEND
    # Baseline anchored to live snapshot, not just manifest.
    live = decision_server.get_live_parameters()
    assert obs.baseline_value == live["confidence_threshold_observe"]

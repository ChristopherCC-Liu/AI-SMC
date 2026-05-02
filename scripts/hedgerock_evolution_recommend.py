"""Stage 4 — HedgeRock Evolution Recommendation CLI (report-only).

DIAGNOSTIC ONLY. Does NOT touch rule_engine, decision_server, .mq5,
config/safety_bounds.yaml, policy_registry/approved/, or
policy_registry/pointer.json.

Pipeline:
    1. Run the existing T4-F2/T4-F3 evolution-report pipeline (read
       evidence + registry-audit, evaluate G1–G8, render the
       diagnostic report).
    2. Feed the resulting candidates + bundle into the
       Stage 3 candidate generator. Each generator output is a
       :class:`CandidateProposal` (RECOMMEND / NO_RECOMMENDATION).
    3. Render a markdown recommendation document with explicit
       NOT LIVE / NOT APPROVED / NOT DEPLOYED banners, the per-
       candidate decision, and the next-step XAUUSD shadow
       validation list.

Outputs:
    - <recommendation-path>: the markdown recommendation file
    - <report-path>: the diagnostic evolution report (same as the
      existing T4-F3 CLI)

The CLI never writes under policy_registry/approved/ or
policy_registry/pointer.json. The recommendation file is written
exactly where the operator points ``--recommendation-path``.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Imports from the report-only sidecar layer. The candidate_generator
# module additionally reads live parameter values from
# decision_server (Tier-1 read-only unseal — RFC §1).
from smc.hedgerock.evolution.ascii_visualisations import (
    render_gate_matrix,
    render_heat_ranking,
    render_parameter_comparison_table,
)
from smc.hedgerock.evolution.adaptive_stops import (
    StopRecommendation,
    VolatilityRegime,
    compute_stop_recommendation,
)
from smc.hedgerock.evolution.anomaly_shield import (
    AnomalyDetector,
    AnomalyLevel,
    AnomalyState,
    shield_action,
)
from smc.hedgerock.evolution.multi_timeframe_state import (
    TimeframeConsensus,
    compute_consensus,
)
from smc.hedgerock.evolution.candidate_generator import (
    CandidateProposal,
    DECISION_NO_RECOMMENDATION,
    DECISION_RECOMMEND,
    REASON_EVIDENCE_CHAIN_INVALID,
    REASON_INSUFFICIENT_XAUUSD_COVERAGE,
    SAFETY_CLAMPS,
    generate_candidate_proposals,
    get_live_parameter_snapshot,
)
from smc.hedgerock.evolution.regime_engine import (
    MarketRegime,
    RegimeDetector,
    RegimeSnapshot,
)
from smc.hedgerock.evolution.candidate_menu import CANDIDATE_MENU_V0
from smc.hedgerock.evolution.policy_manifest import (
    CandidateManifest,
    EvidenceBundle,
    PromotionGateResult,
)


# Reuse the existing T4-F3 pipeline by importing the report CLI from
# scripts/. The report script is sidecar; importing it from another
# sidecar script does not cross the boundary.
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
import hedgerock_evolution_report as report_cli  # noqa: E402
sys.path.pop(0)


_FORBIDDEN_OUTPUT_FRAGMENTS = (
    "policy_registry/approved",
    "policy_registry/pointer.json",
    "config/safety_bounds.yaml",
    "rule_engine.py",
    "decision_server.py",
)


def _assert_recommendation_path_safe(path: Path) -> None:
    text = str(path)
    for fragment in _FORBIDDEN_OUTPUT_FRAGMENTS:
        if fragment in text:
            raise ValueError(
                f"recommendation path would write under a forbidden "
                f"location: {text!r} (matched {fragment!r})"
            )


# ---------------------------------------------------------------------------
# Render a single proposal block
# ---------------------------------------------------------------------------


def _render_proposal(p: CandidateProposal) -> list[str]:
    out: list[str] = []
    out.append(f"### {p.candidate_id}")
    out.append("")
    out.append(f"- parameter target: `{p.parameter_target}`")
    out.append(f"- parameter class: `{p.parameter_class}`")
    out.append(f"- baseline → proposed: `{p.baseline_value}` → `{p.proposed_value}`")
    out.append(f"- decision: `{p.decision}`")
    if p.decision_reason:
        out.append(f"- reason: `{p.decision_reason}`")
    if p.triggered_by:
        out.append("- triggered by:")
        for t in p.triggered_by:
            out.append(f"    - `{t}`")
    if p.expected_improvement:
        out.append(f"- expected improvement: {p.expected_improvement}")
    if p.risks:
        out.append("- risks:")
        for r in p.risks:
            out.append(f"    - {r}")
    if p.next_validation:
        out.append("- next validation:")
        for v in p.next_validation:
            out.append(f"    - {v}")
    out.append(f"- report_only: **{p.report_only}**")
    out.append("")
    return out


def _render_recommendation(
    *,
    proposals: list[CandidateProposal],
    bundle: EvidenceBundle,
    audit_log_path: Path,
    gate_results_per_candidate: dict[str, dict[str, "PromotionGateResult"]] | None = None,
    regime_snapshot: RegimeSnapshot | None = None,
    anomaly_state: AnomalyState | None = None,
    timeframe_consensus: TimeframeConsensus | None = None,
    stop_recommendation: StopRecommendation | None = None,
) -> str:
    audit = getattr(bundle, "registry_audit", None)
    audit_present = bool(getattr(audit, "audit_log_present", False))
    audit_violation = bool(getattr(audit, "registry_append_only_violation", False))

    n_recommend = sum(1 for p in proposals if p.decision == DECISION_RECOMMEND)
    n_no = sum(1 for p in proposals if p.decision == DECISION_NO_RECOMMENDATION)

    out: list[str] = []
    out.append("# HedgeRock Evolution Recommendation (Stage 4 — REPORT-ONLY)")
    out.append("")
    out.append("> This document is **NOT LIVE**, **NOT APPROVED**, "
               "**NOT DEPLOYED**. It is a report-only recommendation "
               "produced by the Stage 3 candidate generator. Promotion "
               "to `policy_registry/approved/` is a manual human step.")
    out.append("")
    out.append("## Banner block (machine-greppable)")
    out.append("")
    out.append("- status: **NOT LIVE**")
    out.append("- approval: **NOT APPROVED**")
    out.append("- deployment: **NOT DEPLOYED**")
    out.append("- coverage: **XAUUSD only** (multi-symbol out of scope)")
    out.append("- audit_log_path: " + f"`{audit_log_path}`")
    out.append(f"- `audit_log_present`: **{audit_present}**")
    out.append(f"- `registry_append_only_violation`: **{audit_violation}**")
    if regime_snapshot is not None:
        out.append(
            f"- `regime`: **{regime_snapshot.regime.value}** "
            f"(confidence={regime_snapshot.confidence:.2f})"
        )
    if anomaly_state is not None:
        out.append(
            f"- `anomaly_level`: **{anomaly_state.level.value}**"
        )
        if anomaly_state.triggers:
            out.append(
                "- `anomaly_triggers`: "
                + ", ".join(f"`{t}`" for t in anomaly_state.triggers)
            )
    out.append("")

    if anomaly_state is not None:
        action = shield_action(anomaly_state)
        if action.full_lockdown:
            out.append(f"> {action.banner}")
            out.append("")
        elif not action.new_candidates_allowed:
            out.append(f"> {action.banner}")
            out.append("")

    if timeframe_consensus is not None:
        out.append("## Timeframe Consensus")
        out.append("")
        out.append(
            f"- session: **{timeframe_consensus.active_session.value}** "
            f"(position_scale={timeframe_consensus.session_profile.position_scale})"
        )
        out.append(
            f"- D1: **{timeframe_consensus.d1_state.value}**, "
            f"H4: **{timeframe_consensus.h4_state.value}**, "
            f"H1: **{timeframe_consensus.h1_state.value}**, "
            f"M5: **{timeframe_consensus.m5_state.value}**"
        )
        out.append(
            f"- consensus_score: **{timeframe_consensus.consensus_score:.3f}** "
            f"(threshold 0.70)"
        )
        out.append(
            f"- can_recommend: **{timeframe_consensus.can_recommend}**"
        )
        if timeframe_consensus.blocking_conditions:
            out.append("- blocking conditions:")
            for b in timeframe_consensus.blocking_conditions:
                out.append(f"    - `{b}`")
        out.append("")

    if stop_recommendation is not None:
        out.append("## Stop-Loss Advisory (ADVISORY ONLY — does not "
                   "modify live stops)")
        out.append("")
        out.append(
            f"- vol_regime: **{stop_recommendation.vol_regime.value}**"
        )
        out.append(
            f"- atr_multiplier: **{stop_recommendation.atr_multiplier}**"
        )
        out.append(
            f"- position_scale: **{stop_recommendation.position_scale}**"
        )
        out.append(
            f"- σ_ratio: **{stop_recommendation.sigma_ratio}**"
        )
        out.append(
            f"- Parkinson: **{stop_recommendation.parkinson_vol}**, "
            f"Garman-Klass: **{stop_recommendation.garman_klass_vol}**"
        )
        out.append(f"- reasoning: `{stop_recommendation.reasoning}`")
        if stop_recommendation.blocking_conditions:
            out.append("- blockers:")
            for b in stop_recommendation.blocking_conditions:
                out.append(f"    - `{b}`")
        out.append("")

    out.append("## Baseline summary")
    out.append("")
    yr = bundle.year_replication.get("XAUUSD", {})
    out.append(f"- XAUUSD years_total: {yr.get('years_total', 0)}")
    out.append(f"- XAUUSD years_passing: {yr.get('years_passing', 0)}")
    out.append(f"- halt_event_count: {bundle.halt_event_count}")
    out.append(f"- no_strategy_change: {bundle.no_strategy_change}")
    out.append("")

    out.append("### Live parameter snapshot")
    out.append("")
    out.append("(Read-only re-export from the candidate generator's "
               "`get_live_parameter_snapshot`, which is the only "
               "approved sidecar door onto the live `decision_server` "
               "values.)")
    out.append("")
    live = get_live_parameter_snapshot()
    for cls in sorted(live.keys()):
        out.append(f"- `{cls}`: **{live[cls]}**")
    out.append("")

    out.append("## Headline")
    out.append("")
    out.append(f"- proposals total: **{len(proposals)}**")
    out.append(f"- RECOMMEND: **{n_recommend}**")
    out.append(f"- NO_RECOMMENDATION: **{n_no}**")
    if audit_violation:
        out.append("")
        out.append(
            "  ⚠️ **Registry append-only contract violated this session.** "
            "All candidates fall back to "
            "`NO_RECOMMENDATION/evidence_chain_invalid`. "
            "Restart with a clean registry before re-running."
        )
    elif not audit_present:
        out.append("")
        out.append(
            "  ℹ️ **Audit log not present at the resolved path.** "
            "All candidates abstain from RECOMMEND until the operator "
            "re-points `--registry-audit-log`. Absent ≠ clean."
        )
    out.append("")

    out.append("## Parameter comparison")
    out.append("")
    out.append(render_parameter_comparison_table(proposals))
    out.append("")
    out.append("Legend: `b` = baseline; `p` = proposed; `B` = both at "
               "the same band position. Band visual width = 10 cells.")
    out.append("")

    if gate_results_per_candidate:
        out.append("## Gate matrix")
        out.append("")
        # Map PromotionGateResult → GateStatus for the visualiser.
        from smc.hedgerock.evolution.policy_manifest import GateStatus as _GS
        gate_status_map: dict[str, dict[str, _GS]] = {}
        for cid, gates in gate_results_per_candidate.items():
            gate_status_map[cid] = {gid: r.status for gid, r in gates.items()}
        out.append(render_gate_matrix(gate_status_map))
        out.append("")
        out.append("Glyphs: ✓ = PASS, ✗ = FAIL, · = ABSTAIN / NOT_RUN / "
                   "missing.")
        out.append("")

    out.append("## Heat ranking")
    out.append("")
    out.append(render_heat_ranking(proposals))
    out.append("")
    out.append("Order: RECOMMEND first, then by trigger count "
               "(descending), then by candidate_id (deterministic).")
    out.append("")

    out.append("## Per-candidate proposals")
    out.append("")
    for p in proposals:
        out.extend(_render_proposal(p))

    out.append("## Safety clamps applied (RFC §5)")
    out.append("")
    for cls, clamp in SAFETY_CLAMPS.items():
        out.append(f"- `{cls}`: [{clamp.lo}, {clamp.hi}]")
    out.append("")

    out.append("## Required next validation (XAUUSD shadow runner)")
    out.append("")
    out.append("- Run `shadow_runner` v0.3.0 on XAUUSD across the next "
               "rolling 8-window cohort.")
    out.append("- Re-evaluate G1–G8 against the refreshed Phase D bundle.")
    out.append("- Append (never overwrite) per-candidate artefacts under "
               "`policy_registry/shadow_artefacts/<candidate_id>/`.")
    out.append("- Feed shadow metrics + window-coverage results back to the "
               "Stage 3 candidate generator.")
    out.append("- **Human review and approval** before any promotion to "
               "`policy_registry/approved/`.")
    out.append("")

    out.append("## Risks (cross-candidate)")
    out.append("")
    out.append("- Tweaking confidence thresholds may flip ranges that pass "
               "today; verify with shadow run before approval.")
    out.append("- Halt-mode parameters depend on small halt-event corpora; "
               "expand the corpus before raising the auto-expiry hour cap.")
    out.append("- Registry-audit invariants must hold across the entire "
               "shadow window — any append-only breach forces NO_RECOMMENDATION.")
    out.append("")

    out.append("## Boundary boilerplate")
    out.append("")
    out.append("- This file does NOT modify `rule_engine.py`, "
               "`decision_server.py`, EA `*.mq5`, "
               "`config/safety_bounds.yaml`, "
               "`policy_registry/approved/`, or "
               "`policy_registry/pointer.json`.")
    out.append("- Every recommendation is sidecar; deployment requires a "
               "manual human step downstream.")
    out.append("")

    out.append(f"Generated at: {datetime.now(timezone.utc).isoformat()}")
    out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run(
    *,
    atlas_path: Path,
    availability_path: Path,
    walk_forward_paths: list[Path],
    safety_bounds_path: Path,
    registry_root: Path,
    report_path: Path,
    recommendation_path: Path,
    registry_audit_log_path: Path | None = None,
    market_bars: list[dict] | None = None,
    macro_context: dict | None = None,
    regime_snapshot: RegimeSnapshot | None = None,
    anomaly_state: AnomalyState | None = None,
    timeframe_consensus: TimeframeConsensus | None = None,
    stop_recommendation: StopRecommendation | None = None,
    timeframe_bars: dict | None = None,
    active_timeframes: tuple[str, ...] = (),
) -> tuple[list[CandidateProposal], Path]:
    """Library entry point.

    Returns the list of proposals + the path the recommendation
    markdown was written to. The diagnostic report is also written to
    ``report_path`` as a side effect (delegated to the existing
    T4-F3 CLI).
    """
    _assert_recommendation_path_safe(Path(recommendation_path))

    # Step 1 — run the diagnostic evolution-report pipeline.
    candidates, _ = report_cli.run(
        atlas_path=atlas_path,
        availability_path=availability_path,
        walk_forward_paths=walk_forward_paths,
        safety_bounds_path=safety_bounds_path,
        registry_root=registry_root,
        report_path=report_path,
        registry_audit_log_path=registry_audit_log_path,
    )

    # Step 2 — derive the gate-result map and blocking-reason map
    # per candidate from the evaluation output.
    gate_results_per_candidate: dict[str, dict[str, PromotionGateResult]] = {}
    blocking_reasons_per_candidate: dict[str, tuple[str, ...]] = {}
    bundle = candidates[0].evidence_bundle if candidates else None
    for c in candidates:
        gate_results_per_candidate[c.candidate_id] = {
            r.gate_id: r for r in c.gates
        }
        blocking_reasons_per_candidate[c.candidate_id] = tuple(c.blocking_reasons)

    if bundle is None:
        # This is a hard failure — the report pipeline would have
        # raised earlier if the bundle was missing.
        raise RuntimeError(
            "evolution report produced no candidates; cannot generate "
            "recommendations"
        )

    # Step 2b — derive regime + anomaly state. Caller may pass
    # pre-computed snapshots; otherwise we run the detectors against
    # ``market_bars`` (when provided). With no bars and no snapshot,
    # the loop runs unguarded — same behaviour as before this layer
    # was added.
    if regime_snapshot is None and market_bars:
        regime_snapshot = RegimeDetector().detect(
            bars=market_bars, macro=macro_context or {},
        )
    if anomaly_state is None and market_bars:
        anomaly_state = AnomalyDetector().detect(bars=market_bars)
    if stop_recommendation is None and market_bars:
        stop_recommendation = compute_stop_recommendation(bars=market_bars)
    if timeframe_consensus is None and timeframe_bars:
        timeframe_consensus = compute_consensus(
            d1_bars=timeframe_bars.get("D1"),
            h4_bars=timeframe_bars.get("H4"),
            h1_bars=timeframe_bars.get("H1"),
            m5_bars=timeframe_bars.get("M5"),
            active_timeframes=active_timeframes,
        )

    # Step 3 — call the candidate generator. Output dir is set to the
    # recommendation file's parent so the JSON snapshot lives alongside
    # the markdown for the operator.
    proposals = generate_candidate_proposals(
        candidate_menu=CANDIDATE_MENU_V0,
        bundle=bundle,
        gate_results_per_candidate=gate_results_per_candidate,
        blocking_reasons_per_candidate=blocking_reasons_per_candidate,
        output_dir=Path(recommendation_path).parent,
        regime_snapshot=regime_snapshot,
        anomaly_state=anomaly_state,
        timeframe_consensus=timeframe_consensus,
        stop_recommendation=stop_recommendation,
    )

    # Step 3b — when the audit log is absent, override every RECOMMEND
    # to NO_RECOMMENDATION/evidence_chain_invalid. The generator's own
    # short-circuit only fires on `violation=True`; absence is a
    # separate "operator must re-point flag" verdict that this CLI
    # owns. (Generator stays decoupled from filesystem state.)
    audit = getattr(bundle, "registry_audit", None)
    if audit is not None and not getattr(audit, "audit_log_present", False):
        proposals = [
            (
                # frozen dataclass → produce a sibling
                _replace_decision(
                    p,
                    decision=DECISION_NO_RECOMMENDATION,
                    decision_reason="audit_log_absent_re_point_flag",
                )
                if p.decision == DECISION_RECOMMEND
                else p
            )
            for p in proposals
        ]

    audit_path = Path(getattr(audit, "audit_log_path", "<unknown>")) \
        if audit is not None else Path("<unknown>")
    body = _render_recommendation(
        proposals=proposals, bundle=bundle, audit_log_path=audit_path,
        gate_results_per_candidate=gate_results_per_candidate,
        regime_snapshot=regime_snapshot,
        anomaly_state=anomaly_state,
        timeframe_consensus=timeframe_consensus,
        stop_recommendation=stop_recommendation,
    )
    Path(recommendation_path).parent.mkdir(parents=True, exist_ok=True)
    Path(recommendation_path).write_text(body, encoding="utf-8")
    return proposals, Path(recommendation_path)


def _replace_decision(
    p: CandidateProposal, *, decision: str, decision_reason: str,
) -> CandidateProposal:
    from dataclasses import replace
    return replace(p, decision=decision, decision_reason=decision_reason)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--atlas-report", type=Path, required=True)
    parser.add_argument("--data-availability-report", type=Path, required=True)
    parser.add_argument(
        "--walk-forward-report", type=Path, action="append", default=None,
    )
    parser.add_argument("--safety-bounds", type=Path, required=True)
    parser.add_argument("--registry-root", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--recommendation-path", type=Path, required=True)
    parser.add_argument("--registry-audit-log", type=Path, default=None)
    args = parser.parse_args(argv)

    wf_paths = args.walk_forward_report or []
    if not wf_paths:
        # The diagnostic report needs at least one walk-forward path;
        # default to the same convention as the report CLI.
        wf_paths = [args.data_availability_report.parent / "wf.md"]

    try:
        proposals, rec_path = run(
            atlas_path=args.atlas_report,
            availability_path=args.data_availability_report,
            walk_forward_paths=wf_paths,
            safety_bounds_path=args.safety_bounds,
            registry_root=args.registry_root,
            report_path=args.report_path,
            recommendation_path=args.recommendation_path,
            registry_audit_log_path=args.registry_audit_log,
        )
    except FileNotFoundError as e:
        print(f"FAILED: {e}", file=sys.stderr)
        return 2
    except ValueError as e:
        print(f"FAILED: {e}", file=sys.stderr)
        return 3

    n_recommend = sum(1 for p in proposals if p.decision == DECISION_RECOMMEND)
    n_no = sum(1 for p in proposals if p.decision == DECISION_NO_RECOMMENDATION)
    print(
        f"recommendations: {len(proposals)} total "
        f"({n_recommend} RECOMMEND / {n_no} NO_RECOMMENDATION)"
    )
    for p in proposals:
        suffix = (
            f" — {p.decision_reason}" if p.decision_reason else ""
        )
        print(f"  {p.candidate_id}: {p.decision}{suffix}")
    print(f"wrote recommendation → {rec_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

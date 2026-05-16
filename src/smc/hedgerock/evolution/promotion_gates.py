"""Phase D-cont3 / Ticket 1 — promotion gates.

**Pure functions, no FS access.** Each gate ``gN(...)`` accepts a
fully-loaded ``EvidenceBundle`` and the in-memory ``CandidateManifest``,
returning a ``PromotionGateResult``. Gate semantics follow Plan §3.
G8 always returns NOT_RUN in the MVP — the shadow runner is MVP+1.

This module never imports ``smc.hedgerock.rule_engine`` or
``smc.hedgerock.decision_server``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from smc.hedgerock.evolution.policy_manifest import (
    CandidateManifest,
    EvidenceBundle,
    GateStatus,
    OverallResult,
    PromotionGateResult,
)


__all__ = [
    "GATE_IDS",
    "GateConfig",
    "MIN_RUNNER_VERSION_FOR_ACTIVE_PASS_EVALUATION",
    "MIN_RUNNER_VERSION_FOR_PASS",
    "SafetyBoundsConfig",
    "compute_overall_result",
    "evaluate_all_gates",
    "g1_min_evidence",
    "g2_year_replication",
    "g3_cross_symbol",
    "g4_no_negative_sign_years",
    "g5_halt_event_corpus",
    "g6_safety_bounds",
    "g7_interface_stability",
    "g8_shadow_comparison",
    "parse_runner_version",
    "version_lt",
]


# Ticket 3 R2: hard gate keeping pre-Ticket-3 (zero-trade) artefacts
# from being eligible for PASS. Old artefacts max out at ABSTAIN.
# Lowering this constant is a human-only config change (same
# treatment as G2 / G5 thresholds per RFC §10.2).
MIN_RUNNER_VERSION_FOR_PASS: str = "shadow_runner-0.2.0"
# Ticket 4 v2 — only artefacts produced by the active multi-window
# PASS evaluator (shadow_runner-0.3.0) can certify PASS. Older
# artefacts predate the per-window risk surface and are downgraded
# to ABSTAIN at G8 (integrity is intact, but no per-window data
# means no honest PASS).
MIN_RUNNER_VERSION_FOR_ACTIVE_PASS_EVALUATION: str = "shadow_runner-0.3.0"


def parse_runner_version(s: str) -> tuple[int, int, int]:
    """Parse a runner_version string of the form 'shadow_runner-X.Y.Z'
    into a (major, minor, patch) tuple. Raises ValueError on anything
    else."""
    if not isinstance(s, str) or not s.startswith("shadow_runner-"):
        raise ValueError(f"unparseable runner_version: {s!r}")
    suffix = s[len("shadow_runner-"):]
    parts = suffix.split(".")
    if len(parts) != 3:
        raise ValueError(f"unparseable runner_version: {s!r}")
    try:
        return (int(parts[0]), int(parts[1]), int(parts[2]))
    except ValueError as e:
        raise ValueError(f"unparseable runner_version: {s!r}") from e


def version_lt(a: str, b: str) -> bool:
    """Return True iff parse(a) < parse(b)."""
    return parse_runner_version(a) < parse_runner_version(b)


GATE_IDS: tuple[str, ...] = (
    "G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8",
)


@dataclass(frozen=True)
class GateConfig:
    """Per RFC §10.2: G2 / G5 thresholds are registry config; lowering
    them is a human-only config change. Defaults match RFC §10.2."""

    g2_pass_rate_threshold: float = 0.75
    g5_min_halt_events: int = 30


@dataclass(frozen=True)
class SafetyBoundsConfig:
    """In-memory mirror of ``config/safety_bounds.yaml``. The sidecar
    NEVER writes to this object's source file; loaders pass the
    parsed mapping in here.

    A target with no entry is **undefined** — G6 returns FAIL with
    ``safety_bound_undefined`` per Plan §3 G6 (b).
    """

    # target dotted-path → (min, max)
    bounds: dict[str, tuple[float, float]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# G1 — minimum evidence
# ---------------------------------------------------------------------------


def g1_min_evidence(
    *, candidate: CandidateManifest, bundle: EvidenceBundle | None,
) -> PromotionGateResult:
    if bundle is None:
        return PromotionGateResult(
            gate_id="G1",
            status=GateStatus.FAIL,
            reason="no evidence bundle attached to candidate",
        )
    missing: list[str] = []
    if not bundle.atlas_report_hash_sha256:
        missing.append("atlas_report_hash")
    if not bundle.data_availability_report_hash_sha256:
        missing.append("data_availability_report_hash")
    if not bundle.walk_forward_run_paths:
        missing.append("walk_forward_runs (need ≥1)")
    if missing:
        return PromotionGateResult(
            gate_id="G1",
            status=GateStatus.FAIL,
            reason=f"missing evidence artefacts: {', '.join(missing)}",
        )
    return PromotionGateResult(
        gate_id="G1",
        status=GateStatus.PASS,
        reason="atlas + data_availability + walk_forward present",
    )


# ---------------------------------------------------------------------------
# G2 — year-replication
# ---------------------------------------------------------------------------


def g2_year_replication(
    *,
    candidate: CandidateManifest,
    bundle: EvidenceBundle | None,
    gate_config: GateConfig | None = None,
) -> PromotionGateResult:
    cfg = gate_config or GateConfig()
    if not candidate.diff.scope.affects_classifier_or_rule_engine:
        return PromotionGateResult(
            gate_id="G2",
            status=GateStatus.ABSTAIN,
            reason="candidate does not affect classifier or rule_engine",
        )
    if bundle is None or not bundle.year_replication:
        return PromotionGateResult(
            gate_id="G2",
            status=GateStatus.NOT_RUN,
            reason="year_replication evidence missing",
        )
    failing: list[str] = []
    for sym, body in bundle.year_replication.items():
        total = int(body["years_total"])
        passing = int(body["years_passing"])
        rate = passing / total if total > 0 else 0.0
        if rate < cfg.g2_pass_rate_threshold:
            failing.append(f"{sym}: {passing}/{total} pass (need ≥ {cfg.g2_pass_rate_threshold:.2f})")
    if failing:
        return PromotionGateResult(
            gate_id="G2",
            status=GateStatus.FAIL,
            reason="; ".join(failing),
            details={"failing_symbols": failing},
        )
    return PromotionGateResult(
        gate_id="G2",
        status=GateStatus.PASS,
        reason=f"every symbol passes ≥ {cfg.g2_pass_rate_threshold:.2f} year-replication threshold",
    )


# ---------------------------------------------------------------------------
# G3 — cross-symbol replication
# ---------------------------------------------------------------------------


def g3_cross_symbol(
    *,
    candidate: CandidateManifest,
    bundle: EvidenceBundle | None,
    gate_config: GateConfig | None = None,
) -> PromotionGateResult:
    cfg = gate_config or GateConfig()
    if bundle is None:
        return PromotionGateResult(
            gate_id="G3",
            status=GateStatus.NOT_RUN,
            reason="no bundle",
        )
    if bundle.cross_symbol_count < 2:
        return PromotionGateResult(
            gate_id="G3",
            status=GateStatus.ABSTAIN,
            reason=f"only {bundle.cross_symbol_count} symbol(s) in lake — "
                   "cross-symbol robustness untestable",
        )
    # ≥ 2 symbols. Each must pass G2's threshold; otherwise FAIL.
    failing: list[str] = []
    for sym, body in bundle.year_replication.items():
        total = int(body["years_total"])
        passing = int(body["years_passing"])
        rate = passing / total if total > 0 else 0.0
        if rate < cfg.g2_pass_rate_threshold:
            failing.append(f"{sym}: {passing}/{total}")
    if failing:
        return PromotionGateResult(
            gate_id="G3",
            status=GateStatus.FAIL,
            reason=f"cross-symbol regression: {'; '.join(failing)}",
        )
    return PromotionGateResult(
        gate_id="G3",
        status=GateStatus.PASS,
        reason=f"{bundle.cross_symbol_count} symbols, all pass G2",
    )


# ---------------------------------------------------------------------------
# G4 — no negative-sign years
# ---------------------------------------------------------------------------


def g4_no_negative_sign_years(
    *, candidate: CandidateManifest, bundle: EvidenceBundle | None,
) -> PromotionGateResult:
    if not candidate.diff.scope.affects_classifier_or_rule_engine:
        return PromotionGateResult(
            gate_id="G4",
            status=GateStatus.ABSTAIN,
            reason="candidate does not affect classifier or rule_engine",
        )
    if bundle is None or not bundle.year_replication:
        return PromotionGateResult(
            gate_id="G4",
            status=GateStatus.NOT_RUN,
            reason="year_replication evidence missing",
        )
    naming: list[str] = []
    for sym, body in bundle.year_replication.items():
        neg = tuple(body.get("negative_sign_years", ()))
        if neg:
            years_str = ", ".join(str(y) for y in neg)
            naming.append(f"{sym}: {years_str}")
    if naming:
        return PromotionGateResult(
            gate_id="G4",
            status=GateStatus.FAIL,
            reason="trend_up reverse-signed in: " + "; ".join(naming),
            details={"negative_sign_by_symbol": {s: list(b.get("negative_sign_years", ())) for s, b in bundle.year_replication.items()}},
        )
    return PromotionGateResult(
        gate_id="G4",
        status=GateStatus.PASS,
        reason="no negative-sign years across any symbol",
    )


# ---------------------------------------------------------------------------
# G5 — halt event corpus
# ---------------------------------------------------------------------------


def g5_halt_event_corpus(
    *,
    candidate: CandidateManifest,
    bundle: EvidenceBundle | None,
    gate_config: GateConfig | None = None,
) -> PromotionGateResult:
    cfg = gate_config or GateConfig()
    if not candidate.diff.scope.affects_halt_mode:
        return PromotionGateResult(
            gate_id="G5",
            status=GateStatus.ABSTAIN,
            reason="candidate does not affect halt-mode behaviour",
        )
    if bundle is None:
        return PromotionGateResult(
            gate_id="G5",
            status=GateStatus.NOT_RUN,
            reason="no bundle",
        )
    if bundle.halt_event_count < cfg.g5_min_halt_events:
        return PromotionGateResult(
            gate_id="G5",
            status=GateStatus.FAIL,
            reason=f"halt-event corpus n={bundle.halt_event_count} < {cfg.g5_min_halt_events}",
        )
    return PromotionGateResult(
        gate_id="G5",
        status=GateStatus.PASS,
        reason=f"halt-event corpus n={bundle.halt_event_count} ≥ {cfg.g5_min_halt_events}",
    )


# ---------------------------------------------------------------------------
# G6 — safety bounds (Plan §3 G6)
# ---------------------------------------------------------------------------


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def g6_safety_bounds(
    *, candidate: CandidateManifest, bounds: SafetyBoundsConfig,
) -> PromotionGateResult:
    proposed = candidate.diff.proposed_value
    target = candidate.diff.target

    if not _is_numeric(proposed):
        return PromotionGateResult(
            gate_id="G6",
            status=GateStatus.ABSTAIN,
            reason="no numeric values to check",
        )

    if target not in bounds.bounds:
        # Plan §3 G6 (b): undefined bound → FAIL: safety_bound_undefined
        return PromotionGateResult(
            gate_id="G6",
            status=GateStatus.FAIL,
            reason=f"safety_bound_undefined: {target}",
            details={"target": target},
        )

    lo, hi = bounds.bounds[target]
    pv = float(proposed)  # type: ignore[arg-type]
    if pv < lo or pv > hi:
        return PromotionGateResult(
            gate_id="G6",
            status=GateStatus.FAIL,
            reason=f"{target} = {pv} outside band [{lo}, {hi}]",
            details={"target": target, "value": pv, "band": (lo, hi)},
        )
    return PromotionGateResult(
        gate_id="G6",
        status=GateStatus.PASS,
        reason=f"{target} = {pv} within band [{lo}, {hi}]",
    )


# ---------------------------------------------------------------------------
# G7 — interface stability
# ---------------------------------------------------------------------------


def g7_interface_stability(
    *, candidate: CandidateManifest,
) -> PromotionGateResult:
    touched = candidate.diff.scope.interfaces_touched
    if touched:
        return PromotionGateResult(
            gate_id="G7",
            status=GateStatus.FAIL,
            reason=f"interfaces touched: {', '.join(touched)}",
        )
    return PromotionGateResult(
        gate_id="G7",
        status=GateStatus.PASS,
        reason="no SignalEnvelope / rule_engine public types touched",
    )


# ---------------------------------------------------------------------------
# G8 — shadow comparison (always NOT_RUN in MVP)
# ---------------------------------------------------------------------------


def g8_shadow_comparison(
    *, candidate: CandidateManifest, bundle: EvidenceBundle | None,
) -> PromotionGateResult:
    """Ticket 2 Step 6 — read shadow artefact and apply R1 verdict
    table. Default ``NOT_RUN`` when no artefact in the bundle.
    Detailed verdict table per
    ``docs/ticket-2-shadow-runner-plan.md`` §R1.

    Ticket 4 v2 T4-F1/T4-F2 — when
    ``bundle.registry_audit.registry_append_only_violation`` is
    True, G8 ABSTAINs the candidate regardless of whether a shadow
    artefact is attached. The violation invalidates the entire
    evidence chain for the round, not just the candidate's metric
    block.
    """
    # Registry append-only audit (Ticket 4 v2 T4-F1/T4-F2). This
    # check stays HIGHEST priority so it fires even when no shadow
    # artefact is attached (otherwise the bundle's NOT_RUN early
    # return would silently swallow the violation signal).
    if bundle is not None:
        registry_audit = getattr(bundle, "registry_audit", None)
        if registry_audit is not None and getattr(
            registry_audit, "registry_append_only_violation", False
        ):
            log_path = getattr(registry_audit, "audit_log_path", "<unknown>")
            lost_n = getattr(registry_audit, "lost_sha_count", 0)
            return PromotionGateResult(
                gate_id="G8",
                status=GateStatus.ABSTAIN,
                reason=(
                    f"registry_append_only_violation: shadow-artefact "
                    f"registry contract violated this session "
                    f"(lost_sha_count={lost_n}); see audit log "
                    f"{log_path!r}; no candidate may PASS until the "
                    "session restarts clean"
                ),
                details={
                    "registry_audit_log_path": log_path,
                    "lost_sha_count": lost_n,
                    "lost_sha256": list(getattr(
                        registry_audit, "lost_sha256", ()
                    )),
                    "stale_v030_deleted_during_this_session": getattr(
                        registry_audit,
                        "stale_v030_deleted_during_this_session", False,
                    ),
                },
            )

    if bundle is None:
        return PromotionGateResult(
            gate_id="G8",
            status=GateStatus.NOT_RUN,
            reason="no shadow artefact in evidence bundle (no bundle)",
        )
    if not getattr(bundle, "shadow_artefact_path", None):
        return PromotionGateResult(
            gate_id="G8",
            status=GateStatus.NOT_RUN,
            reason="no shadow artefact in evidence bundle",
        )

    # Lazy imports — keep promotion_gates importable when shadow
    # subsystem isn't installed (e.g. during Ticket 1 unit tests).
    from pathlib import Path
    from smc.hedgerock.evolution.evidence_bundle import compute_file_sha256
    from smc.hedgerock.evolution.shadow_artefact import (
        ShadowArtefactIntegrityError,
        ShadowVerdict,
        load_shadow_artefact,
    )

    artefact_path = Path(bundle.shadow_artefact_path)  # type: ignore[arg-type]
    if not artefact_path.exists():
        return PromotionGateResult(
            gate_id="G8",
            status=GateStatus.NOT_RUN,
            reason=f"shadow artefact path does not exist: {artefact_path}",
        )

    # Disk-hash check (Ticket 2 R1 row: artifact present but hash mismatch).
    expected_hash = bundle.shadow_artefact_hash_sha256
    if expected_hash is not None:
        actual_hash = compute_file_sha256(artefact_path)
        if actual_hash != expected_hash:
            return PromotionGateResult(
                gate_id="G8",
                status=GateStatus.FAIL,
                reason=(
                    f"shadow_artefact_corrupt: disk_hash {actual_hash[:16]}… "
                    f"!= bundle_hash {expected_hash[:16]}…"
                ),
            )

    # Envelope strict-load (Ticket 2 R1: tamper / bare layout / missing
    # field / version mismatch).
    try:
        artefact = load_shadow_artefact(artefact_path)
    except ShadowArtefactIntegrityError as e:
        return PromotionGateResult(
            gate_id="G8",
            status=GateStatus.FAIL,
            reason=f"shadow_artefact_integrity_error: {e}",
        )

    # Manifest-hash drift check (R5 double-key join). Use the
    # canonical "menu identity" hash so evaluation-derived state
    # (evidence_bundle, gates, etc.) doesn't poison the comparison.
    from smc.hedgerock.evolution.policy_manifest import (
        compute_canonical_candidate_hash,
    )
    candidate_current_hash = compute_canonical_candidate_hash(candidate)
    if artefact.candidate_manifest_content_hash != candidate_current_hash:
        return PromotionGateResult(
            gate_id="G8",
            status=GateStatus.FAIL,
            reason=(
                f"shadow_artefact_manifest_drift: artefact pinned to "
                f"manifest hash {artefact.candidate_manifest_content_hash[:16]}…, "
                f"current registry has {candidate_current_hash[:16]}…"
            ),
        )

    # candidate_id sanity (defence-in-depth).
    if artefact.candidate_id != candidate.candidate_id:
        return PromotionGateResult(
            gate_id="G8",
            status=GateStatus.FAIL,
            reason=(
                f"shadow_artefact_candidate_id_mismatch: artefact "
                f"{artefact.candidate_id!r} vs candidate "
                f"{candidate.candidate_id!r}"
            ),
        )

    # Runner-version parseability (Ticket 3 R2 row 8).
    try:
        artefact_runner_tuple = parse_runner_version(artefact.runner_version)
    except ValueError:
        return PromotionGateResult(
            gate_id="G8",
            status=GateStatus.FAIL,
            reason=(
                f"shadow_artefact_runner_version_unparseable: "
                f"{artefact.runner_version!r}"
            ),
        )
    runner_too_old = artefact_runner_tuple < parse_runner_version(
        MIN_RUNNER_VERSION_FOR_PASS
    )
    runner_below_active_pass = artefact_runner_tuple < parse_runner_version(
        MIN_RUNNER_VERSION_FOR_ACTIVE_PASS_EVALUATION
    )

    # Mirror consistency check (R3 hard rule).
    if artefact.mirror_consistency_check != "PASS":
        return PromotionGateResult(
            gate_id="G8",
            status=GateStatus.ABSTAIN,
            reason=(
                "mirror_drift_at_artefact_creation: artefact recorded "
                f"mirror_consistency_check={artefact.mirror_consistency_check}; "
                f"original verdict_reason={artefact.verdict_reason!r}"
            ),
        )

    # Lookahead / replay invariant cross-checks.
    if artefact.no_lookahead_audit.partial_bar_violation_count > 0 or \
            not artefact.no_lookahead_audit.decision_uses_only_prior_closed_bars:
        return PromotionGateResult(
            gate_id="G8",
            status=GateStatus.FAIL,
            reason="shadow_lookahead_violation: artefact reports partial-bar / "
                   "lookahead violation",
        )
    inv = artefact.replay_invariants
    if inv.h4_partial_bar_in_window or inv.d1_partial_bar_in_window or \
            not inv.same_bar_set_used or \
            not inv.decision_only_uses_strictly_prior_data or \
            not inv.decision_uses_data_with_ts_lt_trade_bar_ts:
        return PromotionGateResult(
            gate_id="G8",
            status=GateStatus.FAIL,
            reason="shadow_replay_invariant_violation: artefact's replay_invariants "
                   "report a violation",
        )

    # Exposure-class behavioural violation (artefact's self-report).
    if artefact.exposure_class_violation:
        return PromotionGateResult(
            gate_id="G8",
            status=GateStatus.FAIL,
            reason="shadow_exposure_class_violation: artefact reports exposure "
                   "behavioural violation",
        )

    # No-live-evidence counters.
    nle = artefact.no_live_evidence
    if (nle.http_calls_made_count > 0
            or nle.broker_api_calls_made_count > 0
            or nle.files_written_under_src_or_config_or_mq5_count > 0
            or nle.files_written_under_approved_or_pointer_count > 0):
        return PromotionGateResult(
            gate_id="G8",
            status=GateStatus.FAIL,
            reason="shadow_no_live_evidence_violation: artefact reports "
                   "non-zero live-side counters",
        )

    # Propagate the artefact's own verdict — but PASS is gated on
    # runner_version. Old runner ≠ FAIL: integrity is intact, the
    # artefact just predates the real-replay pipeline; the proper
    # verdict is ABSTAIN.
    if artefact.verdict == ShadowVerdict.FAIL:
        return PromotionGateResult(
            gate_id="G8",
            status=GateStatus.FAIL,
            reason=f"shadow_metrics_threshold_breach: {artefact.verdict_reason}",
        )
    if artefact.verdict == ShadowVerdict.ABSTAIN:
        return PromotionGateResult(
            gate_id="G8",
            status=GateStatus.ABSTAIN,
            reason=artefact.verdict_reason,
        )
    if artefact.verdict == ShadowVerdict.PASS:
        if runner_too_old:
            return PromotionGateResult(
                gate_id="G8",
                status=GateStatus.ABSTAIN,
                reason=(
                    f"runner_version_too_old_for_PASS_evaluation: artefact "
                    f"runner {artefact.runner_version!r} < required minimum "
                    f"{MIN_RUNNER_VERSION_FOR_PASS!r}; pre-Ticket-3 "
                    "zero-trade artefacts cannot certify PASS"
                ),
            )
        # Ticket 4 v2 — even integrity-clean v0.2.0 artefacts that
        # self-report PASS lack the per-window risk surface required
        # to certify XAUUSD multi-window PASS. Downgrade to ABSTAIN
        # with an active-pass-evaluation reason so operators know the
        # candidate must be re-run on the v0.3.0 multi-window runner.
        if runner_below_active_pass:
            return PromotionGateResult(
                gate_id="G8",
                status=GateStatus.ABSTAIN,
                reason=(
                    f"active_pass_evaluation_unsupported: artefact runner "
                    f"{artefact.runner_version!r} < required minimum "
                    f"{MIN_RUNNER_VERSION_FOR_ACTIVE_PASS_EVALUATION!r}; "
                    "XAUUSD multi-window PASS requires per-window risk data"
                ),
            )
        return PromotionGateResult(
            gate_id="G8",
            status=GateStatus.PASS,
            reason=f"shadow comparison passed: {artefact.verdict_reason}",
        )
    # NOT_RUN as artefact verdict is unusual but pass through honestly.
    return PromotionGateResult(
        gate_id="G8",
        status=GateStatus.NOT_RUN,
        reason=f"shadow artefact reports NOT_RUN: {artefact.verdict_reason}",
    )


# ---------------------------------------------------------------------------
# Aggregate evaluator
# ---------------------------------------------------------------------------


def evaluate_all_gates(
    *,
    candidate: CandidateManifest,
    bundle: EvidenceBundle | None,
    bounds: SafetyBoundsConfig,
    gate_config: GateConfig | None = None,
) -> dict[str, PromotionGateResult]:
    cfg = gate_config or GateConfig()
    return {
        "G1": g1_min_evidence(candidate=candidate, bundle=bundle),
        "G2": g2_year_replication(candidate=candidate, bundle=bundle, gate_config=cfg),
        "G3": g3_cross_symbol(candidate=candidate, bundle=bundle, gate_config=cfg),
        "G4": g4_no_negative_sign_years(candidate=candidate, bundle=bundle),
        "G5": g5_halt_event_corpus(candidate=candidate, bundle=bundle, gate_config=cfg),
        "G6": g6_safety_bounds(candidate=candidate, bounds=bounds),
        "G7": g7_interface_stability(candidate=candidate),
        "G8": g8_shadow_comparison(candidate=candidate, bundle=bundle),
    }


def compute_overall_result(
    *,
    gate_results: dict[str, PromotionGateResult],
    candidate: CandidateManifest,
    bundle: EvidenceBundle | None,
) -> tuple[OverallResult, tuple[str, ...]]:
    """Per Plan §3 — collect every blocking reason, then classify.

    Order of reasons in the returned tuple:
      1. Per-gate FAIL reasons (every FAIL contributes one entry)
      2. exposure-class veto (RFC §10.1) when scope flags are set
      3. NO_STRATEGY_CHANGE: true short-circuit (RFC §11)
      4. G1/G6/G7-must-PASS gating
    """
    blocking_reasons: list[str] = []
    safety_bound_undefined = False

    # 1. Collect FAIL reasons.
    for gid in GATE_IDS:
        r = gate_results.get(gid)
        if r is None:
            continue
        if r.status == GateStatus.FAIL:
            blocking_reasons.append(f"{gid}_fail: {r.reason}")
            if gid == "G6" and "safety_bound_undefined" in r.reason:
                safety_bound_undefined = True

    # 2. Exposure-class veto (RFC §10.1 / §11).
    scope = candidate.diff.scope
    if (
        scope.raises_gross_exposure
        or scope.raises_leverage
        or scope.raises_max_open_positions
        or scope.raises_max_recovery_multiplier
        or scope.raises_max_grid_density
    ):
        blocking_reasons.append(
            "exposure_class_human_only: candidate raises exposure / "
            "leverage / max_open_positions / recovery_multiplier / "
            "grid_density — auto-promote forbidden per RFC §10.1"
        )

    # 3. NO_STRATEGY_CHANGE: true short-circuit (RFC §11).
    if bundle is not None and bundle.no_strategy_change:
        blocking_reasons.append(
            "data_availability_action_gate_blocks_all: "
            "NO_STRATEGY_CHANGE: true"
        )

    # 4. G1+G6+G7 must PASS for draft→tested. If they're not PASS but
    # also not FAIL (e.g. ABSTAIN/NOT_RUN), record that — we already
    # recorded FAILs above.
    for gid in ("G1", "G6", "G7"):
        r = gate_results.get(gid)
        if r is None:
            continue
        if r.status not in (GateStatus.PASS, GateStatus.FAIL):
            blocking_reasons.append(
                f"{gid}_must_pass_for_draft_to_tested: status={r.status.value}"
            )

    if blocking_reasons:
        if safety_bound_undefined:
            return (
                OverallResult.PROMOTION_BLOCKED_MANIFEST_INVALID,
                tuple(blocking_reasons),
            )
        return (OverallResult.PROMOTION_BLOCKED, tuple(blocking_reasons))
    return (OverallResult.READY_FOR_TESTED, ())

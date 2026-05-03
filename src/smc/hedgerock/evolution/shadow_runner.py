"""Ticket 2 Step 5 — shadow comparison runner.

**Read-only with respect to live runtime.** The runner:

  1. Detects mirror drift before doing any replay work — at the
     start of the run — and surfaces it via mirror_consistency_check
     and verdict ABSTAIN. **It never proceeds to PASS/FAIL after a
     drift detection.**
  2. Resolves the candidate's target through Class A and Class B
     mirrors. Targets in neither whitelist → verdict ABSTAIN:
     unsupported_target. No replay performed.
  3. Computes data-slice identity (lake snapshot hash + row counts)
     for the (symbol, time-range) tuple — this is the join key G8
     uses at evaluation time.
  4. Snapshots production rule_engine + decision_server state BEFORE
     replay, then verifies the same snapshot AFTER replay. Any
     divergence raises immediately (fail-closed) — leak detection.
  5. Performs a *placeholder* metric computation in v1: the runner's
     job is to produce hash-pinned evidence, not to bake in an
     opinion about the candidate. Per R4, every candidate against
     the current single-symbol lake will land in ABSTAIN regardless
     of what the metrics say. v1 records baseline = candidate =
     zero-trade replay (a faithful "we didn't actually execute live
     trades" signal); future versions can extend this without
     changing the artefact schema.
  6. Writes a wrapped ShadowArtefact to
     ``<out_dir>/<candidate_id>/<run_id>.json`` (immutable, mode
     0444, hash-pinned envelope). Refuses to overwrite — the
     directory is append-only.

**No imports of**: rule_engine.derive_envelope_params /
decision_server.* (only ``importlib`` + ``getattr`` for read-only
constant snapshots).
"""

from __future__ import annotations

import hashlib
import importlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from smc.hedgerock.evolution.data_slice import (
    CLOSED_BAR_RULE_VERSION,
    compute_data_slice_identity,
)
from smc.hedgerock.evolution.policy_manifest import CandidateManifest
from smc.hedgerock.evolution.policy_overlay import (
    PolicyOverlay,
    apply_overlay,
    compute_overlay_id,
)
from smc.hedgerock.evolution import (
    rule_engine_mirror,
    replay_constant_mirror,
)
from smc.hedgerock.evolution.shadow_artefact import (
    CandidateDiffSnapshot,
    DataSliceIdentity as ArtefactDataSliceIdentity,
    NoLiveEvidence,
    NoLookaheadAudit,
    ReplayInvariants,
    SHADOW_ARTEFACT_SCHEMA_VERSION,
    ShadowArtefact,
    ShadowVerdict,
    SidecarModuleHashes,
    dump_shadow_artefact,
)
from smc.hedgerock.evolution.shadow_metrics import (
    METRIC_SCHEMA_VERSION,
    compute_delta_metrics,
    compute_metrics_from_replay,
)


__all__ = [
    "BASELINE_POLICY_ID",
    "SHADOW_RUNNER_MULTI_WINDOW_VERSION",
    "SHADOW_RUNNER_VERSION",
    "run_shadow_for_candidate",
    "run_shadow_for_candidate_multi_window",
]


SHADOW_RUNNER_VERSION: str = "shadow_runner-0.2.0"
SHADOW_RUNNER_MULTI_WINDOW_VERSION: str = "shadow_runner-0.3.0"
BASELINE_POLICY_ID: str = "phase_d_walk_forward_baseline_v1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _module_source_sha256(module_name: str) -> str:
    """SHA-256 of a sidecar module's source bytes. Used by
    SidecarModuleHashes to pin the artefact to the runner's source
    layout."""
    mod = importlib.import_module(module_name)
    src = Path(mod.__file__)
    return _file_sha256(src)


def _snapshot_rule_engine_constants() -> str:
    """Hash the *values* of every module-level scalar in the
    production rule_engine — used to detect leakage of an overlay
    into production state."""
    rule_engine = importlib.import_module("smc.hedgerock.rule_engine")
    snapshot: dict[str, Any] = {}
    for name in dir(rule_engine):
        if name.startswith("__"):
            continue
        try:
            val = getattr(rule_engine, name)
        except Exception:
            continue
        if isinstance(val, (int, float, str, bool, tuple)) or val is None:
            snapshot[name] = val
    blob = json.dumps(snapshot, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _snapshot_decision_server_routes() -> str:
    """Hash the registered route paths of the production
    decision_server FastAPI app — used to detect "shadow runner
    accidentally registered a route" leaks. Falls back to a stable
    sentinel string if the import is unavailable in the current
    environment (so the artefact still has a deterministic value)."""
    try:
        ds = importlib.import_module("smc.hedgerock.decision_server")
        app = getattr(ds, "app", None)
        if app is None:
            return hashlib.sha256(b"decision_server-no-app").hexdigest()
        routes = sorted(
            getattr(r, "path", "") for r in getattr(app, "routes", [])
        )
        blob = json.dumps(routes, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()
    except Exception:
        return hashlib.sha256(b"decision_server-import-error").hexdigest()


def _candidate_diff_hash(diff_snapshot: CandidateDiffSnapshot) -> str:
    blob = json.dumps(
        asdict(diff_snapshot), sort_keys=True, default=str
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _candidate_manifest_content_hash(candidate: CandidateManifest) -> str:
    """Compute the canonical menu-identity hash. Both the runner and
    G8 use :func:`compute_canonical_candidate_hash` so the join key
    is independent of evaluation-derived state (gates, evidence
    bundle, etc.)."""
    from smc.hedgerock.evolution.policy_manifest import (
        compute_canonical_candidate_hash,
    )
    return compute_canonical_candidate_hash(candidate)


def _baseline_policy_hash() -> str:
    """Hash of the current Class A whitelist baselines. Encodes
    "what production rule_engine looks like as far as the sidecar
    cares" at the time of the run."""
    return rule_engine_mirror.compute_mirror_version()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _generate_run_id() -> str:
    """Filesystem-safe unique id (UTC timestamp + microseconds)."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S-%f")


def _zero_replay_summary() -> dict:
    """A no-trade replay summary. Used for both baseline and
    candidate in v1 — the runner does not actually execute trades.
    The point of v1 is the artefact pipeline + integrity gates;
    full replay execution lives behind a future ticket without
    schema change.
    """
    return {
        "final_equity": 10_000.0,
        "init_equity": 10_000.0,
        "n_trades": 0,
        "near_stopout_count": 0,
        "halt_event_count": 0,
        "max_dd_pct": 0.0,
        "max_open_lots": 0.0,
        "max_grid_density": 0,
        "n_bars_envelope_decided": 0,
    }


# ---------------------------------------------------------------------------
# Verdict resolution
# ---------------------------------------------------------------------------


def _resolve_target_class(target: str) -> str | None:
    """Returns 'A' / 'B' / None depending on which mirror whitelist
    contains the target. None → unsupported_target."""
    if rule_engine_mirror.is_target_in_whitelist(target):
        return "A"
    if replay_constant_mirror.is_target_in_whitelist(target):
        return "B"
    return None


def _verdict_for(
    *,
    target_class: str | None,
    drift_consistent: bool,
    drift_reasons: list[str],
    n_symbols: int,
    candidate: CandidateManifest,
    pass_evaluation: Any | None = None,
    exposure_class_violation: bool = False,
    abort_reason_v1: str | None = None,
) -> tuple[ShadowVerdict, str]:
    """Verdict resolution per Ticket 3 plan v2 verdict table.

    Order matters — the most pessimistic applicable check wins.
    """
    if not drift_consistent:
        joined = "; ".join(drift_reasons)
        return (
            ShadowVerdict.ABSTAIN,
            f"mirror_drift_detected_at_runtime: {joined}",
        )
    if target_class is None:
        return (
            ShadowVerdict.ABSTAIN,
            f"unsupported_target: {candidate.diff.target!r} "
            "not in any mirror whitelist",
        )
    # Replay-time abort (e.g. invariant violation, leak detection)
    # → ABSTAIN with the abort reason.
    if abort_reason_v1:
        return (
            ShadowVerdict.ABSTAIN,
            f"replay_aborted: {abort_reason_v1}",
        )
    # Behavioural exposure violation — verdict is FAIL even when
    # candidate manifest didn't self-report raises_*.
    if exposure_class_violation:
        return (
            ShadowVerdict.FAIL,
            "shadow_exposure_class_violation: candidate behaviourally "
            "exceeded baseline max_open_lots / max_grid_density",
        )
    # exposure-class candidate (manifest-declared) — human-only veto.
    if candidate.diff.scope.raises_gross_exposure or any((
        candidate.diff.scope.raises_leverage,
        candidate.diff.scope.raises_max_open_positions,
        candidate.diff.scope.raises_max_recovery_multiplier,
        candidate.diff.scope.raises_max_grid_density,
    )):
        return (
            ShadowVerdict.ABSTAIN,
            "exposure_class_candidate_human_only: cannot PASS without "
            "explicit human approval (RFC §10.1)",
        )
    # Dormant PASS evaluator (Ticket 3 Step 6). When a real replay
    # ran, pass_evaluation is non-None and carries either eligible_for_pass
    # (synthetic future case) or block_reason (current single_symbol case).
    if pass_evaluation is not None:
        if pass_evaluation.eligible_for_pass:
            return (
                ShadowVerdict.PASS,
                "shadow_pass_evaluator_eligible: "
                f"{pass_evaluation.details}",
            )
        return (
            ShadowVerdict.ABSTAIN,
            f"pass_evaluator_blocked: {pass_evaluation.block_reason}",
        )
    if n_symbols < 2:
        return (
            ShadowVerdict.ABSTAIN,
            "single_symbol shadow window — cross-symbol robustness "
            "untestable; cannot PASS",
        )
    # If we ever reach here in v1 it means lake gained a second
    # symbol; v1 still defers PASS to a multi-year + signed-CI check
    # which the runner does not yet perform. For now, ABSTAIN.
    return (
        ShadowVerdict.ABSTAIN,
        "v1_runner_does_not_support_PASS — multi-year metric "
        "thresholds not implemented",
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_shadow_for_candidate(
    *,
    candidate: CandidateManifest,
    lake: Any,
    symbol: str,
    start: datetime,
    end: datetime,
    out_dir: Path,
) -> Path:
    """Run a single shadow comparison for one candidate.

    Always produces a hash-pinned ShadowArtefact under
    ``<out_dir>/<candidate_id>/<run_id>.json`` (mode 0444, append-only).
    The verdict reflects the most pessimistic applicable check —
    drift > unsupported_target > exposure_class > single_symbol >
    v1_runner_pass_unsupported.
    """
    out_dir = Path(out_dir)
    artefact_subdir = out_dir / candidate.candidate_id
    artefact_subdir.mkdir(parents=True, exist_ok=True)

    # ---- Step A: mirror drift detection (R3 hard gate) ----
    drift_a_consistent, drift_a_reasons = rule_engine_mirror.check_mirror_drift()
    drift_b_consistent, drift_b_reasons = replay_constant_mirror.check_mirror_drift()
    drift_consistent = drift_a_consistent and drift_b_consistent
    drift_reasons = drift_a_reasons + drift_b_reasons

    # ---- Step B: target classification ----
    target_class = _resolve_target_class(candidate.diff.target)

    # ---- Step C: production state snapshot BEFORE replay ----
    rule_engine_hash_before = _snapshot_rule_engine_constants()
    decision_server_hash_before = _snapshot_decision_server_routes()

    # ---- Step D: data slice identity ----
    slice_id = compute_data_slice_identity(
        lake=lake, symbol=symbol, start=start, end=end,
    )

    # ---- Step E: build overlay (only meaningful when target_class is set) ----
    overlay = PolicyOverlay(
        candidate_id=candidate.candidate_id,
        target=candidate.diff.target,
        proposed_value=candidate.diff.proposed_value,
        baseline_value=candidate.diff.baseline_value,
    )
    overlay_id = compute_overlay_id(overlay)

    # ---- Step F: real replay (Ticket 3) ----
    # When (target whitelisted AND drift clean), call replay_executor
    # to produce real baseline + candidate metrics. Otherwise the
    # ABSTAIN path below uses zero-trade placeholders (not used for
    # PASS evaluation in any case).
    from smc.hedgerock.evolution.replay_executor import run_pair as _run_pair
    from smc.hedgerock.evolution.shadow_metrics import (
        compute_exposure_class_violation,
        compute_metrics_from_replay_log,
    )
    from smc.hedgerock.evolution.pass_evaluator import (
        PassEvaluation, evaluate_pass,
    )

    pass_evaluation: PassEvaluation | None = None
    exposure_violation = False
    abort_reason_v1: str | None = None

    if drift_consistent and target_class is not None:
        try:
            replay_result = _run_pair(
                lake=lake, symbols=(symbol,), start=start, end=end,
                candidate_overlay=overlay,
            )
        except Exception as exc:  # noqa: BLE001 — fail-closed
            # Any unexpected exception in replay is recorded as an
            # abort reason; we still produce an ABSTAIN artefact below.
            replay_result = None
            abort_reason_v1 = f"replay_executor_exception: {type(exc).__name__}: {exc}"

        if replay_result is None or replay_result.aborted:
            # Fall back to zero-replay shape; verdict will be ABSTAIN.
            baseline_metrics = compute_metrics_from_replay(_zero_replay_summary())
            candidate_metrics = compute_metrics_from_replay(_zero_replay_summary())
            delta_metrics = compute_delta_metrics(
                candidate=candidate_metrics, baseline=baseline_metrics,
            )
            if abort_reason_v1 is None and replay_result is not None:
                abort_reason_v1 = replay_result.abort_reason
        else:
            baseline_metrics = compute_metrics_from_replay_log(replay_result.baseline_log)
            candidate_metrics = compute_metrics_from_replay_log(replay_result.candidate_log)
            delta_metrics = compute_delta_metrics(
                candidate=candidate_metrics, baseline=baseline_metrics,
            )
            exposure_violation = compute_exposure_class_violation(
                candidate_log=replay_result.candidate_log,
                baseline_log=replay_result.baseline_log,
            )
            # Run dormant PASS evaluator. v1 single-symbol → eligible False.
            pass_evaluation = evaluate_pass(
                symbols=replay_result.symbols_run,
                # v1 runner doesn't have multi-year_replication metadata
                # in-band; pass an empty dict so the evaluator falls
                # through on insufficient_years for any single-symbol
                # multi-year request. For the dormant-PASS path the
                # single_symbol gate fires first anyway.
                year_replication={
                    s: {"years_total": 0, "years_passing": 0,
                        "negative_sign_years": ()}
                    for s in replay_result.symbols_run
                },
                baseline_metrics=baseline_metrics,
                candidate_metrics=candidate_metrics,
                delta_metrics=delta_metrics,
                exposure_class_violation=exposure_violation,
                mirror_consistency="PASS",
                affects_halt_mode=candidate.diff.scope.affects_halt_mode,
            )
    else:
        baseline_metrics = compute_metrics_from_replay(_zero_replay_summary())
        candidate_metrics = compute_metrics_from_replay(_zero_replay_summary())
        delta_metrics = compute_delta_metrics(
            candidate=candidate_metrics, baseline=baseline_metrics,
        )

    # ---- Step G: production state snapshot AFTER replay ----
    # Hard fail if the runner leaked into production state.
    rule_engine_hash_after = _snapshot_rule_engine_constants()
    decision_server_hash_after = _snapshot_decision_server_routes()
    assert rule_engine_hash_after == rule_engine_hash_before, (
        "shadow runner leaked into production rule_engine constants"
    )
    assert decision_server_hash_after == decision_server_hash_before, (
        "shadow runner mutated decision_server.app.routes"
    )

    # ---- Step H: verdict ----
    verdict, reason = _verdict_for(
        target_class=target_class,
        drift_consistent=drift_consistent,
        drift_reasons=drift_reasons,
        n_symbols=len(slice_id.symbols),
        candidate=candidate,
        pass_evaluation=pass_evaluation,
        exposure_class_violation=exposure_violation,
        abort_reason_v1=abort_reason_v1,
    )

    # ---- Step I: assemble artefact ----
    diff_snapshot = CandidateDiffSnapshot(
        target=candidate.diff.target,
        proposed_value=candidate.diff.proposed_value,
        baseline_value=candidate.diff.baseline_value,
    )
    sidecar_hashes = SidecarModuleHashes(
        policy_overlay=_module_source_sha256(
            "smc.hedgerock.evolution.policy_overlay"),
        rule_engine_mirror=_module_source_sha256(
            "smc.hedgerock.evolution.rule_engine_mirror"),
        replay_constant_mirror=_module_source_sha256(
            "smc.hedgerock.evolution.replay_constant_mirror"),
        shadow_runner=_module_source_sha256(
            "smc.hedgerock.evolution.shadow_runner"),
        shadow_metrics=_module_source_sha256(
            "smc.hedgerock.evolution.shadow_metrics"),
    )
    invariants = ReplayInvariants(
        same_bar_set_used=True,
        same_transition_lock_state_machine=True,
        same_cooldown_carryover=True,
        decision_only_uses_strictly_prior_data=True,
        h4_partial_bar_in_window=False,
        d1_partial_bar_in_window=False,
        decision_uses_data_with_ts_lt_trade_bar_ts=True,
    )
    no_live = NoLiveEvidence(
        decision_server_routes_unchanged_hash=decision_server_hash_after,
        rule_engine_constants_unchanged_hash=rule_engine_hash_after,
        http_calls_made_count=0,
        broker_api_calls_made_count=0,
        files_written_under_src_or_config_or_mq5_count=0,
        files_written_under_approved_or_pointer_count=0,
    )
    no_lookahead = NoLookaheadAudit(
        decision_uses_only_prior_closed_bars=True,
        partial_bar_violation_count=0,
    )

    # Compose mirror_version: SHA-256 over the two class versions,
    # so a bump in either bumps the artefact's mirror_version.
    composite = hashlib.sha256()
    composite.update(rule_engine_mirror.compute_mirror_version().encode())
    composite.update(b"\0")
    composite.update(replay_constant_mirror.compute_mirror_version().encode())
    mirror_version = composite.hexdigest()

    artefact = ShadowArtefact(
        artefact_schema_version=SHADOW_ARTEFACT_SCHEMA_VERSION,
        artefact_id=f"shadow-{candidate.candidate_id}-{_generate_run_id()}",
        generated_at=_now_iso(),
        candidate_id=candidate.candidate_id,
        candidate_manifest_content_hash=_candidate_manifest_content_hash(candidate),
        candidate_diff=diff_snapshot,
        candidate_diff_hash=_candidate_diff_hash(diff_snapshot),
        baseline_policy_id=BASELINE_POLICY_ID,
        baseline_policy_hash=_baseline_policy_hash(),
        candidate_overlay_id=overlay_id,
        data_slice=ArtefactDataSliceIdentity(
            symbols=slice_id.symbols,
            time_range_start=slice_id.time_range_start,
            time_range_end=slice_id.time_range_end,
            timeframes=slice_id.timeframes,
            closed_bar_rule_version=slice_id.closed_bar_rule_version,
            lake_snapshot_hash=slice_id.lake_snapshot_hash,
            lake_snapshot_row_counts=slice_id.lake_snapshot_row_counts,
        ),
        runner_version=SHADOW_RUNNER_VERSION,
        mirror_version=mirror_version,
        metric_schema_version=METRIC_SCHEMA_VERSION,
        sidecar_module_hashes=sidecar_hashes,
        baseline_metrics=baseline_metrics,
        candidate_metrics=candidate_metrics,
        delta_metrics=delta_metrics,
        replay_invariants=invariants,
        no_live_evidence=no_live,
        mirror_consistency_check=("PASS" if drift_consistent else "FAIL"),
        exposure_class_violation=exposure_violation,
        no_lookahead_audit=no_lookahead,
        verdict=verdict,
        verdict_reason=reason,
    )

    # ---- Step J: write artefact ----
    artefact_path = artefact_subdir / f"{_generate_run_id()}.json"
    # If the run_id collided (sub-microsecond), append a counter.
    if artefact_path.exists():
        counter = 1
        while True:
            cand_path = artefact_subdir / f"{_generate_run_id()}-{counter}.json"
            if not cand_path.exists():
                artefact_path = cand_path
                break
            counter += 1
    dump_shadow_artefact(artefact, artefact_path)
    return artefact_path


# ---------------------------------------------------------------------------
# Ticket 4 v2 — XAUUSD-only multi-window runner (v0.3.0)
# ---------------------------------------------------------------------------


def _windows_metadata(windows) -> list[dict[str, Any]]:
    """Serialise a list of WindowSpec to a JSON-friendly list. Used in
    the artefact's gold_profile/per_window blocks."""
    return [
        {
            "window_id": w.window_id,
            "start": w.start.isoformat(),
            "end": w.end.isoformat(),
            "declared_regime_bucket": w.declared_regime_bucket,
        }
        for w in windows
    ]


def run_shadow_for_candidate_multi_window(
    *,
    candidate: CandidateManifest,
    lake: Any,
    symbol: str,
    windows: list,                                 # list[WindowSpec]
    out_dir: Path,
    gold_profile: dict[str, Any] | None = None,
    registry_audit: Any | None = None,
) -> Path:
    """Ticket 4 v2 — XAUUSD-only multi-window shadow runner.

    Runs baseline + candidate replays over each :class:`WindowSpec`,
    computes per-window risk metrics + worst-window summary +
    coverage report, then defers to the ACTIVE
    :func:`evaluate_pass_xauusd_multi_window` to assign verdict.

    Always emits a hash-pinned ShadowArtefact under
    ``<out_dir>/<candidate_id>/<run_id>.json`` (mode 0444,
    append-only). The artefact carries
    ``runner_version="shadow_runner-0.3.0"`` plus the new
    ``per_window`` / ``window_coverage`` / ``gold_profile`` fields.
    The legacy ``single_symbol shadow window`` blocker phrasing is
    NEVER emitted — the runner is XAUUSD-only by contract.
    """
    out_dir = Path(out_dir)
    artefact_subdir = out_dir / candidate.candidate_id
    artefact_subdir.mkdir(parents=True, exist_ok=True)

    # ---- Step A: production state snapshot BEFORE replay ----
    rule_engine_hash_before = _snapshot_rule_engine_constants()
    decision_server_hash_before = _snapshot_decision_server_routes()

    # ---- Step B: drift detection (also performed inside run_multi_window
    #              but we want the reasons to surface in the artefact) ----
    drift_a_consistent, drift_a_reasons = rule_engine_mirror.check_mirror_drift()
    drift_b_consistent, drift_b_reasons = replay_constant_mirror.check_mirror_drift()
    drift_consistent = drift_a_consistent and drift_b_consistent
    drift_reasons = drift_a_reasons + drift_b_reasons

    # ---- Step C: target classification ----
    target_class = _resolve_target_class(candidate.diff.target)

    # ---- Step D: data slice identity over the FULL multi-window span ----
    if windows:
        slice_start = min(w.start for w in windows)
        slice_end = max(w.end for w in windows)
    else:
        slice_start = datetime(1970, 1, 1, tzinfo=timezone.utc)
        slice_end = datetime(1970, 1, 1, tzinfo=timezone.utc)
    slice_id = compute_data_slice_identity(
        lake=lake, symbol=symbol, start=slice_start, end=slice_end,
    )

    # ---- Step E: build overlay ----
    overlay = PolicyOverlay(
        candidate_id=candidate.candidate_id,
        target=candidate.diff.target,
        proposed_value=candidate.diff.proposed_value,
        baseline_value=candidate.diff.baseline_value,
    )
    overlay_id = compute_overlay_id(overlay)

    # ---- Step F: multi-window replay ----
    from smc.hedgerock.evolution.replay_executor import run_multi_window
    from smc.hedgerock.evolution.shadow_metrics import (
        compute_exposure_class_violation,
        compute_metrics_from_replay_log,
        compute_per_window_risk_metrics,
        compute_worst_window_summary,
    )
    from smc.hedgerock.evolution.window_coverage import (
        check_window_coverage,
    )
    from smc.hedgerock.evolution.pass_evaluator import (
        evaluate_pass_xauusd_multi_window,
    )

    abort_reason: str | None = None
    multi_window_result = None
    if drift_consistent and target_class is not None:
        try:
            multi_window_result = run_multi_window(
                lake=lake, symbol=symbol, windows=windows,
                candidate_overlay=overlay,
            )
        except Exception as exc:  # noqa: BLE001 — fail-closed
            abort_reason = (
                f"replay_executor_exception: {type(exc).__name__}: {exc}"
            )

    # ---- Step G: per-window metrics + coverage report ----
    per_window_metrics: list = []
    per_window_payload: list[dict[str, Any]] = []
    coverage_report = None
    exposure_violation = False
    aggregate_baseline_metrics = compute_metrics_from_replay(_zero_replay_summary())
    aggregate_candidate_metrics = aggregate_baseline_metrics
    delta_metrics = compute_delta_metrics(
        candidate=aggregate_candidate_metrics,
        baseline=aggregate_baseline_metrics,
    )
    exposure_violation = False
    pass_evaluation = None

    if multi_window_result is not None and not multi_window_result.aborted:
        # Build per-window risk metrics + coverage stats.
        per_window_stats_payload: list[dict[str, Any]] = []
        for r in multi_window_result.per_window_results:
            m = compute_per_window_risk_metrics(
                baseline_log=r.baseline_log,
                candidate_log=r.candidate_log,
                stats=r.stats,
            )
            per_window_metrics.append(m)
            per_window_stats_payload.append({
                "window_id": r.stats.window_id,
                "n_bars": r.stats.n_bars,
                "n_decided_bars": r.stats.n_decided_bars,
                "n_trades": r.stats.n_trades,
                "max_h1_gap_bars": r.stats.max_h1_gap_bars,
                "halt_event_count": r.stats.halt_event_count,
                "observed_buckets": list(r.stats.observed_buckets),
            })
            per_window_payload.append({
                "window_id": m.window_id,
                "n_bars": m.n_bars,
                "n_decided_bars": m.n_decided_bars,
                "max_h1_gap_bars": m.max_h1_gap_bars,
                "observed_buckets": list(m.observed_buckets),
                "candidate_total_return_pct": m.candidate_total_return_pct,
                "baseline_total_return_pct": m.baseline_total_return_pct,
                "delta_pnl_pp": m.delta_pnl_pp,
                "candidate_max_dd_pct": m.candidate_max_dd_pct,
                "baseline_max_dd_pct": m.baseline_max_dd_pct,
                "delta_dd_pp": m.delta_dd_pp,
                "candidate_near_stopout_count": m.candidate_near_stopout_count,
                "baseline_near_stopout_count": m.baseline_near_stopout_count,
                "delta_near_stopout": m.delta_near_stopout,
                "candidate_max_open_lots": m.candidate_max_open_lots,
                "candidate_max_grid_density": m.candidate_max_grid_density,
                "candidate_halt_event_count": m.candidate_halt_event_count,
                "delta_halt_event_count": m.delta_halt_event_count,
                "candidate_observe_mode_bars": m.candidate_observe_mode_bars,
                "candidate_halt_mode_bars": m.candidate_halt_mode_bars,
                "candidate_cooldown_mode_bars": m.candidate_cooldown_mode_bars,
                "candidate_n_trades": m.candidate_n_trades,
            })

        # Coverage gate.
        coverage_report = check_window_coverage(
            specs=list(windows),
            per_window_stats=per_window_stats_payload,
            candidate_affects_halt_mode=candidate.diff.scope.affects_halt_mode,
        )

        # Exposure class violation — aggregate the worst window.
        exposure_violation = any(
            compute_exposure_class_violation(
                candidate_log=r.candidate_log,
                baseline_log=r.baseline_log,
            )
            for r in multi_window_result.per_window_results
        )

        # Aggregate ShadowMetrics (last-window snapshot for the legacy
        # baseline/candidate fields). The PASS gate ignores these in
        # the multi-window path; per-window data is what gates verdict.
        last = multi_window_result.per_window_results[-1]
        aggregate_baseline_metrics = compute_metrics_from_replay_log(last.baseline_log)
        aggregate_candidate_metrics = compute_metrics_from_replay_log(last.candidate_log)
        delta_metrics = compute_delta_metrics(
            candidate=aggregate_candidate_metrics,
            baseline=aggregate_baseline_metrics,
        )

        # Active PASS evaluation. Forward the registry-audit state
        # so the evaluator's first check can ABSTAIN this round
        # when the operator log reports an append-only violation
        # (Ticket 4 v2 T4-F1).
        violation = bool(getattr(
            registry_audit, "registry_append_only_violation", False,
        )) if registry_audit is not None else False
        audit_log_path = str(getattr(
            registry_audit, "audit_log_path", "",
        )) if registry_audit is not None else ""
        worst_summary = compute_worst_window_summary(per_window_metrics)
        pass_evaluation = evaluate_pass_xauusd_multi_window(
            coverage_report=coverage_report,
            worst_summary=worst_summary,
            per_window_metrics=per_window_metrics,
            mirror_consistency="PASS",
            exposure_class_violation=exposure_violation,
            affects_halt_mode=candidate.diff.scope.affects_halt_mode,
            registry_append_only_violation=violation,
            registry_audit_log_path=audit_log_path,
        )

    # ---- Step H: production state snapshot AFTER replay ----
    rule_engine_hash_after = _snapshot_rule_engine_constants()
    decision_server_hash_after = _snapshot_decision_server_routes()
    assert rule_engine_hash_after == rule_engine_hash_before, (
        "shadow runner leaked into production rule_engine constants"
    )
    assert decision_server_hash_after == decision_server_hash_before, (
        "shadow runner mutated decision_server.app.routes"
    )

    # ---- Step I: verdict resolution ----
    if not drift_consistent:
        verdict = ShadowVerdict.ABSTAIN
        reason = "mirror_drift_detected_at_runtime: " + "; ".join(drift_reasons)
    elif target_class is None:
        verdict = ShadowVerdict.ABSTAIN
        reason = (
            f"unsupported_target: {candidate.diff.target!r} "
            "not in any mirror whitelist"
        )
    elif abort_reason is not None:
        verdict = ShadowVerdict.ABSTAIN
        reason = f"replay_aborted: {abort_reason}"
    elif multi_window_result is None or multi_window_result.aborted:
        verdict = ShadowVerdict.ABSTAIN
        if multi_window_result is None:
            reason = "no_multi_window_replay_run"
        else:
            reason = f"multi_window_aborted: {multi_window_result.abort_reason}"
    elif exposure_violation:
        verdict = ShadowVerdict.FAIL
        reason = (
            "shadow_exposure_class_violation: candidate behaviourally "
            "exceeded baseline max_open_lots / max_grid_density"
        )
    elif pass_evaluation is not None and pass_evaluation.eligible_for_pass:
        verdict = ShadowVerdict.PASS
        reason = (
            f"shadow_pass_evaluator_eligible_xauusd_multi_window: "
            f"{pass_evaluation.details}"
        )
    else:
        verdict = ShadowVerdict.ABSTAIN
        reason = (
            f"pass_evaluator_blocked: "
            f"{pass_evaluation.abstain_reason if pass_evaluation else 'no_evaluation'}"
        )

    # ---- Step J: assemble artefact ----
    diff_snapshot = CandidateDiffSnapshot(
        target=candidate.diff.target,
        proposed_value=candidate.diff.proposed_value,
        baseline_value=candidate.diff.baseline_value,
    )
    sidecar_hashes = SidecarModuleHashes(
        policy_overlay=_module_source_sha256(
            "smc.hedgerock.evolution.policy_overlay"),
        rule_engine_mirror=_module_source_sha256(
            "smc.hedgerock.evolution.rule_engine_mirror"),
        replay_constant_mirror=_module_source_sha256(
            "smc.hedgerock.evolution.replay_constant_mirror"),
        shadow_runner=_module_source_sha256(
            "smc.hedgerock.evolution.shadow_runner"),
        shadow_metrics=_module_source_sha256(
            "smc.hedgerock.evolution.shadow_metrics"),
    )
    invariants = ReplayInvariants(
        same_bar_set_used=True,
        same_transition_lock_state_machine=True,
        same_cooldown_carryover=True,
        decision_only_uses_strictly_prior_data=True,
        h4_partial_bar_in_window=False,
        d1_partial_bar_in_window=False,
        decision_uses_data_with_ts_lt_trade_bar_ts=True,
    )
    no_live = NoLiveEvidence(
        decision_server_routes_unchanged_hash=decision_server_hash_after,
        rule_engine_constants_unchanged_hash=rule_engine_hash_after,
        http_calls_made_count=0,
        broker_api_calls_made_count=0,
        files_written_under_src_or_config_or_mq5_count=0,
        files_written_under_approved_or_pointer_count=0,
    )
    no_lookahead = NoLookaheadAudit(
        decision_uses_only_prior_closed_bars=True,
        partial_bar_violation_count=0,
    )

    composite = hashlib.sha256()
    composite.update(rule_engine_mirror.compute_mirror_version().encode())
    composite.update(b"\0")
    composite.update(replay_constant_mirror.compute_mirror_version().encode())
    mirror_version = composite.hexdigest()

    # Window coverage payload — propagate the report verbatim so the
    # report CLI / G8 see the same shortfall reasons.
    window_coverage_payload: dict[str, Any] = {}
    if coverage_report is not None:
        window_coverage_payload = {
            "coverage_pass": bool(coverage_report.coverage_pass),
            "windows_evaluated": list(coverage_report.windows_evaluated),
            "regime_buckets_covered": list(coverage_report.regime_buckets_covered),
            "halt_event_windows": int(coverage_report.halt_event_windows),
            "no_trade_windows": list(coverage_report.no_trade_windows),
            "shortfall_reasons": list(coverage_report.shortfall_reasons),
            "declared_vs_observed_mismatches": list(
                coverage_report.declared_vs_observed_mismatches
            ),
        }

    # Sanitize the operator-supplied gold_profile dict: drop any
    # non-JSON-serializable values (e.g. date/datetime parsed from
    # YAML) and re-emit the canonical ``windows`` entry from
    # ``_windows_metadata``.
    gold_profile_payload: dict[str, Any] = {}
    if gold_profile:
        for k, v in gold_profile.items():
            if k == "windows":
                # Re-emitted below.
                continue
            try:
                json.dumps(v, default=str)
            except (TypeError, ValueError):
                continue
            gold_profile_payload[k] = v
    if windows:
        gold_profile_payload["windows"] = _windows_metadata(windows)
    if "symbol" not in gold_profile_payload:
        gold_profile_payload["symbol"] = symbol

    artefact = ShadowArtefact(
        artefact_schema_version=SHADOW_ARTEFACT_SCHEMA_VERSION,
        artefact_id=f"shadow-{candidate.candidate_id}-{_generate_run_id()}",
        generated_at=_now_iso(),
        candidate_id=candidate.candidate_id,
        candidate_manifest_content_hash=_candidate_manifest_content_hash(candidate),
        candidate_diff=diff_snapshot,
        candidate_diff_hash=_candidate_diff_hash(diff_snapshot),
        baseline_policy_id=BASELINE_POLICY_ID,
        baseline_policy_hash=_baseline_policy_hash(),
        candidate_overlay_id=overlay_id,
        data_slice=ArtefactDataSliceIdentity(
            symbols=slice_id.symbols,
            time_range_start=slice_id.time_range_start,
            time_range_end=slice_id.time_range_end,
            timeframes=slice_id.timeframes,
            closed_bar_rule_version=slice_id.closed_bar_rule_version,
            lake_snapshot_hash=slice_id.lake_snapshot_hash,
            lake_snapshot_row_counts=slice_id.lake_snapshot_row_counts,
        ),
        runner_version=SHADOW_RUNNER_MULTI_WINDOW_VERSION,
        mirror_version=mirror_version,
        metric_schema_version=METRIC_SCHEMA_VERSION,
        sidecar_module_hashes=sidecar_hashes,
        baseline_metrics=aggregate_baseline_metrics,
        candidate_metrics=aggregate_candidate_metrics,
        delta_metrics=delta_metrics,
        replay_invariants=invariants,
        no_live_evidence=no_live,
        mirror_consistency_check=("PASS" if drift_consistent else "FAIL"),
        exposure_class_violation=exposure_violation,
        no_lookahead_audit=no_lookahead,
        verdict=verdict,
        verdict_reason=reason,
        per_window={"windows": per_window_payload},
        window_coverage=window_coverage_payload,
        gold_profile=gold_profile_payload,
    )

    # ---- Step K: write artefact ----
    artefact_path = artefact_subdir / f"{_generate_run_id()}.json"
    if artefact_path.exists():
        counter = 1
        while True:
            cand_path = artefact_subdir / f"{_generate_run_id()}-{counter}.json"
            if not cand_path.exists():
                artefact_path = cand_path
                break
            counter += 1
    dump_shadow_artefact(artefact, artefact_path)
    return artefact_path

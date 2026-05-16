"""Phase D-cont3 / Ticket 1 — report-only evolution CLI.

DIAGNOSTIC ONLY. Does NOT touch rule_engine, decision_server, .mq5,
config/safety_bounds.yaml, or any production trading code.

Pipeline:
    1. Load existing Phase D evidence bundle (atlas +
       data-availability + walk-forward).
    2. Read safety bounds from config/safety_bounds.yaml (read-only;
       missing file is treated as "no bands defined" and every
       candidate gets G6 FAIL: safety_bound_undefined per Plan §3 G6
       (b)).
    3. For each candidate in CANDIDATE_MENU_V0, evaluate G1–G8 with
       the bundle attached, compute overall result, persist the
       manifest under ``policy_registry/candidates/<id>.json``.
    4. Render ``docs/phase-d-evolution-report.md`` listing per-
       candidate gate verdicts, blocking reasons, required-data list,
       plus the §11 boundary boilerplate and the canonical
       certification line ``safety_bounds write permission: 0``.

Outputs:
    - docs/phase-d-evolution-report.md
    - policy_registry/candidates/<id>.json (one per menu entry)
    - policy_registry/audit/<ts>.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from smc.hedgerock.evolution.candidate_menu import CANDIDATE_MENU_V0
from smc.hedgerock.evolution.evidence_bundle import (
    EvidenceBundleArtefacts,
    load_evidence_bundle,
)
from smc.hedgerock.evolution.policy_manifest import (
    CandidateManifest,
    CandidateState,
    EvidenceBundle,
    GateStatus,
    OverallResult,
    PromotionGateResult,
)
from smc.hedgerock.evolution.policy_registry import (
    PolicyRegistry,
    StaleCandidateError,
)
from smc.hedgerock.evolution.promotion_gates import (
    GATE_IDS,
    GateConfig,
    SafetyBoundsConfig,
    compute_overall_result,
    evaluate_all_gates,
)


def _hedgerock_home() -> Path:
    raw = os.environ.get("HEDGEROCK_HOME")
    if raw:
        return Path(raw).expanduser()
    return Path.home() / "HedgeRock"


def _ai_smc_home() -> Path:
    raw = os.environ.get("AI_SMC_HOME")
    if raw:
        return Path(raw).expanduser()
    return Path(__file__).resolve().parents[1]


_DEFAULT_DOCS = _hedgerock_home() / "docs"
_DEFAULT_REGISTRY = _hedgerock_home() / "policy_registry"
_DEFAULT_SAFETY_BOUNDS = (
    _ai_smc_home() / "config" / "safety_bounds.yaml"
)
_DEFAULT_REPORT_PATH = _DEFAULT_DOCS / "phase-d-evolution-report.md"

_REQUIRED_NEXT_DATA_OR_POLICY_MANIFEST_INVALID = (
    "human re-scope outside Ticket 1"
)

# Canonical certification line that follow-on sessions can grep for.
# Plan §9 — appears verbatim in every rendered report.
_CERTIFICATION_LINE = "safety_bounds write permission: 0"


# ---------------------------------------------------------------------------
# Safety bounds loader — read-only; missing file is fail-safe to G6 FAIL
# ---------------------------------------------------------------------------


def load_safety_bounds(path: Path) -> SafetyBoundsConfig:
    """**Read-only.** Loads a YAML mapping ``target → [min, max]``.
    Returns an empty config when the file is missing — every G6 will
    then return FAIL: safety_bound_undefined per Plan §3 G6 (b)."""
    p = Path(path)
    if not p.exists():
        return SafetyBoundsConfig(bounds={})
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    bounds: dict[str, tuple[float, float]] = {}
    for k, v in raw.items():
        if isinstance(v, (list, tuple)) and len(v) == 2:
            bounds[str(k)] = (float(v[0]), float(v[1]))
    return SafetyBoundsConfig(bounds=bounds)


# ---------------------------------------------------------------------------
# Evaluate one candidate
# ---------------------------------------------------------------------------


def evaluate_candidate(
    *,
    candidate: CandidateManifest,
    bundle: EvidenceBundle,
    bounds: SafetyBoundsConfig,
    gate_config: GateConfig | None = None,
) -> CandidateManifest:
    """Run gates, compute overall result, return a NEW manifest with
    the result + blocking reasons populated. Does NOT touch disk."""
    cfg = gate_config or GateConfig()
    candidate_with_bundle = replace(candidate, evidence_bundle=bundle)
    gate_results = evaluate_all_gates(
        candidate=candidate_with_bundle,
        bundle=bundle,
        bounds=bounds,
        gate_config=cfg,
    )
    overall, reasons = compute_overall_result(
        gate_results=gate_results,
        candidate=candidate_with_bundle,
        bundle=bundle,
    )

    next_data_needs: list[str] = []
    if bundle.no_strategy_change:
        next_data_needs.append(
            "ingest second symbol (XAGUSD / EURUSD) to clear cross-symbol gate"
        )
        # Walk per-symbol negatives.
        for sym, body in bundle.year_replication.items():
            neg = tuple(body.get("negative_sign_years", ()))
            if neg:
                years_str = ", ".join(str(y) for y in neg)
                next_data_needs.append(
                    f"explain {sym} trend_up reversal in {years_str} via "
                    "classifier diagnostic"
                )
    if bundle.halt_event_count < cfg.g5_min_halt_events:
        next_data_needs.append(
            f"expand halt-event corpus from {bundle.halt_event_count} to "
            f"≥ {cfg.g5_min_halt_events}"
        )

    required_next: str = ""
    if overall == OverallResult.PROMOTION_BLOCKED_MANIFEST_INVALID:
        required_next = _REQUIRED_NEXT_DATA_OR_POLICY_MANIFEST_INVALID

    return replace(
        candidate_with_bundle,
        gates=tuple(gate_results[gid] for gid in GATE_IDS),
        result=overall,
        blocking_reasons=reasons,
        next_data_needs=tuple(next_data_needs),
        required_next_data_or_policy=required_next,
    )


# ---------------------------------------------------------------------------
# Existing-candidate fail-closed validator
# ---------------------------------------------------------------------------


def _assert_existing_candidate_matches_menu(
    *, existing: CandidateManifest, expected: CandidateManifest,
) -> None:
    """Confirm that an existing on-disk candidate matches what the
    current menu says it should be.

    Ticket 1-closeout-3: this is the second leg of the rerun fail-
    closed contract. The first leg — strict-load with content_sha256
    verification — is handled by :func:`PolicyRegistry.get_candidate`.
    The second leg is here: even a hash-valid manifest must not
    silently outlive a menu change.

    Compared fields:
      - ``candidate_id`` (must match — defence-in-depth, the registry
        already keys on this)
      - ``state`` — menu candidates are always ``draft``; anything
        else means a downstream process touched the registry
      - ``diff.target`` — the knob being proposed
      - ``diff.proposed_value`` — the proposed numeric / value
      - ``diff.baseline_value`` — the recorded "before" value

    Evaluation-derived fields (``gates``, ``result``,
    ``blocking_reasons``, ``next_data_needs``, ``evidence_bundle``)
    are NOT compared here — they legitimately fluctuate when the
    underlying Phase D bundles change.
    """
    if existing.candidate_id != expected.candidate_id:
        raise StaleCandidateError(
            f"existing candidate id {existing.candidate_id!r} does not "
            f"match menu id {expected.candidate_id!r}"
        )
    if existing.state != CandidateState.DRAFT:
        raise StaleCandidateError(
            f"existing candidate {existing.candidate_id!r} has state "
            f"{existing.state.value!r}; menu candidates must remain "
            f"in 'draft' (registry tampering / wrong state suspected)"
        )
    if existing.diff.target != expected.diff.target:
        raise StaleCandidateError(
            f"existing candidate {existing.candidate_id!r} target "
            f"{existing.diff.target!r} does not match menu target "
            f"{expected.diff.target!r} (menu drift / stale registry)"
        )
    if existing.diff.proposed_value != expected.diff.proposed_value:
        raise StaleCandidateError(
            f"existing candidate {existing.candidate_id!r} "
            f"proposed_value {existing.diff.proposed_value!r} does not "
            f"match menu {expected.diff.proposed_value!r} "
            "(menu drift / stale registry)"
        )
    if existing.diff.baseline_value != expected.diff.baseline_value:
        raise StaleCandidateError(
            f"existing candidate {existing.candidate_id!r} "
            f"baseline_value {existing.diff.baseline_value!r} does not "
            f"match menu {expected.diff.baseline_value!r} "
            "(menu drift / stale registry)"
        )


# ---------------------------------------------------------------------------
# Markdown report renderer
# ---------------------------------------------------------------------------


def _fmt_gate_line(r: PromotionGateResult) -> str:
    return f"  - {r.gate_id}: **{r.status.value}** — {r.reason}"


def _fmt_candidate_section(c: CandidateManifest) -> list[str]:
    out: list[str] = []
    out.append(f"### {c.candidate_id}")
    out.append("")
    out.append(f"**Title**: {c.title}")
    out.append("")
    out.append(f"**Diff target**: `{c.diff.target}`")
    out.append("")
    out.append(
        f"**Baseline → proposed**: "
        f"`{c.diff.baseline_value}` → `{c.diff.proposed_value}`"
    )
    out.append("")
    out.append(
        f"**Scope**: regimes={list(c.diff.scope.regimes_affected)}, "
        f"affects_halt_mode={c.diff.scope.affects_halt_mode}, "
        f"raises_gross_exposure={c.diff.scope.raises_gross_exposure}, "
        f"interfaces_touched={list(c.diff.scope.interfaces_touched)}"
    )
    out.append("")
    out.append(f"**RESULT: {c.result.value}**")
    out.append("")
    out.append("**Gate verdicts:**")
    out.append("")
    for r in c.gates:
        out.append(_fmt_gate_line(r))
    out.append("")
    out.append("**Blocking reasons:**")
    out.append("")
    if c.blocking_reasons:
        for r in c.blocking_reasons:
            out.append(f"  - {r}")
    else:
        out.append("  - (none — candidate is READY_FOR_TESTED, advisory only)")
    out.append("")
    out.append("**Required next data:**")
    out.append("")
    if c.next_data_needs:
        for n in c.next_data_needs:
            out.append(f"  - {n}")
    else:
        out.append("  - (none)")
    out.append("")
    if c.required_next_data_or_policy:
        out.append(f"**required_next_data_or_policy**: `{c.required_next_data_or_policy}`")
        out.append("")
    return out


def _render_report(
    *,
    candidates: list[CandidateManifest],
    bundle: EvidenceBundle,
    bounds_path: Path,
    bounds_present: bool,
    registry_root: Path,
) -> str:
    out: list[str] = []
    out.append("# Phase D-cont3 / Ticket 1 — Evolution Report (DIAGNOSTIC ONLY)")
    out.append("")
    out.append(
        "> **Read-only diagnostic.** This report does NOT modify "
        "production rule_engine, decision_server, .mq5, EA wiring, or "
        "`config/safety_bounds.yaml`. It runs Plan §3 promotion gates "
        "G1–G8 against the current Phase D evidence for every candidate "
        "in `CANDIDATE_MENU_V0` and writes manifests under "
        "`policy_registry/candidates/`. No live behaviour change."
    )
    out.append("")
    out.append("## Run config")
    out.append("")
    out.append(f"- Atlas report: `{bundle.atlas_report_path}`")
    out.append(
        f"- Atlas hash (sha256): `{bundle.atlas_report_hash_sha256[:16]}…`"
    )
    out.append(
        f"- Data-availability report: `{bundle.data_availability_report_path}`"
    )
    out.append(
        f"- Data-availability hash (sha256): "
        f"`{bundle.data_availability_report_hash_sha256[:16]}…`"
    )
    out.append(
        f"- Walk-forward artefacts: "
        f"{[Path(p).name for p in bundle.walk_forward_run_paths]}"
    )
    out.append(
        f"- Bundle hash (sha256): `{bundle.bundle_hash_sha256[:16]}…`"
    )
    out.append(
        f"- Safety-bounds source: `{bounds_path}` "
        f"({'PRESENT (read-only)' if bounds_present else 'MISSING — every G6 → safety_bound_undefined per Plan §3 G6 (b)'})"
    )
    out.append(f"- Registry root: `{registry_root}`")
    out.append("")

    # T4-F2 — surface the registry-audit state above the candidate
    # gate sections. Operators must never read a clean per-candidate
    # block while the audit log silently records a violation; the
    # G8 layer also blocks PASS in this case (see registry_audit
    # gate in promotion_gates.py).
    audit = getattr(bundle, "registry_audit", None)
    out.append("## Registry append-only audit (T4-F2)")
    out.append("")
    if audit is None:
        out.append(
            "- `audit_state`: **NOT LOADED** — runner did not "
            "supply audit state; G8 cannot enforce the registry "
            "append-only contract this run."
        )
    else:
        out.append(f"- `audit_log_path`: `{audit.audit_log_path}`")
        out.append(
            f"- `audit_log_present`: **{audit.audit_log_present}**"
        )
        out.append(
            f"- `stale_v030_deleted_during_this_session`: "
            f"**{audit.stale_v030_deleted_during_this_session}**"
        )
        out.append(f"- `lost_sha_count`: **{audit.lost_sha_count}**")
        if audit.lost_sha256:
            out.append("- `lost_sha256`:")
            for s in audit.lost_sha256:
                out.append(f"    - `{s}`")
        out.append(
            f"- `registry_append_only_violation`: "
            f"**{audit.registry_append_only_violation}**"
        )
        if audit.registry_append_only_violation:
            out.append("")
            out.append(
                "  ⚠️ **Registry append-only contract violated this "
                "session.** Every candidate below must show G8 = "
                "ABSTAIN with reason starting "
                "`registry_append_only_violation`. If a candidate "
                "shows a different verdict, the report and the "
                "gate code disagree — investigate before approving."
            )
        elif not audit.audit_log_present:
            out.append("")
            out.append(
                "  ℹ️ Audit log file not present at the resolved "
                "path. Treated as `no incidents on file`. If you "
                "expected a log to exist, verify the "
                "`--registry-audit-log` argument."
            )
    out.append("")
    out.append("## Evidence summary (parsed from data-availability report)")
    out.append("")
    out.append(f"- `cross_symbol_count`: {bundle.cross_symbol_count}")
    out.append(f"- `halt_event_count`: {bundle.halt_event_count}")
    out.append(f"- `no_strategy_change`: **{bundle.no_strategy_change}**")
    out.append("")
    out.append("**Year replication:**")
    out.append("")
    out.append("| Symbol | years_total | years_passing | negative_sign_years |")
    out.append("|---|---|---|---|")
    for sym, body in bundle.year_replication.items():
        out.append(
            f"| {sym} | {body['years_total']} | {body['years_passing']} | "
            f"{list(body.get('negative_sign_years', ()))} |"
        )
    out.append("")

    # Headline counts.
    blocked = sum(
        1 for c in candidates if c.result != OverallResult.READY_FOR_TESTED
    )
    out.append("## Headline")
    out.append("")
    out.append(
        f"- Total candidates: **{len(candidates)}**"
    )
    out.append(
        f"- `PROMOTION_BLOCKED` (any kind): **{blocked}** of "
        f"{len(candidates)}"
    )
    if blocked == len(candidates):
        out.append(
            "- ✅ Outcome matches Ticket 1 acceptance criterion: every "
            "menu candidate is `PROMOTION_BLOCKED` against the current "
            "evidence."
        )
    else:
        out.append(
            "- ⚠️ At least one candidate is `READY_FOR_TESTED` — this is "
            "an advisory label only; no state change is performed by "
            "Ticket 1."
        )
    out.append("")

    out.append("## Per-candidate detail")
    out.append("")
    for c in candidates:
        out.extend(_fmt_candidate_section(c))

    # RFC §11 boundary boilerplate (machine-greppable, mirrors RFC).
    out.append("## Architecture-stability invariants (RFC §11, embedded)")
    out.append("")
    out.append(
        "MUST NOT auto-promote any candidate while the most recent "
        "data-availability report has NO_STRATEGY_CHANGE: true."
    )
    out.append(
        "MUST NOT auto-promote any candidate that raises gross exposure, "
        "leverage, max open positions, max recovery multiplier, or max "
        "grid density."
    )
    out.append(
        "MUST NOT mutate the EA, .mq5 sources, decision_server routes, "
        "or rule_engine public contract from the sidecar."
    )
    out.append(
        "MUST treat config/safety_bounds.yaml as sidecar-read-only."
    )
    out.append("")

    # Canonical certification — Plan §9.
    out.append("## Canonical certification")
    out.append("")
    out.append(_CERTIFICATION_LINE)
    out.append("")
    out.append(
        f"Generated at: {datetime.now(timezone.utc).isoformat()}"
    )
    out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _select_shadow_artefact_for_candidate(
    *,
    candidate: CandidateManifest,
    shadow_artefacts_dir: Path,
) -> tuple[Path | None, str | None]:
    """R5 double-key join.

    For a given candidate, look for an artefact under
    ``<shadow_artefacts_dir>/<candidate_id>/*.json``. If multiple
    artefacts are present, pick the most recently created (artefacts
    are append-only). Return ``(path, recomputed_disk_sha256)``.

    NO FALLBACK to candidate_id-only matching: the caller (G8) will
    additionally verify that ``artefact.candidate_manifest_content_hash``
    equals the candidate's current manifest content hash. If they
    don't match, G8 returns FAIL: shadow_artefact_manifest_drift.
    The selection step does NOT pre-filter by manifest hash —
    that's G8's job, and surfacing FAIL is more honest than
    silently choosing a different artefact.
    """
    if not shadow_artefacts_dir.exists():
        return None, None
    cand_dir = shadow_artefacts_dir / candidate.candidate_id
    if not cand_dir.exists():
        return None, None
    artefacts = sorted(
        cand_dir.glob("*.json"),
        key=lambda p: p.stat().st_mtime_ns,
    )
    if not artefacts:
        return None, None
    # Most recent artefact — append-only assumption.
    chosen = artefacts[-1]
    from smc.hedgerock.evolution.evidence_bundle import compute_file_sha256
    return chosen, compute_file_sha256(chosen)


def _default_registry_audit_log(registry_root: Path) -> Path:
    """Derive the default registry-audit log path from
    ``<registry_root>/shadow_artefacts/_audit.md``.

    The default is registry-relative — never hard-coded — so a tmp
    registry used by tests gets a tmp audit log, and the production
    registry gets the real audit log without the operator having to
    pass an extra flag. CLI users can override via
    ``--registry-audit-log``.
    """
    return Path(registry_root) / "shadow_artefacts" / "_audit.md"


def run(
    *,
    atlas_path: Path,
    availability_path: Path,
    walk_forward_paths: list[Path],
    safety_bounds_path: Path,
    registry_root: Path,
    report_path: Path,
    gate_config: GateConfig | None = None,
    shadow_artefacts_dir: Path | None = None,
    registry_audit_log_path: Path | None = None,
) -> tuple[list[CandidateManifest], Path]:
    """Library entry point. Returns ``(evaluated_candidates, report_path)``.

    When ``shadow_artefacts_dir`` is provided, the report joins each
    candidate with at most one artefact via the R5 double-key
    (candidate_id + manifest_content_hash). G8 enforces the
    manifest-hash equality; this function only does the candidate_id
    side of the lookup.

    Ticket 4 v2 follow-on T4-F2: ``registry_audit_log_path`` defaults
    to ``<registry_root>/shadow_artefacts/_audit.md``. The audit log
    is loaded into a :class:`RegistryAuditState` and attached to the
    base evidence bundle so ``g8_shadow_comparison`` and the active
    multi-window PASS evaluator both see the same registry-violation
    signal that the report renderer surfaces. Without this wiring,
    operators could read a "violation: YES" footer and approve a
    candidate whose G8 still said PASS.
    """
    from dataclasses import replace as _replace

    cfg = gate_config or GateConfig()
    artefacts = EvidenceBundleArtefacts(
        bundle_id=f"evb-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}",
        atlas_report_path=Path(atlas_path),
        data_availability_report_path=Path(availability_path),
        walk_forward_run_paths=tuple(Path(p) for p in walk_forward_paths),
    )
    base_bundle = load_evidence_bundle(artefacts)

    # T4-F2 — load the operator audit log and attach to the bundle.
    # ``load_registry_audit_state`` is forgiving: missing/unreadable
    # audit log → clean state with ``audit_log_present=False``,
    # ``violation=False``. The renderer surfaces this so operators
    # never read a clean report assuming the log was checked.
    from smc.hedgerock.evolution.registry_audit import (
        load_registry_audit_state,
    )
    if registry_audit_log_path is None:
        registry_audit_log_path = _default_registry_audit_log(registry_root)
    registry_audit = load_registry_audit_state(registry_audit_log_path)
    base_bundle = _replace(base_bundle, registry_audit=registry_audit)
    bundle = base_bundle  # may be replaced per-candidate when shadow joins
    bounds = load_safety_bounds(safety_bounds_path)
    bounds_present = Path(safety_bounds_path).exists()

    registry = PolicyRegistry(root=registry_root)

    evaluated: list[CandidateManifest] = []
    for cand in CANDIDATE_MENU_V0:
        # R5 double-key join (id side here; G8 enforces hash side).
        # When --shadow-artefacts is provided, attach the most recent
        # artefact under <dir>/<candidate_id>/ to a per-candidate
        # bundle replica. G8 will then verify
        # artefact.candidate_manifest_content_hash == current manifest
        # hash and FAIL on mismatch.
        per_candidate_bundle = base_bundle
        if shadow_artefacts_dir is not None:
            artefact_path, artefact_hash = _select_shadow_artefact_for_candidate(
                candidate=cand,
                shadow_artefacts_dir=Path(shadow_artefacts_dir),
            )
            if artefact_path is not None:
                per_candidate_bundle = _replace(
                    base_bundle,
                    shadow_artefact_path=str(artefact_path),
                    shadow_artefact_hash_sha256=artefact_hash,
                )

        result_manifest = evaluate_candidate(
            candidate=cand, bundle=per_candidate_bundle,
            bounds=bounds, gate_config=cfg,
        )
        evaluated.append(result_manifest)
        # Persist into the registry. write_candidate refuses to
        # overwrite an existing file (manifests are immutable once
        # written). On re-run against an existing registry, the
        # canonical pattern is to **use a fresh registry root for
        # regeneration; do not delete production registry history**
        # (audit log + candidate manifests are append-only treasure,
        # not scratch space).
        #
        # Ticket 1-closeout-3 fail-closed contract: when the candidate
        # file already exists, the rerun MUST verify it before
        # proceeding. Two checks fire:
        #   (a) PolicyRegistry.get_candidate runs the strict envelope
        #       + content_sha256 load; tamper or bare-layout raises
        #       ManifestIntegrityError.
        #   (b) _assert_existing_candidate_matches_menu cross-checks
        #       the stored manifest against the current
        #       CANDIDATE_MENU_V0 entry; semantic drift (e.g.
        #       proposed_value changed in the menu but registry has
        #       the old value) raises StaleCandidateError.
        # Either error propagates out of run() BEFORE any audit-log
        # entry or report file is written — corrupt / stale registries
        # cannot be silently re-run.
        cand_path = registry.candidates_dir / f"{result_manifest.candidate_id}.json"
        if cand_path.exists():
            existing = registry.get_candidate(result_manifest.candidate_id)
            _assert_existing_candidate_matches_menu(
                existing=existing, expected=cand,
            )
        else:
            registry.write_candidate(result_manifest)

    registry.append_audit({
        "action": "evolution_report_run",
        "candidates": [c.candidate_id for c in evaluated],
        "blocked_count": sum(
            1 for c in evaluated if c.result != OverallResult.READY_FOR_TESTED
        ),
        "bundle_hash_sha256": base_bundle.bundle_hash_sha256,
        "shadow_artefacts_dir": (
            str(shadow_artefacts_dir) if shadow_artefacts_dir else None
        ),
        # T4-F2 — record the registry-audit state operators can
        # cross-reference between the report and the persistent
        # policy_registry/audit log.
        "registry_audit_log_path": str(registry_audit.audit_log_path),
        "registry_audit_log_present": registry_audit.audit_log_present,
        "registry_append_only_violation": (
            registry_audit.registry_append_only_violation
        ),
        "registry_lost_sha_count": registry_audit.lost_sha_count,
    })

    body = _render_report(
        candidates=evaluated,
        bundle=bundle,
        bounds_path=Path(safety_bounds_path),
        bounds_present=bounds_present,
        registry_root=Path(registry_root),
    )
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(body, encoding="utf-8")
    return evaluated, Path(report_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--atlas-report", type=Path,
        default=_DEFAULT_DOCS / "phase-d-regime-opportunity-atlas.md",
    )
    parser.add_argument(
        "--data-availability-report", type=Path,
        default=_DEFAULT_DOCS / "phase-d-data-availability.md",
    )
    parser.add_argument(
        "--walk-forward-report", type=Path, action="append", default=None,
        help="Walk-forward markdown / JSONL artefact path. May be "
             "repeated. Defaults to docs/phase-d-walk-forward-report.md.",
    )
    parser.add_argument(
        "--safety-bounds", type=Path, default=_DEFAULT_SAFETY_BOUNDS,
    )
    parser.add_argument(
        "--registry-root", type=Path, default=_DEFAULT_REGISTRY,
    )
    parser.add_argument(
        "--report-path", type=Path, default=_DEFAULT_REPORT_PATH,
    )
    parser.add_argument(
        "--shadow-artefacts", type=Path, default=None,
        help="Optional directory containing shadow artefacts laid out "
             "as <dir>/<candidate_id>/<run_id>.json. When provided, "
             "G8 evidence is read from these artefacts (R5 double-key "
             "join: candidate_id + manifest_content_hash). Without "
             "this flag, G8 stays NOT_RUN.",
    )
    parser.add_argument(
        "--registry-audit-log", type=Path, default=None,
        help="Path to the operator registry audit log "
             "(``_audit.md``). Defaults to "
             "``<registry_root>/shadow_artefacts/_audit.md``. The "
             "audit state is loaded into the evidence bundle so "
             "G8 / the active multi-window PASS evaluator can "
             "ABSTAIN every candidate when the registry's "
             "append-only contract was violated this session "
             "(Ticket 4 v2 T4-F2).",
    )
    args = parser.parse_args(argv)

    wf_paths = args.walk_forward_report or [
        _DEFAULT_DOCS / "phase-d-walk-forward-report.md",
    ]

    print(f"loading evidence bundle: {args.atlas_report}, "
          f"{args.data_availability_report}, {wf_paths} ...")
    if args.shadow_artefacts is not None:
        print(f"  joining shadow artefacts from: {args.shadow_artefacts}")
    try:
        candidates, report_path = run(
            atlas_path=args.atlas_report,
            availability_path=args.data_availability_report,
            walk_forward_paths=wf_paths,
            safety_bounds_path=args.safety_bounds,
            registry_root=args.registry_root,
            report_path=args.report_path,
            shadow_artefacts_dir=args.shadow_artefacts,
            registry_audit_log_path=args.registry_audit_log,
        )
    except FileNotFoundError as e:
        print(f"FAILED: {e}", file=sys.stderr)
        return 2

    print(f"  evaluated {len(candidates)} candidates")
    for c in candidates:
        print(f"    {c.candidate_id}: {c.result.value}")
        for r in c.blocking_reasons:
            print(f"      - {r}")
    print(f"wrote report → {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

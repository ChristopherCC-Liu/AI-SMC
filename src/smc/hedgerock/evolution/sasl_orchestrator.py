"""SASL (Self-Adjustment & Stability Layer) daily-cycle orchestrator.

Composes the five evolution-sidecar diagnostics + auto-corrections into
a single deterministic pass:

    1. health_check pre-flight    (smc.hedgerock.evolution.health_check)
    2. drift detection            (smc.hedgerock.evolution.drift_detector)
    3. append-only purification   (smc.hedgerock.evolution.auto_purifier)
    4. circuit-breaker gate       (smc.hedgerock.evolution.sasl_circuit_breaker)
    5. parameter auto-adjustment  (smc.hedgerock.evolution.auto_adjuster)
    6. SASL cycle report          (frozen dataclass + JSON dump)

REPORT-ONLY by default. ``apply_adjustments=True`` activates the
adjuster, but the circuit breaker still has the final say. The
orchestrator NEVER writes under ``policy_registry/approved/`` — every
artefact lands under ``<workspace>/sasl/``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


# ---------------------------------------------------------------------------
# Workspace-safety helpers (mirror auto_purifier + dry-run shape).
# ---------------------------------------------------------------------------


def _hedgerock_home() -> Path:
    raw = os.environ.get("HEDGEROCK_HOME")
    if raw:
        return Path(raw).expanduser()
    return Path.home() / "HedgeRock"


def _forbidden_workspace_parents() -> tuple[Path, ...]:
    return (_hedgerock_home() / "policy_registry",)


def _assert_workspace_safe(workspace: Path) -> None:
    abs_ws = workspace.resolve()
    for parent in _forbidden_workspace_parents():
        if parent in abs_ws.parents or abs_ws == parent:
            raise ValueError(
                f"SASL workspace lands under a forbidden parent: "
                f"{workspace!s}"
            )


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SASLStageResult:
    name: str
    status: str          # "OK" | "DEGRADED" | "FAILED" | "SKIPPED"
    details: Mapping[str, Any] = field(default_factory=dict)
    note: str = ""


@dataclass(frozen=True)
class SASLCycleReport:
    workspace: str
    generated_at: str
    trigger: str         # "daily" | "event:<event_type>"
    stages: tuple[SASLStageResult, ...]

    health_overall: str | None
    drift_overall_severity: str | None
    purified_archived_total: int | None

    circuit_breaker_frozen: bool
    circuit_breaker_reason: str | None

    proposals_applied: int
    proposals_rejected: int
    proposals: tuple[Mapping[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "workspace": self.workspace,
            "generated_at": self.generated_at,
            "trigger": self.trigger,
            "stages": [
                {
                    "name": s.name, "status": s.status,
                    "note": s.note, "details": dict(s.details),
                } for s in self.stages
            ],
            "health_overall": self.health_overall,
            "drift_overall_severity": self.drift_overall_severity,
            "purified_archived_total": self.purified_archived_total,
            "circuit_breaker_frozen": self.circuit_breaker_frozen,
            "circuit_breaker_reason": self.circuit_breaker_reason,
            "proposals_applied": self.proposals_applied,
            "proposals_rejected": self.proposals_rejected,
            "proposals": [dict(p) for p in self.proposals],
        }


def _ok(name: str, **details: Any) -> SASLStageResult:
    return SASLStageResult(name=name, status="OK", details=dict(details))


def _degraded(name: str, note: str, **details: Any) -> SASLStageResult:
    return SASLStageResult(
        name=name, status="DEGRADED", note=note, details=dict(details),
    )


def _failed(name: str, note: str, **details: Any) -> SASLStageResult:
    return SASLStageResult(
        name=name, status="FAILED", note=note, details=dict(details),
    )


def _skipped(name: str, note: str, **details: Any) -> SASLStageResult:
    return SASLStageResult(
        name=name, status="SKIPPED", note=note, details=dict(details),
    )


# ---------------------------------------------------------------------------
# Markdown report renderer (one-shot, deterministic).
# ---------------------------------------------------------------------------


def _render_report(report: SASLCycleReport) -> str:
    lines: list[str] = []
    lines.append("# SASL Cycle Report")
    lines.append("")
    lines.append(
        f"_Generated at {report.generated_at} — REPORT-ONLY. NOT LIVE._"
    )
    lines.append("")
    lines.append(f"- **trigger:** `{report.trigger}`")
    lines.append(f"- **workspace:** `{report.workspace}`")
    lines.append(f"- **health_overall:** `{report.health_overall}`")
    lines.append(
        f"- **drift_overall_severity:** `{report.drift_overall_severity}`"
    )
    lines.append(
        f"- **purified_archived_total:** {report.purified_archived_total}"
    )
    lines.append(
        f"- **circuit_breaker_frozen:** {report.circuit_breaker_frozen}"
    )
    if report.circuit_breaker_reason:
        lines.append(f"  - reason: {report.circuit_breaker_reason}")
    lines.append(f"- **proposals_applied:** {report.proposals_applied}")
    lines.append(f"- **proposals_rejected:** {report.proposals_rejected}")
    lines.append("")
    lines.append("## Stages")
    lines.append("")
    lines.append("| Stage | Status | Note |")
    lines.append("|---|---|---|")
    for s in report.stages:
        lines.append(f"| {s.name} | {s.status} | {s.note} |")
    lines.append("")
    if report.proposals:
        lines.append("## Adjustment proposals")
        lines.append("")
        lines.append("| parameter | current | proposed | delta | applied | reason |")
        lines.append("|---|---|---|---|---|---|")
        for p in report.proposals:
            lines.append(
                f"| {p.get('parameter')} | {p.get('current_value')} | "
                f"{p.get('proposed_value')} | {p.get('delta')} | "
                f"{p.get('applied')} | {p.get('reason')} |"
            )
        lines.append("")
    lines.append("---")
    lines.append("**status: NOT LIVE / NOT APPROVED / NOT DEPLOYED**")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class SASLOrchestrator:
    """Daily-cycle composer. One instance per (workspace, trigger)
    invocation; safe to construct cheaply."""

    def __init__(
        self,
        *,
        workspace: Path,
        circuit_breaker: object | None = None,
        apply_adjustments: bool = False,
    ) -> None:
        workspace = Path(workspace).resolve()
        _assert_workspace_safe(workspace)
        workspace.mkdir(parents=True, exist_ok=True)
        self.workspace = workspace
        self.circuit_breaker = circuit_breaker
        self.apply_adjustments = apply_adjustments

    # ------------------------------------------------------------------
    # Public entries.
    # ------------------------------------------------------------------

    def run_daily_cycle(
        self,
        *,
        market_bars: Sequence[Mapping[str, float]] | None = None,
        baseline_bars: Sequence[Mapping[str, float]] | None = None,
        evidence_registry: Mapping[str, str | datetime | None] | None = None,
        current_params: Mapping[str, float] | None = None,
        baseline_params: Mapping[str, float] | None = None,
        recent_outcomes: Sequence[Mapping[str, float]] | None = None,
        baseline_outcomes: Sequence[Mapping[str, float]] | None = None,
        now: datetime | None = None,
    ) -> SASLCycleReport:
        return self._run(
            trigger="daily",
            now=now,
            market_bars=market_bars, baseline_bars=baseline_bars,
            evidence_registry=evidence_registry,
            current_params=current_params,
            baseline_params=baseline_params,
            recent_outcomes=recent_outcomes,
            baseline_outcomes=baseline_outcomes,
        )

    def run_event_triggered(
        self,
        *,
        event_type: str,
        event_data: Mapping[str, Any] | None = None,
        market_bars: Sequence[Mapping[str, float]] | None = None,
        baseline_bars: Sequence[Mapping[str, float]] | None = None,
        evidence_registry: Mapping[str, str | datetime | None] | None = None,
        current_params: Mapping[str, float] | None = None,
        baseline_params: Mapping[str, float] | None = None,
        recent_outcomes: Sequence[Mapping[str, float]] | None = None,
        baseline_outcomes: Sequence[Mapping[str, float]] | None = None,
        now: datetime | None = None,
    ) -> SASLCycleReport:
        return self._run(
            trigger=f"event:{event_type}",
            now=now,
            market_bars=market_bars, baseline_bars=baseline_bars,
            evidence_registry=evidence_registry,
            current_params=current_params,
            baseline_params=baseline_params,
            recent_outcomes=recent_outcomes,
            baseline_outcomes=baseline_outcomes,
        )

    # ------------------------------------------------------------------
    # Internal pipeline.
    # ------------------------------------------------------------------

    def _run(
        self,
        *,
        trigger: str,
        now: datetime | None,
        market_bars: Sequence[Mapping[str, float]] | None,
        baseline_bars: Sequence[Mapping[str, float]] | None,
        evidence_registry: Mapping[str, str | datetime | None] | None,
        current_params: Mapping[str, float] | None,
        baseline_params: Mapping[str, float] | None,
        recent_outcomes: Sequence[Mapping[str, float]] | None,
        baseline_outcomes: Sequence[Mapping[str, float]] | None,
    ) -> SASLCycleReport:
        now = now or datetime.now(timezone.utc)
        stages: list[SASLStageResult] = []

        # Stage 1 — health check.
        health_overall = None
        try:
            from smc.hedgerock.evolution.health_check import diagnose
            h = diagnose(
                registry_root=self.workspace / "registry",
                queue_path=self.workspace / "queue" / "shadow_test_queue.jsonl",
                ledger_path=self.workspace / "ledger" / "paper_test_ledger.jsonl",
            )
            health_overall = h.overall.value
            stages.append(_ok("health_check", overall=health_overall))
        except Exception as e:
            stages.append(_degraded("health_check", note=f"raised: {e!r}"))

        # Stage 2 — drift detection.
        drift_overall = None
        drift_report_obj = None
        try:
            from smc.hedgerock.evolution.drift_detector import DriftDetector
            drift_report_obj = DriftDetector().overall_assessment(
                bars_recent=market_bars, bars_baseline=baseline_bars,
                evidence_registry=evidence_registry,
                current_params=current_params,
                baseline_params=baseline_params,
                recent_outcomes=recent_outcomes,
                baseline_outcomes=baseline_outcomes,
                now=now,
            )
            drift_overall = drift_report_obj.overall_severity
            stages.append(_ok(
                "drift_detect", overall_severity=drift_overall,
            ))
        except Exception as e:
            stages.append(_degraded("drift_detect", note=f"raised: {e!r}"))

        # Stage 3 — auto-purify.
        purified_total: int | None = None
        try:
            from smc.hedgerock.evolution.auto_purifier import AutoPurifier
            purge = AutoPurifier.run_full_purification(
                self.workspace, now=now,
            )
            purified_total = int(purge.overall_archived)
            stages.append(_ok(
                "auto_purify", archived_total=purified_total,
            ))
        except Exception as e:
            stages.append(_degraded("auto_purify", note=f"raised: {e!r}"))

        # Stage 4 — circuit breaker gate.
        cb_frozen = False
        cb_reason: str | None = None
        try:
            cb = self.circuit_breaker
            if cb is not None:
                cb_frozen = bool(cb.is_frozen(now=now))
                cb_reason = cb.freeze_reason(now=now) if cb_frozen else None
                stages.append(_ok(
                    "circuit_breaker",
                    frozen=cb_frozen, reason=cb_reason or "",
                ))
            else:
                stages.append(_skipped(
                    "circuit_breaker",
                    note="no breaker supplied; auto-adjust runs unguarded",
                ))
        except Exception as e:
            stages.append(_degraded("circuit_breaker", note=f"raised: {e!r}"))

        # Stage 5 — auto-adjust (gated by circuit breaker AND apply flag).
        proposals_applied = 0
        proposals_rejected = 0
        proposals_payload: list[dict[str, Any]] = []
        try:
            if drift_report_obj is None:
                stages.append(_skipped(
                    "auto_adjust", note="no drift report",
                ))
            elif not self.apply_adjustments:
                # Still PROPOSE so the operator sees the would-be diffs.
                from smc.hedgerock.evolution.auto_adjuster import AutoAdjuster
                proposals = AutoAdjuster.propose_adjustments(
                    drift_report_obj, current_params=current_params,
                )
                for p in proposals:
                    proposals_payload.append({
                        "parameter": p.parameter,
                        "current_value": p.current_value,
                        "proposed_value": p.proposed_value,
                        "delta": p.delta,
                        "applied": False,
                        "reason": p.rationale,
                        "triggered_by": p.triggered_by,
                    })
                stages.append(_ok(
                    "auto_adjust", mode="propose_only",
                    n_proposals=len(proposals),
                ))
                proposals_rejected = len(proposals)  # all rejected (not applied)
            else:
                from smc.hedgerock.evolution.auto_adjuster import AutoAdjuster
                proposals = AutoAdjuster.propose_adjustments(
                    drift_report_obj, current_params=current_params,
                )
                adjusted = AutoAdjuster.apply_adjustments(
                    proposals, current_params or {},
                    circuit_breaker=self.circuit_breaker,
                )
                proposals_applied = len(adjusted.proposals_applied)
                proposals_rejected = len(adjusted.proposals_rejected)
                for p in adjusted.proposals_applied:
                    proposals_payload.append({
                        "parameter": p.parameter,
                        "current_value": p.current_value,
                        "proposed_value": p.proposed_value,
                        "delta": p.delta,
                        "applied": True,
                        "reason": p.rationale,
                        "triggered_by": p.triggered_by,
                    })
                for p, why in adjusted.proposals_rejected:
                    proposals_payload.append({
                        "parameter": p.parameter,
                        "current_value": p.current_value,
                        "proposed_value": p.proposed_value,
                        "delta": p.delta,
                        "applied": False,
                        "reason": why,
                        "triggered_by": p.triggered_by,
                    })
                # Record one circuit-breaker tick per APPLIED adjustment.
                if self.circuit_breaker is not None:
                    for p in adjusted.proposals_applied:
                        try:
                            self.circuit_breaker.record_adjustment(
                                timestamp=now,
                                adjustment_id=f"{p.parameter}@{now.isoformat()}",
                            )
                        except Exception:
                            pass
                stages.append(_ok(
                    "auto_adjust", mode="apply",
                    n_applied=proposals_applied,
                    n_rejected=proposals_rejected,
                ))
        except ValueError as e:
            # FORBIDDEN_PARAMS hit — surface but do not crash the cycle.
            stages.append(_failed("auto_adjust", note=f"refused: {e!r}"))
        except Exception as e:
            stages.append(_degraded("auto_adjust", note=f"raised: {e!r}"))

        # Stage 6 — emit report bundle.
        report = SASLCycleReport(
            workspace=str(self.workspace),
            generated_at=now.isoformat(),
            trigger=trigger,
            stages=tuple(stages),
            health_overall=health_overall,
            drift_overall_severity=drift_overall,
            purified_archived_total=purified_total,
            circuit_breaker_frozen=cb_frozen,
            circuit_breaker_reason=cb_reason,
            proposals_applied=proposals_applied,
            proposals_rejected=proposals_rejected,
            proposals=tuple(proposals_payload),
        )

        sasl_dir = self.workspace / "sasl"
        sasl_dir.mkdir(parents=True, exist_ok=True)
        stamp = now.strftime("%Y%m%dT%H%M%SZ")
        (sasl_dir / f"sasl_cycle_{stamp}.json").write_text(
            json.dumps(report.to_dict(), indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        (sasl_dir / f"sasl_cycle_{stamp}.md").write_text(
            _render_report(report), encoding="utf-8",
        )
        return report

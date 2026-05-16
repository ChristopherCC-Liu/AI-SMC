"""Auto-adjuster — proposes and applies parameter micro-adjustments based
on a :class:`DriftReport`.

Pure-stdlib, read-only on inputs. Never mutates ``current_params``.
Hard red-line parameters listed in :data:`AutoAdjuster.FORBIDDEN_PARAMS`
are never touched: attempting to apply a proposal that targets one of
them raises :class:`ValueError` immediately.

This module deliberately does NOT import ``rule_engine`` or
``decision_server`` — it operates on plain mappings and the
:class:`DriftReport` shape from :mod:`drift_detector`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping

from smc.hedgerock.evolution.drift_detector import DriftReport


# ---------------------------------------------------------------------------
# Frozen result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AdjustmentProposal:
    parameter: str
    current_value: float
    proposed_value: float
    delta: float
    rationale: str
    severity: str
    triggered_by: str


@dataclass(frozen=True)
class AdjustedParams:
    before: Mapping[str, float]
    after: Mapping[str, float]
    proposals_applied: tuple[AdjustmentProposal, ...]
    proposals_rejected: tuple[tuple[AdjustmentProposal, str], ...]
    audit_entries: tuple[dict, ...]
    generated_at: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clamp(value: float, lo: float, hi: float) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Auto adjuster
# ---------------------------------------------------------------------------


class AutoAdjuster:
    """Translate :class:`DriftReport` signals into bounded adjustments."""

    ADJUSTABLE_PARAMS: dict[str, dict] = {
        "regime_vol_threshold": {
            "delta_max": 0.05,
            "floor": 0.5,
            "ceiling": 3.0,
        },
        "anomaly_sensitivity_multiplier": {
            "delta_max": 0.1,
            "floor": 1.0,
            "ceiling": 5.0,
        },
        "stop_atr_multiplier": {
            "delta_max": 0.3,
            "floor": 1.2,
            "ceiling": 3.5,
        },
        "confidence_prior_alpha": {
            "delta_max": 2.0,
            "floor": 1.0,
            "ceiling": 100.0,
        },
    }

    FORBIDDEN_PARAMS: frozenset = frozenset(
        {
            "position_size",
            "leverage",
            "risk_floor",
            "max_drawdown_limit",
        }
    )

    # Mapping each DriftReport component → (parameter, sign, severity_levels).
    # `sign` is +1 (raise) or -1 (lower). `severity_levels` maps a severity
    # label to the fraction of delta_max to apply; absent labels yield no
    # proposal for that axis.
    _TRIGGER_MAP: dict[str, tuple[str, int, dict[str, float]]] = {
        "regime_baseline": (
            "regime_vol_threshold",
            +1,
            {"high": 1.0, "moderate": 0.5},
        ),
        "evidence_freshness": (
            "confidence_prior_alpha",
            +1,
            {"stale": 1.0, "aging": 0.5},
        ),
        "parameter_stability": (
            "anomaly_sensitivity_multiplier",
            -1,
            {"unstable": 1.0, "drifting": 0.5},
        ),
        "recommendation_quality": (
            "stop_atr_multiplier",
            +1,
            {"degraded": 1.0, "watching": 0.5},
        ),
    }

    # ------------------------------------------------------------------
    # Proposal generation
    # ------------------------------------------------------------------

    @classmethod
    def _midpoint(cls, parameter: str) -> float:
        spec = cls.ADJUSTABLE_PARAMS[parameter]
        return (spec["floor"] + spec["ceiling"]) / 2.0

    @classmethod
    def _make_proposal(
        cls,
        parameter: str,
        sign: int,
        fraction: float,
        severity: str,
        triggered_by: str,
        current_params: Mapping[str, float] | None,
    ) -> AdjustmentProposal:
        spec = cls.ADJUSTABLE_PARAMS[parameter]
        delta_max = float(spec["delta_max"])
        delta = sign * fraction * delta_max

        if current_params is not None and parameter in current_params:
            try:
                current_value = float(current_params[parameter])
            except (TypeError, ValueError):
                current_value = cls._midpoint(parameter)
        else:
            current_value = cls._midpoint(parameter)

        proposed_value = current_value + delta
        # Don't pre-clamp here — validate_adjustment will catch out-of-range
        # cases and apply_adjustments will reject them. We DO clamp to keep
        # the proposed value within sane numeric range so downstream audit
        # entries are readable; but we keep the intended delta intact.
        rationale = (
            f"{triggered_by} severity={severity} → "
            f"{'+' if sign > 0 else '-'}{fraction:.2f}*delta_max"
            f" on {parameter}"
        )
        return AdjustmentProposal(
            parameter=parameter,
            current_value=current_value,
            proposed_value=proposed_value,
            delta=delta,
            rationale=rationale,
            severity=severity,
            triggered_by=triggered_by,
        )

    @classmethod
    def propose_adjustments(
        cls,
        drift_report: DriftReport,
        current_params: Mapping[str, float] | None = None,
    ) -> list[AdjustmentProposal]:
        """Translate the four DriftReport components into proposals."""
        proposals: list[AdjustmentProposal] = []

        component_severity = {
            "regime_baseline": drift_report.regime_baseline.severity,
            "evidence_freshness": drift_report.evidence_freshness.severity,
            "parameter_stability": drift_report.parameter_stability.severity,
            "recommendation_quality": drift_report.recommendation_quality.severity,
        }

        for component, severity in component_severity.items():
            parameter, sign, severity_levels = cls._TRIGGER_MAP[component]
            fraction = severity_levels.get(severity)
            if fraction is None or fraction == 0.0:
                continue
            proposals.append(
                cls._make_proposal(
                    parameter=parameter,
                    sign=sign,
                    fraction=fraction,
                    severity=severity,
                    triggered_by=component,
                    current_params=current_params,
                )
            )

        return proposals

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @classmethod
    def validate_adjustment(cls, proposal: AdjustmentProposal) -> bool:
        """Strict range check. Returns False on any violation."""
        if proposal.parameter in cls.FORBIDDEN_PARAMS:
            return False
        if proposal.parameter not in cls.ADJUSTABLE_PARAMS:
            return False
        spec = cls.ADJUSTABLE_PARAMS[proposal.parameter]
        delta_max = float(spec["delta_max"])
        floor = float(spec["floor"])
        ceiling = float(spec["ceiling"])

        if abs(proposal.delta) > delta_max + 1e-12:
            return False
        if proposal.proposed_value < floor - 1e-12:
            return False
        if proposal.proposed_value > ceiling + 1e-12:
            return False
        return True

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------

    @classmethod
    def _validation_failure_reason(cls, proposal: AdjustmentProposal) -> str:
        """Return human-readable reason matching validate_adjustment failure."""
        if proposal.parameter in cls.FORBIDDEN_PARAMS:
            return "forbidden_parameter"
        if proposal.parameter not in cls.ADJUSTABLE_PARAMS:
            return "unknown_parameter"
        spec = cls.ADJUSTABLE_PARAMS[proposal.parameter]
        delta_max = float(spec["delta_max"])
        floor = float(spec["floor"])
        ceiling = float(spec["ceiling"])
        if abs(proposal.delta) > delta_max + 1e-12:
            return f"delta_exceeds_max ({proposal.delta} vs {delta_max})"
        if proposal.proposed_value < floor - 1e-12:
            return f"proposed_below_floor ({proposal.proposed_value} < {floor})"
        if proposal.proposed_value > ceiling + 1e-12:
            return f"proposed_above_ceiling ({proposal.proposed_value} > {ceiling})"
        return "validation_failed"

    @classmethod
    def apply_adjustments(
        cls,
        proposals: list[AdjustmentProposal],
        current_params: Mapping[str, float],
        *,
        circuit_breaker: object | None = None,
    ) -> AdjustedParams:
        """Apply proposals atomically (per-proposal). Never mutates inputs."""
        # Hard red-line check FIRST. A proposal aimed at a forbidden
        # parameter raises immediately — no partial application.
        for proposal in proposals:
            if proposal.parameter in cls.FORBIDDEN_PARAMS:
                raise ValueError(
                    f"auto_adjuster refuses to touch forbidden parameter "
                    f"'{proposal.parameter}'"
                )

        before = dict(current_params)
        after: dict[str, float] = dict(current_params)
        applied: list[AdjustmentProposal] = []
        rejected: list[tuple[AdjustmentProposal, str]] = []
        audit_entries: list[dict] = []

        frozen = bool(
            circuit_breaker is not None
            and getattr(circuit_breaker, "is_frozen", lambda: False)()
        )

        for proposal in proposals:
            timestamp = _now_iso()

            if frozen:
                rejected.append((proposal, "circuit_breaker_frozen"))
                audit_entries.append(
                    {
                        "timestamp": timestamp,
                        "parameter": proposal.parameter,
                        "current_value": proposal.current_value,
                        "proposed_value": proposal.proposed_value,
                        "applied": False,
                        "reason": "circuit_breaker_frozen",
                        "triggered_by": proposal.triggered_by,
                    }
                )
                continue

            if cls.validate_adjustment(proposal):
                after[proposal.parameter] = float(proposal.proposed_value)
                applied.append(proposal)
                audit_entries.append(
                    {
                        "timestamp": timestamp,
                        "parameter": proposal.parameter,
                        "current_value": proposal.current_value,
                        "proposed_value": proposal.proposed_value,
                        "applied": True,
                        "reason": "applied",
                        "triggered_by": proposal.triggered_by,
                    }
                )
            else:
                reason = cls._validation_failure_reason(proposal)
                rejected.append((proposal, reason))
                audit_entries.append(
                    {
                        "timestamp": timestamp,
                        "parameter": proposal.parameter,
                        "current_value": proposal.current_value,
                        "proposed_value": proposal.proposed_value,
                        "applied": False,
                        "reason": reason,
                        "triggered_by": proposal.triggered_by,
                    }
                )

        return AdjustedParams(
            before=before,
            after=after,
            proposals_applied=tuple(applied),
            proposals_rejected=tuple(rejected),
            audit_entries=tuple(audit_entries),
            generated_at=_now_iso(),
        )


__all__ = [
    "AdjustmentProposal",
    "AdjustedParams",
    "AutoAdjuster",
]

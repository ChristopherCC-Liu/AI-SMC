"""Ticket 3 Step 6 — dormant PASS evaluator + multi-symbol skeleton.

**Pure function.** Given the full set of evidence + metrics, returns
a :class:`PassEvaluation` indicating whether the candidate is
eligible for G8 PASS or which gate blocks promotion.

Per Ticket 3 plan v2 §R1: implementation is **complete** so future
multi-symbol data triggers PASS automatically; v1's single-symbol
lake hits the single_symbol gate first and never reaches the
metrics-comparison part.

This evaluator is consumed by ``g8_shadow_comparison`` (Step 7) at
the very end of the verdict table — it represents row 26 of the
27-row verdict.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from smc.hedgerock.evolution.shadow_artefact import ShadowMetrics


__all__ = [
    "MultiWindowPassEvaluation",
    "MultiWindowPassThresholds",
    "PassEvaluation",
    "PassThresholds",
    "evaluate_pass",
    "evaluate_pass_xauusd_multi_window",
]


@dataclass(frozen=True)
class PassThresholds:
    """v1 default thresholds. Lowering is a human-only config change
    (same treatment as G2/G5 per RFC §10.2). v1 lake never reaches
    metric thresholds — single_symbol gate fires first."""

    min_symbols: int = 2
    min_years_per_symbol: int = 3
    min_delta_pnl_pct: float = 0.5         # candidate must beat baseline by ≥ 0.5pp
    max_delta_dd_pct: float = 0.5          # DD may worsen by ≤ 0.5pp
    max_near_stopout_increase: int = 0     # any increase blocks
    max_halt_event_increase: int = 0       # any increase blocks
    min_halt_events: int = 30              # when affects_halt_mode


@dataclass(frozen=True)
class PassEvaluation:
    eligible_for_pass: bool
    block_reason: str   # "" when eligible
    details: dict[str, Any]


def evaluate_pass(
    *,
    symbols: tuple[str, ...],
    year_replication: dict[str, dict[str, Any]],
    baseline_metrics: ShadowMetrics,
    candidate_metrics: ShadowMetrics,
    delta_metrics: ShadowMetrics,
    exposure_class_violation: bool,
    mirror_consistency: str,                # "PASS" | "FAIL"
    affects_halt_mode: bool,
    thresholds: PassThresholds | None = None,
) -> PassEvaluation:
    """Decide whether the candidate is eligible for G8 PASS.

    The order mirrors the verdict-table priority: hard fail-closed
    gates first, then metric gates. Returns the FIRST blocker.
    """
    t = thresholds or PassThresholds()

    # 1. Mirror consistency — drift blocks PASS unconditionally.
    if mirror_consistency != "PASS":
        return PassEvaluation(
            eligible_for_pass=False,
            block_reason=(
                f"mirror_consistency={mirror_consistency!r} — drift blocks PASS"
            ),
            details={"mirror_consistency": mirror_consistency},
        )

    # 2. Single-symbol — DORMANT gate per Ticket 3 R1.
    if len(symbols) < t.min_symbols:
        return PassEvaluation(
            eligible_for_pass=False,
            block_reason=(
                f"single_symbol shadow window — {len(symbols)} symbol(s) "
                f"in slice; need ≥ {t.min_symbols} for PASS"
            ),
            details={"symbols": list(symbols)},
        )

    # 3. Insufficient years per symbol.
    for sym in symbols:
        body = year_replication.get(sym, {})
        years_total = int(body.get("years_total", 0))
        if years_total < t.min_years_per_symbol:
            return PassEvaluation(
                eligible_for_pass=False,
                block_reason=(
                    f"insufficient_years: symbol {sym} has {years_total} "
                    f"years < {t.min_years_per_symbol}"
                ),
                details={"symbol": sym, "years_total": years_total},
            )

    # 4. Negative-sign year on any symbol.
    for sym in symbols:
        body = year_replication.get(sym, {})
        neg_years = tuple(body.get("negative_sign_years", ()))
        if neg_years:
            return PassEvaluation(
                eligible_for_pass=False,
                block_reason=(
                    f"negative_sign_year_present: {sym} has reverse-signed "
                    f"years {list(neg_years)}"
                ),
                details={"symbol": sym, "negative_sign_years": list(neg_years)},
            )

    # 5. Halt corpus when candidate affects halt mode.
    if affects_halt_mode and candidate_metrics.halt_event_count < t.min_halt_events:
        return PassEvaluation(
            eligible_for_pass=False,
            block_reason=(
                f"halt_corpus_insufficient: n="
                f"{candidate_metrics.halt_event_count} < {t.min_halt_events}"
            ),
            details={"halt_event_count": candidate_metrics.halt_event_count},
        )

    # 6. Exposure-class behavioural violation.
    if exposure_class_violation:
        return PassEvaluation(
            eligible_for_pass=False,
            block_reason=(
                "exposure_class_violation: candidate's behavioural "
                "max_open_lots / max_grid_density exceeds baseline"
            ),
            details={"exposure_class_violation": True},
        )

    # 7. Metric thresholds.
    if delta_metrics.total_return_pct < t.min_delta_pnl_pct:
        return PassEvaluation(
            eligible_for_pass=False,
            block_reason=(
                f"delta_pnl_below_threshold: delta_pnl="
                f"{delta_metrics.total_return_pct:+.3f}pp < "
                f"{t.min_delta_pnl_pct}pp"
            ),
            details={"delta_pnl": delta_metrics.total_return_pct},
        )
    if delta_metrics.max_dd_pct > t.max_delta_dd_pct:
        return PassEvaluation(
            eligible_for_pass=False,
            block_reason=(
                f"delta_dd_above_tolerance: delta_dd="
                f"{delta_metrics.max_dd_pct:+.3f}pp > "
                f"{t.max_delta_dd_pct}pp"
            ),
            details={"delta_dd": delta_metrics.max_dd_pct},
        )
    if delta_metrics.near_stopout_count > t.max_near_stopout_increase:
        return PassEvaluation(
            eligible_for_pass=False,
            block_reason=(
                f"near_stopout_increased: delta="
                f"{delta_metrics.near_stopout_count}"
            ),
            details={"delta_near_stopout": delta_metrics.near_stopout_count},
        )
    if delta_metrics.halt_event_count > t.max_halt_event_increase:
        return PassEvaluation(
            eligible_for_pass=False,
            block_reason=(
                f"halt_event_count_increased: delta="
                f"{delta_metrics.halt_event_count}"
            ),
            details={"delta_halt": delta_metrics.halt_event_count},
        )

    # All gates clear.
    return PassEvaluation(
        eligible_for_pass=True,
        block_reason="",
        details={
            "delta_pnl_pct": delta_metrics.total_return_pct,
            "delta_dd_pct": delta_metrics.max_dd_pct,
            "symbols": list(symbols),
        },
    )


# ---------------------------------------------------------------------------
# Ticket 4 v2 — XAUUSD-only multi-window PASS evaluator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MultiWindowPassThresholds:
    """Active XAUUSD multi-window PASS gate thresholds.

    Defaults are duplicated in
    ``policy_registry/config/promotion_gates.yaml`` (operator
    documentation) — runtime code is the source of truth. Lowering
    any value here is a HUMAN-ONLY config change (same treatment
    as G2 / G5 floors per RFC v2 §10.2).

    Sign / unit conventions (matching :class:`shadow_metrics.PerWindowRiskMetrics`):

    * ``min_delta_pnl_pp`` — minimum acceptable per-window candidate
      − baseline PnL delta in percentage points (delta_pnl_pp).
    * ``max_delta_dd_pp`` — maximum tolerated per-window candidate
      − baseline DD delta in percentage points (delta_dd_pp).
    * ``max_near_stopout_increase`` / ``max_halt_event_increase`` —
      per-window max candidate − baseline increase (counts).
    * ``worst_window_dd_ceiling_pp`` — absolute ceiling on the
      candidate's max DD on ANY window (in percentage points; the
      :class:`WorstWindowSummary.worst_candidate_dd_pp` field is
      already converted to pp).
    * ``min_pass_window_fraction`` — fraction of windows that must
      individually clear every per-window threshold. Default 1.0 =
      every window must pass; the gate is fail-closed by design.
    """

    min_delta_pnl_pp: float = 0.5
    max_delta_dd_pp: float = 0.5
    max_near_stopout_increase: int = 0
    max_halt_event_increase: int = 0
    worst_window_dd_ceiling_pp: float = 1.0
    min_pass_window_fraction: float = 1.0


@dataclass(frozen=True)
class MultiWindowPassEvaluation:
    """Active XAUUSD multi-window PASS gate output.

    On ABSTAIN (``eligible_for_pass=False``), ``abstain_reason``
    carries a single XAUUSD-specific reason string and
    ``blocking_window_id`` (when applicable) names the binding
    window so the report CLI can point operators at the weak
    window — never at an aggregate.
    """

    eligible_for_pass: bool
    abstain_reason: str
    blocking_window_id: str
    coverage_pass: bool
    n_windows_total: int
    n_windows_passing: int
    failing_window_ids: tuple[str, ...]
    details: dict[str, Any]


def _per_window_passes(
    m,                     # PerWindowRiskMetrics
    t: MultiWindowPassThresholds,
) -> tuple[bool, str]:
    """Return (passes, first_failing_axis) for one per-window
    metric record. Used both for failing-window enumeration and
    blocking-window-id selection."""
    if m.delta_pnl_pp < t.min_delta_pnl_pp:
        return False, "delta_pnl_below_floor"
    if m.delta_dd_pp > t.max_delta_dd_pp:
        return False, "delta_dd_above_ceiling"
    if m.delta_near_stopout > t.max_near_stopout_increase:
        return False, "near_stopout_increase"
    if m.delta_halt_event_count > t.max_halt_event_increase:
        return False, "halt_event_increase"
    # Absolute candidate DD compared to ceiling (both in pp).
    if (m.candidate_max_dd_pct * 100.0) > t.worst_window_dd_ceiling_pp:
        return False, "candidate_dd_above_ceiling"
    return True, ""


def evaluate_pass_xauusd_multi_window(
    *,
    coverage_report,                    # CoverageReport
    worst_summary,                      # WorstWindowSummary
    per_window_metrics,                 # list[PerWindowRiskMetrics]
    mirror_consistency: str,            # "PASS" | "FAIL"
    exposure_class_violation: bool,
    affects_halt_mode: bool,
    thresholds: MultiWindowPassThresholds | None = None,
    registry_append_only_violation: bool = False,
    registry_audit_log_path: str = "",
) -> MultiWindowPassEvaluation:
    """Decide whether the candidate is eligible for G8 PASS in the
    XAUUSD-only multi-window regime.

    Order of checks:
      1. mirror_consistency != "PASS" → ABSTAIN
      2. coverage_report.coverage_pass is False → ABSTAIN
         (reason prefixed ``xauusd_window_coverage_failed``)
      3. exposure_class_violation True → ABSTAIN
      4. Per-window enumeration — first window failing any axis
         becomes ``blocking_window_id``; the binding axis becomes
         the abstain_reason prefix.
      5. min_pass_window_fraction enforcement (default 1.0).

    The reason vocabulary stays XAUUSD-only — none of the v1
    cross-instrument blocker phrases are produced anywhere in the
    returned ``abstain_reason`` string.
    """
    t = thresholds or MultiWindowPassThresholds()
    coverage_pass_flag = bool(coverage_report.coverage_pass)
    n_total = len(per_window_metrics)
    failing: list[str] = []

    # 0. Registry append-only violation — Ticket 4 v2 T4-F1 hard
    # gate. When the operator audit log reports the shadow-artefact
    # registry was tampered with this session, no candidate may
    # PASS regardless of its own metric verdict. This stays
    # FIRST in the evaluation order because it invalidates the
    # entire evidence chain, not just one candidate's metrics.
    if registry_append_only_violation:
        return MultiWindowPassEvaluation(
            eligible_for_pass=False,
            abstain_reason=(
                "registry_append_only_violation: shadow-artefact "
                "registry contract violated this session; "
                f"audit_log={registry_audit_log_path!r}"
            ),
            blocking_window_id="",
            coverage_pass=coverage_pass_flag,
            n_windows_total=n_total,
            n_windows_passing=0,
            failing_window_ids=(),
            details={
                "registry_audit_log_path": registry_audit_log_path,
                "registry_append_only_violation": True,
            },
        )

    # 1. Mirror consistency — drift blocks PASS unconditionally.
    if mirror_consistency != "PASS":
        return MultiWindowPassEvaluation(
            eligible_for_pass=False,
            abstain_reason=(
                f"mirror_consistency={mirror_consistency!r} — drift blocks PASS"
            ),
            blocking_window_id="",
            coverage_pass=coverage_pass_flag,
            n_windows_total=n_total,
            n_windows_passing=0,
            failing_window_ids=(),
            details={"mirror_consistency": mirror_consistency},
        )

    # 2. Window coverage gate — propagate coverage shortfalls verbatim
    # so the report CLI shows the operator exactly which floor missed.
    if not coverage_pass_flag:
        joined = "; ".join(coverage_report.shortfall_reasons) or \
                 "coverage_floor_unmet"
        return MultiWindowPassEvaluation(
            eligible_for_pass=False,
            abstain_reason=f"xauusd_window_coverage_failed: {joined}",
            blocking_window_id="",
            coverage_pass=False,
            n_windows_total=n_total,
            n_windows_passing=0,
            failing_window_ids=(),
            details={"shortfall_reasons": list(coverage_report.shortfall_reasons)},
        )

    # 3. Exposure-class behavioural violation.
    if exposure_class_violation:
        return MultiWindowPassEvaluation(
            eligible_for_pass=False,
            abstain_reason=(
                "exposure_class_violation: candidate's behavioural "
                "max_open_lots / max_grid_density exceeds baseline"
            ),
            blocking_window_id="",
            coverage_pass=True,
            n_windows_total=n_total,
            n_windows_passing=0,
            failing_window_ids=(),
            details={"exposure_class_violation": True},
        )

    # 4. Halt-corpus pre-flight when the candidate affects halt mode.
    if affects_halt_mode and coverage_report.halt_event_windows < 1:
        return MultiWindowPassEvaluation(
            eligible_for_pass=False,
            abstain_reason=(
                f"halt_corpus_insufficient: only "
                f"{coverage_report.halt_event_windows} halt-bearing "
                "windows for halt-mode candidate"
            ),
            blocking_window_id="",
            coverage_pass=True,
            n_windows_total=n_total,
            n_windows_passing=0,
            failing_window_ids=(),
            details={"halt_event_windows": coverage_report.halt_event_windows},
        )

    # 5. Per-window enumeration.
    first_blocker_window = ""
    first_blocker_axis = ""
    for m in per_window_metrics:
        passes, axis = _per_window_passes(m, t)
        if not passes:
            failing.append(m.window_id)
            if not first_blocker_window:
                first_blocker_window = m.window_id
                first_blocker_axis = axis

    n_pass = n_total - len(failing)
    pass_fraction = (n_pass / n_total) if n_total > 0 else 0.0

    # Worst-window axis ABSTAIN — fail-closed by design (default
    # min_pass_window_fraction = 1.0 → any failing window blocks).
    if pass_fraction < t.min_pass_window_fraction:
        worst_value_for_axis = ""
        if first_blocker_axis == "delta_pnl_below_floor":
            worst_value_for_axis = (
                f"worst_delta_pnl_pp="
                f"{worst_summary.worst_delta_pnl_pp:+.3f}pp < "
                f"{t.min_delta_pnl_pp:.3f}pp"
            )
        elif first_blocker_axis == "delta_dd_above_ceiling":
            worst_value_for_axis = (
                f"worst_delta_dd_pp="
                f"{worst_summary.worst_delta_dd_pp:+.3f}pp > "
                f"{t.max_delta_dd_pp:.3f}pp"
            )
        elif first_blocker_axis == "near_stopout_increase":
            worst_value_for_axis = (
                f"worst_delta_near_stopout="
                f"{worst_summary.worst_delta_near_stopout:+d} > "
                f"{t.max_near_stopout_increase}"
            )
        elif first_blocker_axis == "halt_event_increase":
            worst_value_for_axis = (
                f"worst_delta_halt_event_count="
                f"{worst_summary.worst_delta_halt_event_count:+d} > "
                f"{t.max_halt_event_increase}"
            )
        elif first_blocker_axis == "candidate_dd_above_ceiling":
            worst_value_for_axis = (
                f"worst_candidate_dd_pp="
                f"{worst_summary.worst_candidate_dd_pp:.3f}pp > "
                f"{t.worst_window_dd_ceiling_pp:.3f}pp"
            )

        return MultiWindowPassEvaluation(
            eligible_for_pass=False,
            abstain_reason=(
                f"worst_window_{first_blocker_axis}: "
                f"window={first_blocker_window!r} {worst_value_for_axis}; "
                f"failing_windows={list(failing)}"
            ),
            blocking_window_id=first_blocker_window,
            coverage_pass=True,
            n_windows_total=n_total,
            n_windows_passing=n_pass,
            failing_window_ids=tuple(failing),
            details={
                "blocking_axis": first_blocker_axis,
                "failing_window_ids": list(failing),
                "worst_summary": {
                    "delta_pnl_pp": worst_summary.worst_delta_pnl_pp,
                    "delta_dd_pp": worst_summary.worst_delta_dd_pp,
                    "candidate_dd_pp": worst_summary.worst_candidate_dd_pp,
                    "delta_near_stopout": worst_summary.worst_delta_near_stopout,
                    "delta_halt_event_count": worst_summary.worst_delta_halt_event_count,
                },
            },
        )

    # All gates clear.
    return MultiWindowPassEvaluation(
        eligible_for_pass=True,
        abstain_reason="",
        blocking_window_id="",
        coverage_pass=True,
        n_windows_total=n_total,
        n_windows_passing=n_pass,
        failing_window_ids=(),
        details={
            "worst_delta_pnl_pp": worst_summary.worst_delta_pnl_pp,
            "worst_delta_dd_pp": worst_summary.worst_delta_dd_pp,
            "worst_candidate_dd_pp": worst_summary.worst_candidate_dd_pp,
            "n_windows_total": n_total,
        },
    )

"""Stress tester — runs candidate proposals against adversarial
scenarios and produces a survival verdict.

Plug-and-play: takes a list of :class:`CandidateProposal` plus the
live-parameter snapshot from
:func:`decision_server.get_live_parameters` and returns one
:class:`StressTestResult` per (candidate, scenario) pair.

Isolation: this module DOES NOT import ``rule_engine``. It only
calls the evolution-layer sidecars (regime_engine, anomaly_shield,
adaptive_stops) plus :func:`phase_d_walk_forward.run_walk_forward_backtest`
— which are all pure functions or pure-data sidecars.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Iterable, Mapping

from smc.hedgerock.evolution.adaptive_stops import (
    StopRecommendation,
    VolatilityRegime,
    compute_stop_recommendation,
)
from smc.hedgerock.evolution.adversarial_scenarios import (
    BUILTIN_SCENARIOS,
    StressScenario,
    iter_builtin_scenarios,
    scenario_to_window_history,
)
from smc.hedgerock.evolution.anomaly_shield import (
    AnomalyDetector,
    AnomalyLevel,
    shield_action,
)
from smc.hedgerock.evolution.regime_engine import (
    MarketRegime,
    RegimeDetector,
)


__all__ = [
    "REASON_STRESS_TEST_BREACHED",
    "StressTestResult",
    "StressTester",
    "VERDICT_BREACHED",
    "VERDICT_PARTIAL",
    "VERDICT_SURVIVED",
    "render_survival_report",
]


REASON_STRESS_TEST_BREACHED = "stress_test_breached"
VERDICT_SURVIVED = "SURVIVED"
VERDICT_PARTIAL = "PARTIAL"
VERDICT_BREACHED = "BREACHED"


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StressTestResult:
    scenario_id: str
    scenario_name: str
    candidate_id: str
    parameter_class: str
    survived: bool
    max_drawdown_pct: float
    recovery_bars: int | None
    pnl_pct: float
    regime_transitions: tuple[tuple[str, str], ...]
    anomaly_triggers: tuple[str, ...]
    shield_actions: tuple[str, ...]
    stop_adjustments: tuple[str, ...]
    verdict: str
    risk_factor: float
    historical_max_drawdown_pct: float
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Risk-factor model — translate a candidate's parameter tweak into a
# multiplicative position-size factor used by the simulator.
# ---------------------------------------------------------------------------


_RISK_FACTOR_MIN = 0.1
_RISK_FACTOR_MAX = 3.0


def _risk_factor_for_proposal(
    *, parameter_class: str, baseline: float, proposed: float,
) -> float:
    """Return a [0.1, 3.0] multiplier capturing how aggressive the
    candidate's tweak is vs the baseline.

    Lower confidence floor → larger risk_factor (more positions taken).
    Higher aggressive cap → larger risk_factor (more aggressive sizing).
    Longer halt expiry → larger risk_factor (faster re-entry after halt).
    """
    if baseline <= 0 or proposed <= 0:
        return 1.0
    if parameter_class in (
        "confidence_threshold_observe", "confidence_threshold_range_2",
    ):
        # Lower threshold → MORE risk.
        rf = baseline / proposed
    elif parameter_class == "confidence_threshold_aggressive":
        # Lower aggressive floor → MORE risk.
        rf = baseline / proposed
    elif parameter_class == "halt_expiry_observe_hours":
        # Longer expiry → faster re-entry → MORE risk.
        rf = proposed / baseline
    else:
        rf = 1.0
    return max(_RISK_FACTOR_MIN, min(_RISK_FACTOR_MAX, rf))


# ---------------------------------------------------------------------------
# Per-scenario simulation
# ---------------------------------------------------------------------------


def _simulate_equity_curve(
    *, scenario: StressScenario, risk_factor: float,
) -> tuple[float, float, int | None]:
    """Walk the scenario's bars holding a long position sized by
    ``risk_factor``. Return ``(pnl_pct, max_dd_pct, recovery_bars)``.

    ``recovery_bars`` is the index of the first bar AFTER the worst
    drawdown where equity recovered to the running peak (None if it
    never did).
    """
    bars = scenario.ohlc_bars
    if not bars:
        return 0.0, 0.0, None

    entry = bars[0].close
    if entry <= 0:
        return 0.0, 0.0, None

    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    worst_idx = 0
    for i, bar in enumerate(bars):
        ret = (bar.close - entry) / entry * risk_factor
        equity = 1.0 + ret
        if equity > peak:
            peak = equity
        if peak > 0:
            dd_pct = (peak - equity) / peak * 100.0
            if dd_pct > max_dd:
                max_dd = dd_pct
                worst_idx = i

    # Find first bar after worst_idx where equity climbs back to peak.
    recovery_bars: int | None = None
    if worst_idx < len(bars) - 1:
        target_peak_at_worst = peak  # peak as of worst_idx
        # Recompute peak at worst (may differ if peak grew later).
        e_local = 1.0
        peak_local = 1.0
        for i, bar in enumerate(bars[: worst_idx + 1]):
            ret = (bar.close - entry) / entry * risk_factor
            e_local = 1.0 + ret
            peak_local = max(peak_local, e_local)
        for j in range(worst_idx + 1, len(bars)):
            ret = (bars[j].close - entry) / entry * risk_factor
            e = 1.0 + ret
            if e >= peak_local:
                recovery_bars = j - worst_idx
                break

    pnl_pct = (equity - 1.0) * 100.0
    return round(pnl_pct, 4), round(max_dd, 4), recovery_bars


def _verdict_from_drawdown(
    *, simulated_dd: float, scenario_dd: float,
) -> str:
    """Compare the simulated drawdown against the scenario's historical
    drawdown.

    * SURVIVED — simulated ≤ historical (the candidate handles the
      shock no worse than the live system did).
    * BREACHED — simulated > historical * 1.5 (clearly worse — block).
    * PARTIAL  — anything in between (worth a human look).
    """
    if simulated_dd <= scenario_dd:
        return VERDICT_SURVIVED
    if simulated_dd > scenario_dd * 1.5:
        return VERDICT_BREACHED
    return VERDICT_PARTIAL


def _ohlc_to_dicts(scenario: StressScenario) -> list[dict]:
    return [
        {
            "open": b.open, "close": b.close,
            "high": b.high, "low": b.low,
        }
        for b in scenario.ohlc_bars
    ]


# ---------------------------------------------------------------------------
# Tester
# ---------------------------------------------------------------------------


@dataclass
class StressTester:
    """Runs the adversarial scenarios against a candidate proposal."""

    scenarios: tuple[StressScenario, ...] = field(
        default_factory=iter_builtin_scenarios
    )

    def __post_init__(self) -> None:
        if not self.scenarios:
            raise ValueError("scenarios must be a non-empty tuple")

    # ------------------------------------------------------------------
    # Single candidate
    # ------------------------------------------------------------------

    def test_candidate(
        self,
        candidate_proposal,
        live_params: Mapping[str, float],
        *,
        scenarios: Iterable[StressScenario] | None = None,
    ) -> list[StressTestResult]:
        """Run every scenario against ``candidate_proposal``.

        Plug-and-play: the only required inputs are the proposal and
        the live-parameter snapshot. Optional ``scenarios`` overrides
        the constructor's tuple.
        """
        run_scenarios = (
            tuple(scenarios) if scenarios is not None else self.scenarios
        )
        out: list[StressTestResult] = []
        baseline = float(
            live_params.get(
                candidate_proposal.parameter_class,
                candidate_proposal.baseline_value,
            )
        )
        risk_factor = _risk_factor_for_proposal(
            parameter_class=candidate_proposal.parameter_class,
            baseline=baseline,
            proposed=float(candidate_proposal.proposed_value),
        )
        for scenario in run_scenarios:
            out.append(self._test_one(
                scenario=scenario,
                candidate=candidate_proposal,
                risk_factor=risk_factor,
            ))
        return out

    # ------------------------------------------------------------------
    # Batch
    # ------------------------------------------------------------------

    def test_all_candidates(
        self,
        proposals: Iterable,
        live_params: Mapping[str, float],
        *,
        scenarios: Iterable[StressScenario] | None = None,
    ) -> dict[str, list[StressTestResult]]:
        """Run every scenario against every proposal."""
        results: dict[str, list[StressTestResult]] = {}
        for p in proposals:
            results[p.candidate_id] = self.test_candidate(
                p, live_params, scenarios=scenarios,
            )
        return results

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _test_one(
        self, *, scenario: StressScenario, candidate, risk_factor: float,
    ) -> StressTestResult:
        ohlc_dicts = _ohlc_to_dicts(scenario)

        # Regime detection — collect transitions across the scenario.
        regime_detector = RegimeDetector()
        transitions: list[tuple[str, str]] = []
        prev_regime: str | None = None
        # Sliding window — use 60 bars when available, else whole scenario.
        for i in range(1, len(ohlc_dicts) + 1):
            window = ohlc_dicts[max(0, i - 60): i]
            snap = regime_detector.detect(bars=window)
            r = snap.regime.value
            if prev_regime is None:
                prev_regime = r
                continue
            if r != prev_regime:
                transitions.append((prev_regime, r))
                prev_regime = r

        # Anomaly detection — track every level escalation.
        anomaly_detector = AnomalyDetector()
        anomaly_state = anomaly_detector.detect(bars=ohlc_dicts)
        anomaly_triggers = tuple(anomaly_state.triggers)
        action = shield_action(anomaly_state)
        shield_actions: tuple[str, ...] = (
            (
                f"level={anomaly_state.level.value};"
                f"new_candidates_allowed={action.new_candidates_allowed};"
                f"queue_frozen={action.queue_frozen};"
                f"full_lockdown={action.full_lockdown}"
            ),
        )

        # Adaptive-stops vol regime.
        stop_rec: StopRecommendation = compute_stop_recommendation(
            bars=ohlc_dicts,
        )
        stop_adjustments: tuple[str, ...] = (
            (
                f"vol_regime={stop_rec.vol_regime.value};"
                f"atr_multiplier={stop_rec.atr_multiplier};"
                f"position_scale={stop_rec.position_scale}"
            ),
        )

        # Simulate the candidate's equity curve.
        pnl_pct, max_dd, recovery_bars = _simulate_equity_curve(
            scenario=scenario, risk_factor=risk_factor,
        )

        # Build the per-window history adapter so downstream callers
        # (e.g. the replay_validator) can re-run the same scenario
        # through the Tier-1 backtest if they want richer per-window
        # deltas. We don't call the backtest directly from here —
        # stress_tester is not on the Tier-1 unseal whitelist by
        # design (keeps the unseal tight); the windowed adapter is
        # available via :func:`scenario_to_window_history`.
        _ = scenario_to_window_history(scenario)

        verdict = _verdict_from_drawdown(
            simulated_dd=max_dd,
            scenario_dd=scenario.max_drawdown_pct,
        )
        survived = verdict == VERDICT_SURVIVED

        return StressTestResult(
            scenario_id=scenario.scenario_id,
            scenario_name=scenario.name,
            candidate_id=candidate.candidate_id,
            parameter_class=candidate.parameter_class,
            survived=survived,
            max_drawdown_pct=max_dd,
            recovery_bars=recovery_bars,
            pnl_pct=pnl_pct,
            regime_transitions=tuple(transitions),
            anomaly_triggers=anomaly_triggers,
            shield_actions=shield_actions,
            stop_adjustments=stop_adjustments,
            verdict=verdict,
            risk_factor=round(risk_factor, 4),
            historical_max_drawdown_pct=scenario.max_drawdown_pct,
        )


# ---------------------------------------------------------------------------
# Markdown survival report
# ---------------------------------------------------------------------------


def render_survival_report(
    results: Mapping[str, list[StressTestResult]],
) -> str:
    """Render the per-candidate scenario matrix as a markdown table."""
    out: list[str] = []
    out.append("## Stress Test Results")
    out.append("")
    out.append(
        "Per-candidate adversarial scenario survival. **BREACHED** "
        "candidates are demoted to NO_RECOMMENDATION upstream."
    )
    out.append("")
    if not results:
        out.append("- (no candidates evaluated)")
        out.append("")
        return "\n".join(out)

    # Column header = scenario_ids (sorted, deterministic).
    scenario_ids: set[str] = set()
    for rs in results.values():
        for r in rs:
            scenario_ids.add(r.scenario_id)
    ordered = sorted(scenario_ids)

    out.append("| candidate | " + " | ".join(ordered) + " | overall |")
    out.append("|" + "---|" * (len(ordered) + 2))
    for cid in sorted(results):
        cells: list[str] = []
        worst = VERDICT_SURVIVED
        for sid in ordered:
            match = next(
                (r for r in results[cid] if r.scenario_id == sid), None,
            )
            if match is None:
                cells.append("—")
                continue
            cells.append(_verdict_cell(match.verdict))
            if match.verdict == VERDICT_BREACHED:
                worst = VERDICT_BREACHED
            elif (
                match.verdict == VERDICT_PARTIAL
                and worst != VERDICT_BREACHED
            ):
                worst = VERDICT_PARTIAL
        out.append(
            f"| `{cid}` | " + " | ".join(cells) + f" | **{worst}** |"
        )
    out.append("")
    out.append("Legend: ✅ SURVIVED · ⚠️ PARTIAL · 🛑 BREACHED.")
    out.append("")
    return "\n".join(out)


def _verdict_cell(verdict: str) -> str:
    if verdict == VERDICT_SURVIVED:
        return "✅"
    if verdict == VERDICT_BREACHED:
        return "🛑"
    return "⚠️"

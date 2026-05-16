"""Stage 6-followup task 5 — end-to-end self-evolution demo (report-only).

Operator-runnable demo orchestrating the full report-only loop:

    OBSERVE → DETECT → RECOMMEND → QUEUE → INSPECT
                                       └→ (paper-test seed) → DRY-RUN PACKET

All artefacts land under ``--workspace`` (a tmp/sidecar directory).
The demo refuses to write under the production registry root and
verifies the JSON count is unchanged before/after.

Stages produced:

    <workspace>/report/phase-d-evolution-report.md
    <workspace>/report/hedgerock-evolution-recommendation.md
    <workspace>/queue/shadow_test_queue.jsonl
    <workspace>/queue/queue_inspection.md
    <workspace>/ledger/paper_test_ledger.jsonl       (when --seed-paper-trades)
    <workspace>/promotion/packet.md                  (when --produce-packet)
"""

from __future__ import annotations

import argparse
import io
import os
import sys
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path

import math

from smc.hedgerock.evolution.adaptive_stops import (
    VolatilityRegime,
    compute_stop_recommendation,
)
from smc.hedgerock.evolution.anomaly_shield import (
    AnomalyDetector,
    AnomalyLevel,
    shield_action,
)
from smc.hedgerock.evolution.candidate_generator import (
    CandidateProposal,
    DECISION_RECOMMEND,
)
from smc.hedgerock.evolution.multi_timeframe_state import (
    TimeframeState,
    compute_consensus,
)
from smc.hedgerock.evolution.regime_engine import (
    MarketRegime,
    RegimeDetector,
)
from smc.hedgerock.evolution.paper_test_ledger import (
    PaperTestLedger,
    build_paper_test_entry,
)
from smc.hedgerock.evolution.shadow_test_queue import ShadowTestQueue


_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
import hedgerock_evolution_recommend as recommend_cli  # noqa: E402
import hedgerock_evolution_queue_inspect as inspect_cli  # noqa: E402
import hedgerock_evolution_promote as promote_cli  # noqa: E402
sys.path.pop(0)


def _build_demo_market_bars() -> list[dict]:
    """Synthetic XAUUSD-like OHLC bars for the demo's regime + anomaly
    stage. Deterministic, NORMAL-regime calm tape."""
    out: list[dict] = []
    price = 2000.0
    pattern = (0.5, -1.0, 1.5, -0.5, 1.0, -1.5)
    for i in range(80):
        step = 0.005 * pattern[i % len(pattern)]
        new = price * math.exp(step)
        out.append({
            "open": price, "close": new,
            "high": max(price, new) * 1.001,
            "low": min(price, new) * 0.999,
        })
        price = new
    return out


def _hedgerock_home() -> Path:
    raw = os.environ.get("HEDGEROCK_HOME")
    if raw:
        return Path(raw).expanduser()
    return Path.home() / "HedgeRock"


def _ai_smc_home() -> Path:
    raw = os.environ.get("AI_SMC_HOME")
    if raw:
        return Path(raw).expanduser()
    # scripts/<this>.py → repo root via parents[1].
    return Path(__file__).resolve().parents[1]


_REAL_REGISTRY_ROOT = _hedgerock_home() / "policy_registry"
_FORBIDDEN_WORKSPACE_PARENTS = (
    _REAL_REGISTRY_ROOT,
    _ai_smc_home() / "config",
)


_DEMO_AVAILABILITY = """
# Phase D-cont3-preflight (demo fixture — full XAUUSD coverage)

## Year-replication summary

| Symbol | Year | Bars | trend_up | range@≥0.80 | breakout_signed_by_h4 | halt events |
|---|---|---|---|---|---|---|
| XAUUSD | 2021 | 5651 | +0.167% ±0.048 | -0.037% ±0.039 (CI∋0) | -0.001% ±0.085 (CI∋0) | 1 |
| XAUUSD | 2022 | 5674 | +0.222% ±0.074 | -0.035% ±0.044 (CI∋0) | -0.188% ±0.128 | 1 |
| XAUUSD | 2023 | 4913 | +0.163% ±0.069 | +0.155% ±0.042 | -0.401% ±0.111 | 1 |
| XAUUSD | 2024 | 5693 | +0.239% ±0.067 | +0.172% ±0.040 | -0.015% ±0.096 (CI∋0) | 1 |

## Action gate

```yaml
NO_STRATEGY_CHANGE: false
```
"""

_CONFIRMATION = promote_cli.CONFIRMATION_SENTINEL


def _assert_workspace_safe(workspace: Path) -> None:
    abs_ws = workspace.resolve()
    for parent in _FORBIDDEN_WORKSPACE_PARENTS:
        if parent in abs_ws.parents or abs_ws == parent:
            raise ValueError(
                f"workspace lands under a forbidden location: "
                f"{workspace!s}"
            )


def _seed_evidence(workspace: Path) -> tuple[Path, Path, Path, Path, Path]:
    fixture = workspace / "fixture"
    fixture.mkdir(parents=True, exist_ok=True)
    atlas = fixture / "atlas.md"
    atlas.write_text("# atlas (demo fixture)\n", encoding="utf-8")
    avail = fixture / "availability.md"
    avail.write_text(_DEMO_AVAILABILITY, encoding="utf-8")
    wf = fixture / "wf.md"
    wf.write_text("# walk-forward (demo fixture)\n", encoding="utf-8")
    # Point at a NON-EXISTENT bounds file. This mirrors today's
    # production state (config/safety_bounds.yaml is absent, per
    # the Stage-6 acceptance report) and lets G6 fire
    # safety_bound_undefined — which is the v0 candidate-generator
    # trigger that produces RECOMMEND proposals downstream.
    bounds = fixture / "safety_bounds.yaml"
    assert not bounds.exists()
    audit_log = fixture / "_audit.md"
    audit_log.write_text(
        "# Shadow-Artefact Registry Audit Log (demo)\n\n(no incidents)\n",
        encoding="utf-8",
    )
    return atlas, avail, wf, bounds, audit_log


def _seed_paper_trades(*, ledger_path: Path, audit_log: Path,
                       candidate_id: str) -> int:
    led = PaperTestLedger(path=ledger_path, audit_log_path=audit_log)
    base = datetime(2026, 5, 1, 9, 0, 0, tzinfo=timezone.utc)
    n = 25
    for i in range(n):
        et = base + timedelta(hours=i * 4)
        xt = et + timedelta(hours=2)
        led.append(build_paper_test_entry(
            candidate_id=candidate_id, symbol="XAUUSD",
            entry_at=et, exit_at=xt,
            entry_price=2050.0 + 0.4 * i,
            exit_price=2052.5 + 0.4 * i,
            side="long", size_lots=0.10,
            pnl=4.5, drawdown=-1.2,
            gates_at_entry=("G1:PASS", "G6:PASS", "G8:NOT_RUN"),
            audit_log_path=str(audit_log),
        ))
    return n


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--seed-paper-trades", action="store_true")
    parser.add_argument("--produce-packet", action="store_true")
    parser.add_argument("--candidate-id",
                        default="c1-lower-observe-floor-0.50")
    parser.add_argument(
        "--audit-trail", type=Path, default=None,
        help="When set, append one operation-audit entry per "
             "successful demo stage to this path.",
    )
    parser.add_argument(
        "--operator", default=None,
        help="Operator name recorded in the audit trail. Defaults "
             "to $USER, then 'anonymous'.",
    )
    args = parser.parse_args(argv)

    # Lazy import so the demo still loads without an audit trail.
    from smc.hedgerock.evolution.operation_audit import append_operation

    def _audit(op: str, result: str, **details) -> None:
        if args.audit_trail is None:
            return
        try:
            append_operation(
                trail_path=Path(args.audit_trail),
                operation=op, result=result,
                operator=args.operator, details=details,
            )
        except ValueError as e:
            print(f"AUDIT WARN: {e}", file=sys.stderr)

    workspace = Path(args.workspace).resolve()
    try:
        _assert_workspace_safe(workspace)
    except ValueError as e:
        print(f"FAILED (forbidden workspace): {e}", file=sys.stderr)
        return 2
    workspace.mkdir(parents=True, exist_ok=True)

    real_pre = (
        sum(1 for _ in _REAL_REGISTRY_ROOT.rglob("*.json"))
        if _REAL_REGISTRY_ROOT.exists() else 0
    )

    print("== Stage: HEALTH-CHECK — pre-flight subsystem scan ==")
    from smc.hedgerock.evolution.health_check import (
        HealthStatus as _HS, diagnose as _diag, auto_recover as _autorecover,
    )
    health = _diag(
        registry_root=workspace / "registry",
        queue_path=workspace / "queue" / "shadow_test_queue.jsonl",
        ledger_path=workspace / "ledger" / "paper_test_ledger.jsonl",
        config_path=None,
        calibrator_state_path=None,
    )
    if health.overall in (_HS.CRITICAL, _HS.DOWN):
        print(
            f"  pre-recovery overall={health.overall.value}; attempting safe recovery",
        )
        rec_result = _autorecover(
            health,
            registry_root=workspace / "registry",
            queue_path=workspace / "queue" / "shadow_test_queue.jsonl",
            ledger_path=workspace / "ledger" / "paper_test_ledger.jsonl",
        )
        for a in rec_result.actions:
            print(f"  recover [{a.subsystem}] {a.description} → "
                  f"{'OK' if a.succeeded else 'FAIL: ' + a.error}")
        print(f"  post-recovery overall={rec_result.post_status.value}")
        _audit("health_check", "ok", overall=health.overall.value,
               post=rec_result.post_status.value,
               recovered=len(rec_result.actions))
    else:
        print(f"  overall={health.overall.value}")
        _audit("health_check", "ok", overall=health.overall.value,
               post=health.overall.value, recovered=0)

    print("== Stage: OBSERVE — load Phase D evidence + audit state ==")
    atlas, avail, wf, bounds, audit_log = _seed_evidence(workspace)
    print(f"  fixture under: {workspace / 'fixture'}")
    _audit("observe", "ok", workspace=str(workspace))

    print("== Stage: REGIME + ANOMALY — sidecar self-protection check ==")
    market_bars = _build_demo_market_bars()
    regime = RegimeDetector().detect(bars=market_bars)
    anomaly = AnomalyDetector().detect(bars=market_bars)
    action = shield_action(anomaly)
    print(
        f"  regime={regime.regime.value} "
        f"(confidence={regime.confidence:.2f}); "
        f"anomaly={anomaly.level.value}; "
        f"new_candidates_allowed={action.new_candidates_allowed}; "
        f"queue_frozen={action.queue_frozen}; "
        f"full_lockdown={action.full_lockdown}"
    )
    _audit(
        "regime_anomaly", "ok",
        regime=regime.regime.value,
        anomaly=anomaly.level.value,
        new_candidates_allowed=action.new_candidates_allowed,
    )

    print("== Stage: TIMEFRAME + STOP — multi-TF consensus + stop advisory ==")
    consensus = compute_consensus(
        # Demo: stamp explicit READY across timeframes so the consensus
        # passes the threshold without needing realistic OHLC for each.
        d1_state=TimeframeState.READY,
        h4_state=TimeframeState.READY,
        h1_state=TimeframeState.VALIDATING,
        m5_state=TimeframeState.VALIDATING,
    )
    stop = compute_stop_recommendation(bars=market_bars)
    print(
        f"  session={consensus.active_session.value}; "
        f"score={consensus.consensus_score:.2f}; "
        f"can_recommend={consensus.can_recommend}; "
        f"vol_regime={stop.vol_regime.value}; "
        f"atr_mult={stop.atr_multiplier}; "
        f"position_scale={stop.position_scale}"
    )
    _audit(
        "timeframe_stop", "ok",
        session=consensus.active_session.value,
        consensus_score=consensus.consensus_score,
        can_recommend=consensus.can_recommend,
        vol_regime=stop.vol_regime.value,
        atr_multiplier=stop.atr_multiplier,
    )

    print("== Stage: STRESS-TEST — adversarial scenario survival ==")
    from smc.hedgerock.evolution.candidate_generator import (
        CandidateProposal as _CP,
        DECISION_RECOMMEND as _DR,
        get_live_parameter_snapshot as _live_snapshot,
    )
    from smc.hedgerock.evolution.stress_tester import (
        StressTester,
        VERDICT_BREACHED,
    )
    demo_proposal = _CP(
        candidate_id="demo-stress-probe",
        parameter_target="smc.hedgerock.rule_engine._CONFIDENCE_OBSERVE_FLOOR",
        parameter_class="confidence_threshold_observe",
        baseline_value=0.55,
        proposed_value=0.55,  # neutral — should SURVIVE every scenario
        triggered_by=("demo",),
        expected_improvement="demo: neutral probe to exercise the tester",
        risks=(),
        next_validation=(),
        decision=_DR,
        decision_reason="",
    )
    stress_results = StressTester().test_candidate(
        demo_proposal, _live_snapshot(),
    )
    n_survived = sum(1 for r in stress_results if r.verdict != VERDICT_BREACHED)
    n_breached = sum(1 for r in stress_results if r.verdict == VERDICT_BREACHED)
    print(
        f"  scenarios={len(stress_results)} "
        f"survived={n_survived} breached={n_breached}"
    )
    _audit(
        "stress_test", "ok",
        scenarios_total=len(stress_results),
        scenarios_survived=n_survived,
        scenarios_breached=n_breached,
        all_passed=(n_breached == 0),
    )

    print("== Stage: DETECT + RECOMMEND — run report-only recommendation CLI ==")
    report_path = workspace / "report" / "phase-d-evolution-report.md"
    rec_path = workspace / "report" / "hedgerock-evolution-recommendation.md"
    rec_stdout = io.StringIO()
    with redirect_stdout(rec_stdout):
        recommend_rc = recommend_cli.main([
            "--atlas-report", str(atlas),
            "--data-availability-report", str(avail),
            "--walk-forward-report", str(wf),
            "--safety-bounds", str(bounds),
            "--registry-root", str(workspace / "registry"),
            "--report-path", str(report_path),
            "--recommendation-path", str(rec_path),
            "--registry-audit-log", str(audit_log),
        ])
    print(rec_stdout.getvalue())
    if recommend_rc != 0:
        _audit("recommend", "fail", rc=recommend_rc)
        print(f"FAILED (recommend stage): rc={recommend_rc}", file=sys.stderr)
        return 3
    _audit("recommend", "ok", report=str(report_path), recommendation=str(rec_path))

    # Manually parse the recommendation snapshot the candidate
    # generator wrote; pick the first RECOMMEND we find for the
    # default candidate id.
    snap_path = workspace / "report" / "candidate_proposals.json"
    if not snap_path.exists():
        print(
            f"FAILED: candidate generator snapshot missing at {snap_path}",
            file=sys.stderr,
        )
        return 4
    import json
    snap = json.loads(snap_path.read_text(encoding="utf-8"))
    proposals = snap.get("proposals", [])
    recommend_proposals = [
        p for p in proposals if p["decision"] == DECISION_RECOMMEND
    ]
    if not recommend_proposals:
        print("Note: no RECOMMEND proposals this run — queue stage will be empty.")

    print("== Stage: QUEUE — append RECOMMEND proposals to shadow-test queue ==")
    queue_path = workspace / "queue" / "shadow_test_queue.jsonl"
    q = ShadowTestQueue(path=queue_path, audit_log_path=audit_log)
    enqueued = 0
    for d in recommend_proposals:
        q.enqueue_proposals([
            CandidateProposal(
                candidate_id=d["candidate_id"],
                parameter_target=d["parameter_target"],
                parameter_class=d["parameter_class"],
                baseline_value=float(d["baseline_value"]),
                proposed_value=float(d["proposed_value"]),
                triggered_by=tuple(d.get("triggered_by", ())),
                expected_improvement=d.get("expected_improvement", ""),
                risks=tuple(d.get("risks", ())),
                next_validation=tuple(d.get("next_validation", ())),
                decision=DECISION_RECOMMEND,
                decision_reason=d.get("decision_reason", ""),
            ),
        ])
        enqueued += 1
    print(f"  enqueued: {enqueued}")
    _audit("queue", "ok", enqueued=enqueued, queue_path=str(queue_path))

    print("== Stage: INSPECT — read-only queue snapshot ==")
    insp_path = workspace / "queue" / "queue_inspection.md"
    insp_stdout = io.StringIO()
    with redirect_stdout(insp_stdout):
        insp_rc = inspect_cli.main([
            "--queue-path", str(queue_path),
            "--report-path", str(insp_path),
        ])
    print(insp_stdout.getvalue())
    if insp_rc != 0:
        _audit("inspect", "fail", rc=insp_rc)
        print(f"FAILED (inspect stage): rc={insp_rc}", file=sys.stderr)
        return 5
    _audit("inspect", "ok", inspection=str(insp_path))

    if args.seed_paper_trades:
        print("== Stage: PAPER-TEST SEED — write demo paper-test ledger ==")
        ledger_path = workspace / "ledger" / "paper_test_ledger.jsonl"
        n = _seed_paper_trades(
            ledger_path=ledger_path, audit_log=audit_log,
            candidate_id=args.candidate_id,
        )
        print(f"  wrote {n} demo paper-test entries to {ledger_path}")
        _audit("paper_test_seed", "ok", ledger=str(ledger_path), n_trades=n)

    if args.produce_packet:
        print("== Stage: DRY-RUN PROMOTION — produce manual-approval packet ==")
        if not args.seed_paper_trades:
            print(
                "FAILED: --produce-packet requires --seed-paper-trades "
                "(promotion helper validates ledger state).",
                file=sys.stderr,
            )
            return 6
        packet_path = workspace / "promotion" / "packet.md"
        ledger_path = workspace / "ledger" / "paper_test_ledger.jsonl"
        promote_stdout = io.StringIO()
        with redirect_stdout(promote_stdout):
            promote_rc = promote_cli.main([
                "--candidate-id", args.candidate_id,
                "--queue-path", str(queue_path),
                "--paper-test-ledger", str(ledger_path),
                "--registry-audit-log", str(audit_log),
                "--packet-path", str(packet_path),
                "--operator-confirmation", _CONFIRMATION,
            ])
        print(promote_stdout.getvalue())
        if promote_rc != 0:
            _audit("promotion_packet", "fail", rc=promote_rc)
            print(
                f"FAILED (promotion-packet stage): rc={promote_rc}",
                file=sys.stderr,
            )
            return 7
        _audit("promotion_packet", "ok", packet=str(packet_path))

    real_post = (
        sum(1 for _ in _REAL_REGISTRY_ROOT.rglob("*.json"))
        if _REAL_REGISTRY_ROOT.exists() else 0
    )
    if real_pre != real_post:
        _audit("demo_complete", "fail",
               reason=f"real_registry_json_count_drift:{real_pre}->{real_post}")
        print(
            f"FAILED (red-line breach): real registry json count "
            f"{real_pre} → {real_post}",
            file=sys.stderr,
        )
        return 8

    _audit("demo_complete", "ok",
           real_registry_json_count=real_pre,
           workspace=str(workspace))
    print("== Demo complete. ==")
    print(f"  workspace: {workspace}")
    print(f"  real-registry json count unchanged: {real_pre}")
    print("  status: NOT LIVE / NOT APPROVED / NOT DEPLOYED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

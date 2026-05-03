"""End-to-end XAUUSD self-evolution dry-run orchestrator.

REPORT-ONLY. NEVER touches live EA, ``rule_engine.py`` (red-line), or
``policy_registry/approved/``. Every artefact is written under
``--output-dir`` (default ``tmp/xauusd_evolution_dry_run/<UTC ISO>``).

XAUUSD-ONLY by hard assertion. ``--symbol`` other than ``XAUUSD`` is
rejected before any IO.

Stages (every stage is wrapped in graceful degradation — missing
inputs surface as DEGRADED markers in the final report, not a
crash):

    1.  XAUUSD-only assertion
    2.  Health-check pre-flight (with safe auto-recovery)
    3.  Load real XAUUSD H1 / H4 / D1 bars from the data lake
    4.  Walk-forward statistics over real H1 (PnL / MaxDD / WinRate / Sharpe)
    5.  Regime detection
    6.  Anomaly shield check
    7.  Timeframe consensus
    8.  Adaptive stops
    9.  Stress test against a neutral demo proposal
    10. Recommend CLI (calibrator + explainability + fingerprint enabled)
    11. Fingerprint chain verify
    12. Registry audit summary
    13. Approval checklist (dry-run, never approves)
    14. Final XAUUSD report (markdown) + JSON snapshot

Exit codes:

    0 — all stages OK or DEGRADED (report still produced)
    2 — XAUUSD-only constraint violated
    3 — health check returned CRITICAL/DOWN even after auto-recovery
    4 — workspace lands under a forbidden parent (production registry)
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import statistics
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Constants — XAUUSD-only by design.
# ---------------------------------------------------------------------------

SYMBOL = "XAUUSD"
DEFAULT_LOOKBACK_DAYS = 365


# ---------------------------------------------------------------------------
# Path helpers (shared with the rest of the sidecar).
# ---------------------------------------------------------------------------


def _ai_smc_home() -> Path:
    raw = os.environ.get("AI_SMC_HOME")
    if raw:
        return Path(raw).expanduser()
    return Path(__file__).resolve().parents[1]


def _hedgerock_home() -> Path:
    raw = os.environ.get("HEDGEROCK_HOME")
    if raw:
        return Path(raw).expanduser()
    return Path.home() / "HedgeRock"


_REAL_REGISTRY_ROOT = _hedgerock_home() / "policy_registry"
_FORBIDDEN_OUTPUT_PARENTS = (
    _REAL_REGISTRY_ROOT,
    _ai_smc_home() / "config",
)


def _assert_output_dir_safe(output_dir: Path) -> None:
    abs_out = output_dir.resolve()
    for parent in _FORBIDDEN_OUTPUT_PARENTS:
        if parent in abs_out.parents or abs_out == parent:
            raise ValueError(
                f"output dir lands under a forbidden location: "
                f"{output_dir!s}"
            )


def _assert_xauusd_only(symbol: str) -> None:
    if symbol != SYMBOL:
        raise ValueError(
            f"this orchestrator is XAUUSD-only; got symbol={symbol!r}"
        )


# ---------------------------------------------------------------------------
# Per-stage result envelope.
# ---------------------------------------------------------------------------


@dataclass
class StageResult:
    name: str
    status: str  # "OK" | "DEGRADED" | "FAILED"
    details: dict[str, Any] = field(default_factory=dict)
    note: str = ""


def _ok(name: str, **details: Any) -> StageResult:
    return StageResult(name=name, status="OK", details=dict(details))


def _degraded(name: str, note: str, **details: Any) -> StageResult:
    return StageResult(
        name=name, status="DEGRADED", note=note, details=dict(details),
    )


# ---------------------------------------------------------------------------
# Real XAUUSD bars loader.
# ---------------------------------------------------------------------------


def _bars_from_lake(
    *,
    timeframe: str,
    start: datetime,
    end: datetime,
    lake_root: Path,
) -> list[dict]:
    """Return OHLC dicts ordered by ts. Empty list on any read error."""
    try:
        from smc.data.lake import ForexDataLake
        from smc.data.schemas import Timeframe
    except Exception:  # pragma: no cover — defensive
        return []
    try:
        lake = ForexDataLake(lake_root)
        tf = Timeframe(timeframe)
        df = lake.query(SYMBOL, tf, start=start, end=end)
    except Exception:
        return []
    if df.is_empty():
        return []
    cols = {"open", "high", "low", "close"}
    if not cols.issubset(set(df.columns)):
        return []
    rows = df.select(["ts", "open", "high", "low", "close"]).to_dicts()
    out: list[dict] = []
    for r in rows:
        out.append({
            "ts": r["ts"].isoformat() if r["ts"] is not None else None,
            "open": float(r["open"]),
            "high": float(r["high"]),
            "low": float(r["low"]),
            "close": float(r["close"]),
        })
    return out


# ---------------------------------------------------------------------------
# Walk-forward statistics from real H1 bars.
#
# We deliberately keep this simple and dependency-light — a long-only
# bar-to-bar baseline whose only purpose is to surface real, reproducible
# numbers (PnL / MaxDD / WinRate / Sharpe) computed against the actual
# XAUUSD price history. NOT a trading strategy.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WalkForwardStats:
    n_bars: int
    pnl_pct: float
    max_drawdown_pct: float
    win_rate: float
    sharpe_annualised: float
    bars_per_year_assumed: int
    window_start: str
    window_end: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_bars": self.n_bars,
            "pnl_pct": self.pnl_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "win_rate": self.win_rate,
            "sharpe_annualised": self.sharpe_annualised,
            "bars_per_year_assumed": self.bars_per_year_assumed,
            "window_start": self.window_start,
            "window_end": self.window_end,
        }


_BARS_PER_YEAR_H1 = 24 * 365


def _walk_forward_stats(bars: list[dict]) -> WalkForwardStats | None:
    if len(bars) < 24:
        return None
    closes = [b["close"] for b in bars if b.get("close") is not None]
    if len(closes) < 24:
        return None

    rets: list[float] = []
    for i in range(1, len(closes)):
        prev = closes[i - 1]
        if prev <= 0:
            continue
        rets.append((closes[i] - prev) / prev)
    if not rets:
        return None

    pnl_pct = (closes[-1] / closes[0] - 1.0) * 100.0
    wins = sum(1 for r in rets if r > 0)
    win_rate = wins / len(rets)

    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for r in rets:
        equity *= (1.0 + r)
        peak = max(peak, equity)
        dd = (equity / peak) - 1.0
        max_dd = min(max_dd, dd)
    max_drawdown_pct = max_dd * 100.0

    if len(rets) >= 2:
        mu = statistics.mean(rets)
        sigma = statistics.pstdev(rets)
        sharpe = (mu / sigma) * math.sqrt(_BARS_PER_YEAR_H1) if sigma > 0 else 0.0
    else:
        sharpe = 0.0

    return WalkForwardStats(
        n_bars=len(bars),
        pnl_pct=round(pnl_pct, 4),
        max_drawdown_pct=round(max_drawdown_pct, 4),
        win_rate=round(win_rate, 4),
        sharpe_annualised=round(sharpe, 4),
        bars_per_year_assumed=_BARS_PER_YEAR_H1,
        window_start=bars[0].get("ts") or "",
        window_end=bars[-1].get("ts") or "",
    )


# ---------------------------------------------------------------------------
# Approval checklist (dry-run).
# ---------------------------------------------------------------------------


def _approval_checklist(
    *,
    health_ok: bool,
    walk_forward_ok: bool,
    regime_ok: bool,
    anomaly_state: Any | None,
    consensus: Any | None,
    stress_all_passed: bool,
    fingerprint_ok: bool,
    registry_present: bool,
) -> list[tuple[str, str]]:
    """Return (item, status) tuples — status is PASS / WAIT / SKIP."""
    rows: list[tuple[str, str]] = []
    rows.append(("health-check pre-flight", "PASS" if health_ok else "WAIT"))
    rows.append((
        "walk-forward stats produced",
        "PASS" if walk_forward_ok else "WAIT",
    ))
    rows.append((
        "regime detection produced snapshot",
        "PASS" if regime_ok else "WAIT",
    ))
    if anomaly_state is None:
        rows.append(("anomaly level below halt", "WAIT"))
    else:
        try:
            level = anomaly_state.level.value
        except Exception:
            level = str(anomaly_state)
        rows.append((
            f"anomaly level = {level}",
            "PASS" if level in ("NORMAL", "ELEVATED") else "WAIT",
        ))
    if consensus is None:
        rows.append(("multi-TF consensus reached", "WAIT"))
    else:
        rows.append((
            f"multi-TF consensus={consensus.consensus_score:.2f}",
            "PASS" if consensus.can_recommend else "WAIT",
        ))
    rows.append((
        "stress scenarios all survived",
        "PASS" if stress_all_passed else "WAIT",
    ))
    rows.append(("fingerprint chain verified", "PASS" if fingerprint_ok else "WAIT"))
    rows.append((
        "operator registry present",
        "PASS" if registry_present else "SKIP (DEGRADED — re-point flag)",
    ))
    rows.append(("operator promotion confirmation", "WAIT (DRY-RUN, never asks)"))
    return rows


# ---------------------------------------------------------------------------
# Markdown report renderer.
# ---------------------------------------------------------------------------


def _render_report(
    *,
    output_dir: Path,
    symbol: str,
    stages: list[StageResult],
    walk_forward: WalkForwardStats | None,
    regime: Any | None,
    anomaly: Any | None,
    consensus: Any | None,
    stop: Any | None,
    stress_total: int,
    stress_survived: int,
    n_proposals: int,
    n_recommend: int,
    fingerprint_verify: dict[str, Any] | None,
    approval_rows: list[tuple[str, str]],
    registry_present: bool,
) -> str:
    lines: list[str] = []
    lines.append(f"# XAUUSD Evolution — Dry-Run Report ({symbol})")
    lines.append("")
    lines.append(
        f"_Generated at {datetime.now(timezone.utc).isoformat()} — "
        "REPORT-ONLY. NOT LIVE / NOT APPROVED / NOT DEPLOYED._"
    )
    lines.append("")

    lines.append("## 1. Stage status")
    lines.append("")
    lines.append("| Stage | Status | Note |")
    lines.append("|---|---|---|")
    for s in stages:
        lines.append(f"| {s.name} | {s.status} | {s.note} |")
    lines.append("")

    lines.append("## 2. Walk-forward statistics (real XAUUSD H1)")
    lines.append("")
    if walk_forward is not None:
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        lines.append(f"| Bars | {walk_forward.n_bars} |")
        lines.append(f"| Window | {walk_forward.window_start} → {walk_forward.window_end} |")
        lines.append(f"| PnL % (long-only baseline) | {walk_forward.pnl_pct:+.4f} |")
        lines.append(f"| Max drawdown % | {walk_forward.max_drawdown_pct:.4f} |")
        lines.append(f"| Win rate (per-bar) | {walk_forward.win_rate:.4f} |")
        lines.append(f"| Sharpe (annualised) | {walk_forward.sharpe_annualised:+.4f} |")
        lines.append(
            "| Annualisation factor (bars/yr) "
            f"| {walk_forward.bars_per_year_assumed} |"
        )
        lines.append("")
        lines.append(
            "*Baseline = long-only bar-to-bar reference on closes; surfaces "
            "the true price-history shape, not a trading strategy.*"
        )
    else:
        lines.append("_DEGRADED — no XAUUSD bars available in the lake._")
    lines.append("")

    lines.append("## 3. Regime + anomaly")
    lines.append("")
    if regime is not None:
        lines.append(
            f"- regime = `{regime.regime.value}` "
            f"(confidence = {regime.confidence:.2f})"
        )
    else:
        lines.append("- regime = _DEGRADED_")
    if anomaly is not None:
        lines.append(f"- anomaly level = `{anomaly.level.value}`")
    else:
        lines.append("- anomaly level = _DEGRADED_")
    lines.append("")

    lines.append("## 4. Multi-timeframe consensus + adaptive stop")
    lines.append("")
    if consensus is not None:
        lines.append(f"- session = `{consensus.active_session.value}`")
        lines.append(
            f"- consensus score = {consensus.consensus_score:.2f}; "
            f"can_recommend = {consensus.can_recommend}"
        )
    else:
        lines.append("- consensus = _DEGRADED_")
    if stop is not None:
        lines.append(
            f"- vol regime = `{stop.vol_regime.value}`; "
            f"atr_mult = {stop.atr_multiplier}; "
            f"position_scale = {stop.position_scale}"
        )
    else:
        lines.append("- adaptive stop = _DEGRADED_")
    lines.append("")

    lines.append("## 5. Stress-test survival matrix")
    lines.append("")
    lines.append(
        f"- scenarios = {stress_total}; "
        f"survived = {stress_survived}; "
        f"breached = {stress_total - stress_survived}"
    )
    lines.append("")

    lines.append("## 6. Recommendation pipeline")
    lines.append("")
    lines.append(f"- candidates produced: {n_proposals}")
    lines.append(f"- proposals with decision=RECOMMEND: {n_recommend}")
    lines.append("")

    lines.append("## 7. Fingerprint chain")
    lines.append("")
    if fingerprint_verify is None:
        lines.append("- _DEGRADED — fingerprint stage skipped or not produced._")
    else:
        lines.append(f"- entries: {fingerprint_verify.get('n_entries', 0)}")
        lines.append(f"- verify ok: {fingerprint_verify.get('ok', False)}")
        if fingerprint_verify.get("first_break_reason"):
            lines.append(
                f"- first break: {fingerprint_verify['first_break_reason']}"
            )
    lines.append("")

    lines.append("## 8. Registry audit")
    lines.append("")
    lines.append(
        f"- HedgeRock policy_registry root: `{_REAL_REGISTRY_ROOT}`"
    )
    lines.append(
        "- registry present on this machine: "
        f"{'yes' if registry_present else 'no (DEGRADED — re-point flag)'}"
    )
    lines.append("")

    lines.append("## 9. Approval checklist (DRY-RUN)")
    lines.append("")
    lines.append("| Item | Status |")
    lines.append("|---|---|")
    for item, status in approval_rows:
        lines.append(f"| {item} | {status} |")
    lines.append("")

    lines.append("---")
    lines.append("**status: NOT LIVE / NOT APPROVED / NOT DEPLOYED**")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Recommend pipeline shim — calls the existing recommend CLI with a
# minimal evidence fixture so the chain (calibrator + explainability +
# fingerprint) runs end to end against the demo evidence skeleton.
# ---------------------------------------------------------------------------


_DEMO_AVAILABILITY = """
# Phase D-cont3-preflight (XAUUSD-only dry-run fixture)

## Year-replication summary

| Symbol | Year | Bars | trend_up | range@≥0.80 | breakout_signed_by_h4 | halt events |
|---|---|---|---|---|---|---|
| XAUUSD | 2024 | 5693 | +0.239% ±0.067 | +0.172% ±0.040 | -0.015% ±0.096 (CI∋0) | 1 |

## Action gate

```yaml
NO_STRATEGY_CHANGE: false
```
"""


def _seed_evidence(workspace: Path) -> tuple[Path, Path, Path, Path, Path]:
    fixture = workspace / "fixture"
    fixture.mkdir(parents=True, exist_ok=True)
    atlas = fixture / "atlas.md"
    atlas.write_text("# atlas (xauusd-dry-run fixture)\n", encoding="utf-8")
    avail = fixture / "availability.md"
    avail.write_text(_DEMO_AVAILABILITY, encoding="utf-8")
    wf = fixture / "wf.md"
    wf.write_text("# walk-forward (xauusd-dry-run fixture)\n", encoding="utf-8")
    bounds = fixture / "safety_bounds.yaml"  # absent on purpose — fires G6
    audit_log = fixture / "_audit.md"
    audit_log.write_text(
        "# Shadow-Artefact Registry Audit Log (xauusd-dry-run)\n\n(no incidents)\n",
        encoding="utf-8",
    )
    return atlas, avail, wf, bounds, audit_log


# ---------------------------------------------------------------------------
# Main orchestrator.
# ---------------------------------------------------------------------------


def run(
    *,
    output_dir: Path,
    symbol: str = SYMBOL,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    lake_root: Path | None = None,
    registry_root: Path | None = None,
) -> int:
    _assert_xauusd_only(symbol)
    output_dir = Path(output_dir).resolve()
    try:
        _assert_output_dir_safe(output_dir)
    except ValueError as e:
        print(f"FAILED (forbidden output): {e}", file=sys.stderr)
        return 4
    output_dir.mkdir(parents=True, exist_ok=True)

    lake_root = (lake_root or (_ai_smc_home() / "data" / "parquet")).resolve()
    registry_root = (registry_root or _REAL_REGISTRY_ROOT).resolve()

    stages: list[StageResult] = []

    # Stage 1 — XAUUSD-only assertion (already done above; record it).
    stages.append(_ok("xauusd_only", symbol=symbol))
    print(f"[1/13] XAUUSD-only assertion OK (symbol={symbol})")

    # Stage 2 — health check.
    print("[2/13] HEALTH-CHECK")
    from smc.hedgerock.evolution.health_check import (
        HealthStatus as _HS, auto_recover as _autorecover, diagnose as _diag,
    )
    health = _diag(
        registry_root=output_dir / "registry",
        queue_path=output_dir / "queue" / "shadow_test_queue.jsonl",
        ledger_path=output_dir / "ledger" / "paper_test_ledger.jsonl",
    )
    health_ok = True
    if health.overall in (_HS.CRITICAL, _HS.DOWN):
        rec = _autorecover(
            health,
            registry_root=output_dir / "registry",
            queue_path=output_dir / "queue" / "shadow_test_queue.jsonl",
            ledger_path=output_dir / "ledger" / "paper_test_ledger.jsonl",
        )
        if rec.post_status in (_HS.CRITICAL, _HS.DOWN):
            stages.append(StageResult(
                name="health_check", status="FAILED",
                details={"overall": rec.post_status.value},
                note="health check unrecoverable",
            ))
            print(f"  FAILED — overall={rec.post_status.value}", file=sys.stderr)
            return 3
        stages.append(_ok(
            "health_check",
            pre=health.overall.value, post=rec.post_status.value,
            recovered=len(rec.actions),
        ))
        print(f"  recovered: {health.overall.value} → {rec.post_status.value}")
    elif health.overall == _HS.DEGRADED:
        stages.append(_degraded(
            "health_check",
            note="DEGRADED (acceptable on fresh machine)",
            overall=health.overall.value,
        ))
        health_ok = False
        print(f"  DEGRADED — overall={health.overall.value}")
    else:
        stages.append(_ok("health_check", overall=health.overall.value))
        print(f"  OK — overall={health.overall.value}")

    # Stage 3 — load real XAUUSD bars.
    print(f"[3/13] LOAD REAL XAUUSD BARS (lake={lake_root})")
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days)
    h1_bars = _bars_from_lake(
        timeframe="H1", start=start, end=end, lake_root=lake_root,
    )
    h4_bars = _bars_from_lake(
        timeframe="H4", start=start, end=end, lake_root=lake_root,
    )
    d1_bars = _bars_from_lake(
        timeframe="D1", start=start - timedelta(days=lookback_days),
        end=end, lake_root=lake_root,
    )
    if not h1_bars:
        # Try a wider historical window — fresh checkouts may not have
        # bars in the trailing year, but the lake has 2020-2024.
        end_fallback = datetime(2025, 1, 1, tzinfo=timezone.utc)
        start_fallback = end_fallback - timedelta(days=lookback_days)
        h1_bars = _bars_from_lake(
            timeframe="H1", start=start_fallback, end=end_fallback,
            lake_root=lake_root,
        )
        h4_bars = _bars_from_lake(
            timeframe="H4", start=start_fallback, end=end_fallback,
            lake_root=lake_root,
        )
        d1_bars = _bars_from_lake(
            timeframe="D1",
            start=start_fallback - timedelta(days=lookback_days),
            end=end_fallback, lake_root=lake_root,
        )
    if h1_bars:
        stages.append(_ok(
            "load_bars",
            n_h1=len(h1_bars), n_h4=len(h4_bars), n_d1=len(d1_bars),
        ))
        print(
            f"  loaded H1={len(h1_bars)} H4={len(h4_bars)} D1={len(d1_bars)}"
        )
    else:
        stages.append(_degraded(
            "load_bars", note=f"lake at {lake_root} has no XAUUSD bars",
        ))
        print(f"  DEGRADED — no bars under {lake_root}")

    # Stage 4 — walk-forward stats.
    print("[4/13] WALK-FORWARD STATS")
    walk_forward = _walk_forward_stats(h1_bars) if h1_bars else None
    if walk_forward is not None:
        stages.append(_ok("walk_forward", **walk_forward.to_dict()))
        print(
            f"  PnL={walk_forward.pnl_pct:+.4f}% "
            f"MaxDD={walk_forward.max_drawdown_pct:.4f}% "
            f"WR={walk_forward.win_rate:.4f} "
            f"Sharpe={walk_forward.sharpe_annualised:+.4f}"
        )
    else:
        stages.append(_degraded(
            "walk_forward", note="insufficient bars for stats",
        ))
        print("  DEGRADED — insufficient bars")

    # Stage 5 — regime.
    print("[5/13] REGIME DETECTION")
    regime = None
    if h1_bars:
        try:
            from smc.hedgerock.evolution.regime_engine import RegimeDetector
            regime = RegimeDetector().detect(bars=h1_bars)
            stages.append(_ok(
                "regime",
                regime=regime.regime.value, confidence=regime.confidence,
            ))
            print(
                f"  regime={regime.regime.value} "
                f"confidence={regime.confidence:.2f}"
            )
        except Exception as e:
            stages.append(_degraded(
                "regime", note=f"detector raised: {e!r}",
            ))
            print(f"  DEGRADED — {e!r}")
    else:
        stages.append(_degraded("regime", note="no bars"))

    # Stage 6 — anomaly.
    print("[6/13] ANOMALY SHIELD")
    anomaly = None
    if h1_bars:
        try:
            from smc.hedgerock.evolution.anomaly_shield import (
                AnomalyDetector, shield_action,
            )
            anomaly = AnomalyDetector().detect(bars=h1_bars)
            action = shield_action(anomaly)
            stages.append(_ok(
                "anomaly",
                level=anomaly.level.value,
                new_candidates_allowed=action.new_candidates_allowed,
            ))
            print(
                f"  level={anomaly.level.value} "
                f"new_candidates_allowed={action.new_candidates_allowed}"
            )
        except Exception as e:
            stages.append(_degraded("anomaly", note=f"detector raised: {e!r}"))
            print(f"  DEGRADED — {e!r}")
    else:
        stages.append(_degraded("anomaly", note="no bars"))

    # Stage 7 — multi-TF consensus.
    print("[7/13] TIMEFRAME CONSENSUS")
    consensus = None
    try:
        from smc.hedgerock.evolution.multi_timeframe_state import (
            compute_consensus,
        )
        consensus = compute_consensus(
            d1_bars=d1_bars or None,
            h4_bars=h4_bars or None,
            h1_bars=h1_bars or None,
        )
        stages.append(_ok(
            "consensus",
            session=consensus.active_session.value,
            score=consensus.consensus_score,
            can_recommend=consensus.can_recommend,
        ))
        print(
            f"  session={consensus.active_session.value} "
            f"score={consensus.consensus_score:.2f} "
            f"can_recommend={consensus.can_recommend}"
        )
    except Exception as e:
        stages.append(_degraded("consensus", note=f"raised: {e!r}"))
        print(f"  DEGRADED — {e!r}")

    # Stage 8 — adaptive stops.
    print("[8/13] ADAPTIVE STOPS")
    stop = None
    if h1_bars:
        try:
            from smc.hedgerock.evolution.adaptive_stops import (
                compute_stop_recommendation,
            )
            stop = compute_stop_recommendation(bars=h1_bars)
            stages.append(_ok(
                "adaptive_stop",
                vol_regime=stop.vol_regime.value,
                atr_multiplier=stop.atr_multiplier,
                position_scale=stop.position_scale,
            ))
            print(
                f"  vol_regime={stop.vol_regime.value} "
                f"atr_mult={stop.atr_multiplier} "
                f"position_scale={stop.position_scale}"
            )
        except Exception as e:
            stages.append(_degraded("adaptive_stop", note=f"raised: {e!r}"))
            print(f"  DEGRADED — {e!r}")
    else:
        stages.append(_degraded("adaptive_stop", note="no bars"))

    # Stage 9 — stress test against a neutral demo proposal.
    print("[9/13] STRESS TEST (neutral demo proposal)")
    stress_total = 0
    stress_survived = 0
    stress_all_passed = False
    try:
        from smc.hedgerock.evolution.candidate_generator import (
            CandidateProposal as _CP,
            DECISION_RECOMMEND as _DR,
            get_live_parameter_snapshot as _live_snapshot,
        )
        from smc.hedgerock.evolution.stress_tester import (
            StressTester, VERDICT_BREACHED,
        )
        demo = _CP(
            candidate_id="xauusd-dry-run-probe",
            parameter_target="smc.hedgerock.rule_engine._CONFIDENCE_OBSERVE_FLOOR",
            parameter_class="confidence_threshold_observe",
            baseline_value=0.55,
            proposed_value=0.55,
            triggered_by=("dry_run",),
            expected_improvement="dry-run: neutral probe",
            risks=(),
            next_validation=(),
            decision=_DR,
            decision_reason="",
        )
        results = StressTester().test_candidate(demo, _live_snapshot())
        stress_total = len(results)
        stress_survived = sum(
            1 for r in results if r.verdict != VERDICT_BREACHED
        )
        stress_all_passed = stress_survived == stress_total
        stages.append(_ok(
            "stress_test",
            scenarios=stress_total, survived=stress_survived,
            all_passed=stress_all_passed,
        ))
        print(f"  scenarios={stress_total} survived={stress_survived}")
    except Exception as e:
        stages.append(_degraded("stress_test", note=f"raised: {e!r}"))
        print(f"  DEGRADED — {e!r}")

    # Stage 10 — recommend CLI (with calibrator + explainability + fingerprint).
    print("[10/13] RECOMMEND (calibrator + explainability + fingerprint)")
    atlas, avail, wf, bounds, audit_log = _seed_evidence(output_dir)
    report_path = output_dir / "report" / "phase-d-evolution-report.md"
    rec_path = output_dir / "report" / "hedgerock-evolution-recommendation.md"
    fingerprint_path = output_dir / "fingerprint" / "chain.jsonl"
    calibrator_path = output_dir / "calibrator" / "state.json"
    # Reserve the dirs eagerly so operators can find the slot even if a
    # particular invocation does not persist anything to it.
    fingerprint_path.parent.mkdir(parents=True, exist_ok=True)
    calibrator_path.parent.mkdir(parents=True, exist_ok=True)

    n_proposals = 0
    n_recommend = 0
    try:
        _this_dir = Path(__file__).resolve().parent
        sys.path.insert(0, str(_this_dir))
        try:
            import hedgerock_evolution_recommend as recommend_cli  # type: ignore
        finally:
            sys.path.pop(0)
        rc_stdout = io.StringIO()
        with redirect_stdout(rc_stdout):
            rc = recommend_cli.main([
                "--atlas-report", str(atlas),
                "--data-availability-report", str(avail),
                "--walk-forward-report", str(wf),
                "--safety-bounds", str(bounds),
                "--registry-root", str(output_dir / "registry"),
                "--report-path", str(report_path),
                "--recommendation-path", str(rec_path),
                "--registry-audit-log", str(audit_log),
                "--explainability",
                "--fingerprint", str(fingerprint_path),
                "--calibrator-state", str(calibrator_path),
            ])
        snap = output_dir / "report" / "candidate_proposals.json"
        if rc == 0 and snap.exists():
            data = json.loads(snap.read_text(encoding="utf-8"))
            proposals = data.get("proposals", [])
            n_proposals = len(proposals)
            n_recommend = sum(
                1 for p in proposals if p.get("decision") == "RECOMMEND"
            )
            stages.append(_ok(
                "recommend",
                n_proposals=n_proposals, n_recommend=n_recommend,
            ))
            print(f"  proposals={n_proposals} recommend={n_recommend}")
        else:
            stages.append(_degraded(
                "recommend", note=f"rc={rc} or snapshot missing",
            ))
            print(f"  DEGRADED — rc={rc}")
    except Exception as e:
        stages.append(_degraded("recommend", note=f"raised: {e!r}"))
        print(f"  DEGRADED — {e!r}")

    # Stage 11 — fingerprint verify.
    print("[11/13] FINGERPRINT VERIFY")
    fingerprint_verify: dict[str, Any] | None = None
    if fingerprint_path.exists():
        try:
            from smc.hedgerock.evolution.fingerprint import verify_chain
            v = verify_chain(fingerprint_path)
            fingerprint_verify = {
                "ok": v.ok,
                "n_entries": v.n_entries,
                "first_break_index": v.first_break_index,
                "first_break_reason": v.first_break_reason,
            }
            if v.ok:
                stages.append(_ok(
                    "fingerprint_verify",
                    ok=True, n_entries=v.n_entries,
                ))
                print(f"  OK — entries={v.n_entries}")
            else:
                stages.append(_degraded(
                    "fingerprint_verify",
                    note=f"break: {v.first_break_reason}",
                ))
                print(f"  DEGRADED — {v.first_break_reason}")
        except Exception as e:
            stages.append(_degraded(
                "fingerprint_verify", note=f"raised: {e!r}",
            ))
            print(f"  DEGRADED — {e!r}")
    else:
        stages.append(_degraded(
            "fingerprint_verify", note=f"chain not produced at {fingerprint_path}",
        ))
        print("  DEGRADED — chain not produced")

    # Stage 12 — registry audit summary.
    print("[12/13] REGISTRY AUDIT")
    registry_present = registry_root.exists()
    if registry_present:
        json_count = sum(1 for _ in registry_root.rglob("*.json"))
        stages.append(_ok(
            "registry_audit",
            registry_root=str(registry_root), json_count=json_count,
        ))
        print(f"  registry present at {registry_root} ({json_count} JSON)")
    else:
        stages.append(_degraded(
            "registry_audit",
            note=f"registry absent at {registry_root}",
        ))
        print(f"  DEGRADED — registry absent at {registry_root}")

    # Stage 13 — approval checklist.
    print("[13/13] APPROVAL CHECKLIST (DRY-RUN)")
    approval_rows = _approval_checklist(
        health_ok=health_ok,
        walk_forward_ok=walk_forward is not None,
        regime_ok=regime is not None,
        anomaly_state=anomaly,
        consensus=consensus,
        stress_all_passed=stress_all_passed,
        fingerprint_ok=bool(fingerprint_verify and fingerprint_verify.get("ok")),
        registry_present=registry_present,
    )
    stages.append(_ok("approval_checklist", n_items=len(approval_rows)))

    # Final report.
    body = _render_report(
        output_dir=output_dir,
        symbol=symbol,
        stages=stages,
        walk_forward=walk_forward,
        regime=regime,
        anomaly=anomaly,
        consensus=consensus,
        stop=stop,
        stress_total=stress_total,
        stress_survived=stress_survived,
        n_proposals=n_proposals,
        n_recommend=n_recommend,
        fingerprint_verify=fingerprint_verify,
        approval_rows=approval_rows,
        registry_present=registry_present,
    )
    final_md = output_dir / "xauusd_dry_run_report.md"
    final_md.write_text(body, encoding="utf-8")
    snap_path = output_dir / "xauusd_dry_run_snapshot.json"
    snap_path.write_text(
        json.dumps(
            {
                "symbol": symbol,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "stages": [
                    {
                        "name": s.name, "status": s.status,
                        "note": s.note, "details": s.details,
                    } for s in stages
                ],
                "walk_forward": walk_forward.to_dict() if walk_forward else None,
                "stress_total": stress_total,
                "stress_survived": stress_survived,
                "n_proposals": n_proposals,
                "n_recommend": n_recommend,
                "fingerprint_verify": fingerprint_verify,
                "registry_present": registry_present,
                "approval_rows": [
                    {"item": i, "status": st} for i, st in approval_rows
                ],
            },
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\n== XAUUSD dry-run complete ==")
    print(f"  report:   {final_md}")
    print(f"  snapshot: {snap_path}")
    print(f"  status: NOT LIVE / NOT APPROVED / NOT DEPLOYED")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    default_out = (
        _ai_smc_home() / "tmp" / "xauusd_evolution_dry_run"
        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=default_out,
        help="Where to write the report + snapshot. "
             "Defaults to tmp/xauusd_evolution_dry_run/<UTC ISO>/.",
    )
    parser.add_argument(
        "--symbol", default=SYMBOL,
        help="MUST be XAUUSD; the orchestrator refuses everything else.",
    )
    parser.add_argument(
        "--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS,
        help="How many days of bars to load (rolling window).",
    )
    parser.add_argument(
        "--lake-root", type=Path, default=None,
        help="Path to the parquet data lake root (defaults to "
             "$AI_SMC_HOME/data/parquet).",
    )
    parser.add_argument(
        "--registry-root", type=Path, default=None,
        help="HedgeRock policy_registry root (defaults to "
             "$HEDGEROCK_HOME/policy_registry).",
    )
    args = parser.parse_args(argv)

    try:
        return run(
            output_dir=Path(args.output_dir),
            symbol=args.symbol,
            lookback_days=args.lookback_days,
            lake_root=args.lake_root,
            registry_root=args.registry_root,
        )
    except ValueError as e:
        # XAUUSD-only assertion failure.
        print(f"FAILED: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
